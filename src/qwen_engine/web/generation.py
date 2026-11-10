"""Opt-in serial web execution over the same verified engine as the CLI.

No ML imports at startup. Paths, host attestation and executor are operator
owned; the browser supplies only the shared editable config envelope.
"""
from __future__ import annotations

import gc
import json
import re
import secrets
import sys
import threading
import time
import traceback
from pathlib import Path
from dataclasses import replace

from qwen_engine.config import ConfigError
from qwen_engine.local_bundle import VerificationReusePolicy
from qwen_engine.core import generate
from qwen_engine.errors import GenerationCancelled
from qwen_engine.jobs import (
    JobCancelled, JobController, JobError, JobHistoryFullError,
    JobQueueLimitError, release_cyclic_references,
)
from qwen_engine.owners import LOCAL_OWNER
from qwen_engine.web import remote as _remote
from qwen_engine.log import PhaseLogger
from qwen_engine.web.api import (
    ApiError, ApiResponse, WebRuntime, build_config, editable_config_dict,
    models_missing, parse_request_json, startup_blockers,
)

JOB_SCHEMA = "qwen_engine.web_job.v1"
JOBS_SCHEMA = "qwen_engine.web_jobs.v1"
#: Closed map from executor failure codes to public web error codes (all
#: fixed, prompt-free literals). ``out_of_gpu_memory`` is a classified CUDA
#: out-of-memory at the executor boundary instead of the generic
#: ``execution_failed``; an unknown job-layer code passes through unchanged
#: (a new engine code must never be silently hidden from the UI).
PUBLIC_ERROR_CODES = {
    "execution_failed": "execution_failed",
    "worker_stopped": "worker_stopped",
    "invalid_cancellation_acknowledgment": "invalid_cancellation_acknowledgment",
    "out_of_gpu_memory": "out_of_gpu_memory",
}


def release_gpu_cache():
    """Return PyTorch's cached-but-unused CUDA memory to the driver.

    Jobs run in this process and each one loads its own models, so after a
    job the caching allocator holds ~9-15 GiB that nothing is using; other
    programs cannot have it until the server exits. Measured on the V100
    (2026-10-26, Turbo 1024 x3): the release takes ~0.17 s and the next job's
    load phase is unchanged (7.9-8.0 s kept vs 7.8-8.0 s released), so it
    runs after every job rather than on a timer. A no-op when torch was never
    imported or CUDA never initialised (CPU tests, planning-only servers).
    """
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return
    gc.collect()
    torch.cuda.empty_cache()


class WebGeneration:
    def __init__(self, runtime: WebRuntime, *, trust_unbounded_host=False, executor=None,
                 performance=False, memory_release=None, reuse_verified_assets=None,
                 gpu_cache_release=release_gpu_cache, gpu_lock=None):
        if startup_blockers(runtime):
            # `models_missing` deliberately does NOT refuse construction: the
            # first-run page is how those bytes arrive, and a server that would
            # not start on an empty models dir could never offer it. submit()
            # refuses a job until the download has finished.
            raise ValueError("generation runtime is incomplete or disabled")
        self.runtime = runtime
        if type(performance) is not bool:
            raise ValueError("performance must be boolean")
        self.performance = performance
        self.trust_unbounded_host = trust_unbounded_host
        # Opt-in verification reuse (server flag --reuse-verified-assets):
        # None (default) content-verifies every artifact on every job; a
        # positive-finite age (seconds) or a ready-made policy lets a job
        # reuse a recorded SUCCESSFUL verification while every artifact's
        # stat fingerprint is unchanged — strictly weaker than re-hashing.
        if reuse_verified_assets is None:
            self.reuse_policy = None
        elif isinstance(reuse_verified_assets, VerificationReusePolicy):
            self.reuse_policy = reuse_verified_assets
        else:
            self.reuse_policy = VerificationReusePolicy(max_age_s=reuse_verified_assets)
        self._gpu_cache_release = gpu_cache_release
        # The GPU is ONE resource, shared with the official prompt enhancer
        # (9.5 GiB of the 16 GiB V100, loaded by a llama-server child that this
        # process owns). The server creates the lock and hands the same object
        # to the rewrite service; without it the enhancer and an image job
        # would load onto the card together and one of them would die of it.
        # A job WAITS here — a rewrite takes tens of seconds, which is honest
        # queueing — while a rewrite asked to wait refuses instead, so a
        # running image is never delayed by a text request.
        self.gpu_lock = threading.Lock() if gpu_lock is None else gpu_lock
        self._lock = threading.Lock()
        self._metrics = {}
        self._executor = executor or self._execute
        if memory_release is None:
            memory_release = self._release_after_failed_job
        self.controller = JobController(runtime.output_dir, self._run, memory_release=memory_release)

    def _release_after_failed_job(self):
        """The controller's release for a failed or cancelled job.

        The release in _run's ``finally`` runs while the failure is still being
        raised, and its traceback holds the job's tensors, so it frees almost
        nothing: a 2K out-of-memory left 15.8 GiB on the card with the server
        idle (2026-10-27). The controller calls this once the exception is gone,
        so the tensors are collectable and the cache goes back to the driver
        now rather than never.
        """
        release_cyclic_references()
        self._gpu_cache_release()

    def _execute(self, context):
        # Identical asset verification, adapter and host policy to --run,
        # including the operator's opt-in reuse policy (None = full
        # content verification per job, the default).
        from qwen_engine.cli import _make_adapter, _verify_assets

        context.check_cancelled()
        assets = _verify_assets(context.config, reuse_policy=self.reuse_policy)
        context.check_cancelled()
        adapter = _make_adapter(trust_unbounded_host=self.trust_unbounded_host)
        progress_binder = getattr(adapter, "set_progress_callback", None)
        if progress_binder is not None:
            progress_binder(lambda completed, total: self.report_steps(context.job_id, completed, total))

        def sink(line):
            record = json.loads(line)
            if record.get("event") == "phase_end" and record.get("elapsed_ms") is not None:
                with self._lock:
                    self._metrics[context.job_id]["phase_ms"][record["phase"]] = record["elapsed_ms"]

        return generate(context.config, adapter, assets.resolved, PhaseLogger(sink=sink),
                        cancel_event=context._cancel, phase_callback=context.report_phase)

    def _run(self, context):
        started = time.monotonic()
        succeeded = False
        with self._lock:
            self._metrics[context.job_id] = {"started": started, "elapsed_ms": None,
                                             "phase_ms": {}, "completed_steps": 0,
                                             "total_steps": context.config.steps}
        # Taken after the metrics bookkeeping and held across the cache release
        # below, so the memory is back with the driver before the next holder
        # (the following job, or the enhancer's llama-server) starts loading.
        # Nothing else in this method may hold it, and nothing else does: a job
        # that is queued on a rewrite has not started yet.
        with self.gpu_lock:
            try:
                if self.performance:
                    from qwen_engine.performance import PerformanceRecorder, stage
                    recorder = PerformanceRecorder(context.config)
                    with (Path(context.config.output_dir) / "performance.json").open("x", encoding="utf-8") as handle:
                        status = "failed"
                        try:
                            with recorder.activate(), stage("total_job"):
                                result = self._executor(context)
                            status = "succeeded"
                            succeeded = True
                            return result
                        finally:
                            recorder.write(handle, status)
                result = self._executor(context)
                succeeded = True
                return result
            except (JobCancelled, GenerationCancelled):
                raise
            except Exception:
                print(f"qwen-web job {context.job_id} failed:", file=sys.stderr)
                traceback.print_exc()
                raise
            finally:
                with self._lock:
                    self._metrics[context.job_id]["elapsed_ms"] = (time.monotonic() - started) * 1000
                # After every outcome, outside the job's elapsed time. A failed
                # release is reported but never replaces the job's own result.
                try:
                    self._gpu_cache_release()
                except Exception:
                    print(f"qwen-web job {context.job_id}: GPU cache release failed:",
                          file=sys.stderr)
                    traceback.print_exc()
                # C3 history cap: only a SUCCESS may evict — a failed job never
                # costs the person an older picture. Remote owners are exactly
                # the non-local ones, and each may hold one live job at a time,
                # so this settles that owner's history with no cross-job race.
                if succeeded and context.owner != LOCAL_OWNER:
                    # `pending=1`: this job is still RUNNING as far as the
                    # controller knows (the worker records its terminal state
                    # only after this method returns), so counting terminal
                    # records alone would evict one job LATE and let the
                    # person keep cap+1 pictures.
                    self._reap_history(context.owner, _remote.REMOTE_HISTORY_CAP,
                                       pending=1)

    def _reap_history(self, owner, cap, *, pending=0):
        """Forget this owner's oldest finished job(s) over ``cap``, files too.

        ``pending`` counts successful jobs about to become terminal that the
        controller does not yet record as such (the one that just succeeded).

        Deletion goes through the controller's own forget path, which is the
        directory-identity/parent-fd safe delete (never outside the root,
        never through a symlink). A refusal is reported and kept: the person
        keeps a picture rather than the server losing a file it cannot prove
        it owns.
        """
        finished = [record for record in self.controller.snapshots(owner=owner)
                    if record.terminal]
        for record in finished[:max(0, len(finished) + pending - cap)]:
            try:
                self.controller.forget(record.job_id, delete_files=True, owner=owner)
            except (JobError, OSError):
                print(f"qwen-web job {record.job_id}: history cap could not "
                      "delete the oldest entry", file=sys.stderr)
                continue
            with self._lock:
                self._metrics.pop(record.job_id, None)

    def report_steps(self, job_id, completed, total):
        """Actual completed sampler steps; never a wall-clock estimate."""
        with self._lock:
            metric = self._metrics[job_id]
            if (type(completed) is not int or type(total) is not int
                    or total != metric["total_steps"]
                    or not metric["completed_steps"] <= completed <= total):
                raise ValueError("invalid sampler progress")
            metric["completed_steps"] = completed

    def public_job(self, record, *, history=None):
        """One job document. ``history`` is the owner's occupancy block (C3),
        added only where a cap applies — the local owner has none, so their
        documents keep exactly the keys they had before."""
        with self._lock:
            metric = self._metrics.get(record.job_id)
            elapsed = None
            phases = {}
            completed = 0
            if metric:
                elapsed = metric["elapsed_ms"]
                if elapsed is None:
                    elapsed = (time.monotonic() - metric["started"]) * 1000
                phases = dict(metric["phase_ms"])
                completed = metric["completed_steps"]
        document = {
            "schema": JOB_SCHEMA, "job_id": record.job_id,
            "state": record.state.value, "phase": record.phase,
            "cancel_requested": record.cancel_requested,
            "config": editable_config_dict(record.config),
            "elapsed_ms": elapsed, "phase_ms": phases,
            "error_code": (PUBLIC_ERROR_CODES.get(record.error_code, record.error_code)
                           if record.error_code else None),
            # Prompt-free boundary release cost for terminal failed/cancelled
            # jobs (None for successes and never-executed cancellations); it
            # is recorded just after the terminal state becomes visible.
            "collect_ms": self.controller.collect_ms(record.job_id),
            "progress": {"completed_steps": completed, "total_steps": record.config.steps},
            "image_url": (f"/api/v1/jobs/{record.job_id}/image" if record.artifact else None),
            "image_filename": record.artifact.filename if record.artifact else None,
            # The recorded byte size of THIS job's own image (None when it has
            # no artifact): what the History tab names in its delete
            # confirmations so the user sees what "also delete the files"
            # means before ticking it. The record is the source; nothing is
            # stat'ed off disk here.
            "image_bytes": record.artifact.bytes if record.artifact else None,
            # C3: 0 = running, N = N queued ahead, None = no longer waiting.
            # Read from the controller's own pending order, so it is the order
            # the worker will take, not an estimate.
            "queue_position": self.controller.queue_position(record.job_id),
        }
        if history is not None:
            document["history"] = history
        return document

    def submit(self, body, *, owner=LOCAL_OWNER, allow=None,
               history_cap=None, queue_limit=None):
        """Accept one job for ``owner``.

        ``allow`` (C3 feature permissions), ``history_cap`` and
        ``queue_limit`` are the REMOTE user's terms; every one of them is
        ``None`` for the local owner, whose behaviour is exactly as before.
        Permissions are checked against the CONFIG this request built — never
        against what the browser says it will send — so hiding a control in
        the page is a courtesy and this is the rule.
        """
        if models_missing(self.runtime):
            # The page disables Generate for exactly this, and the server says
            # the same thing to a client that posts anyway: no job is queued to
            # fail inside the loader. A client that has just downloaded the
            # weights is unblocked the moment they are on disk — no restart.
            raise ApiError(503, "models_missing",
                           "the image model files are not downloaded on this "
                           "machine yet; open Settings → Models to fetch them")
        config = build_config(self.runtime, parse_request_json(body))
        if allow is not None:
            _remote.check_feature_gates(config, allow)
        if config.references:
            # Every id must resolve to a hash-verified blob BEFORE enqueue:
            # a job may not fail deep in the worker over something the server
            # can see now. Public message carries the ORDINAL only — never
            # the id, never the store path (design §2.4). A remote user's ids
            # must also be THEIRS, or the job cannot see the picture at all.
            self._require_references(config.references, owner=owner)
        from qwen_engine.adapters.qwen21_diffusers import _validate_policy

        try:
            _validate_policy(config)
        except ConfigError as exc:
            # Path values never cross the wire. Editable policy errors are safe.
            # The recipe gate is an editable policy rule too (turbo owns the
            # step count and refuses true-CFG / Q8), so those fields must be
            # named rather than dropped into an empty field list.
            editable_policy = ("quant", "offload", "attention_backend",
                               "steps", "true_cfg_scale", "recipe")
            fields = tuple((e.field, e.message) for e in exc.errors
                           if e.field in editable_policy)
            raise ApiError(422, "unsupported_generation_config",
                           "this configuration is not supported by the current adapter", fields) from None
        if config.latent_tokens > 4096 and not config.vae_tiled_decode:
            raise ApiError(422, "tiled_decode_required", "enable tiled VAE decode for images above 4096 latent tokens")
        # Choose a real seed once, before enqueueing, so History can reproduce it.
        config = replace(config, output_dir=None,
                         seed=config.seed if config.seed is not None else secrets.randbits(64))
        try:
            record = self.controller.submit(config, owner=owner, max_live=queue_limit)
        except JobQueueLimitError:
            raise ApiError(409, "queue_limit",
                           "you already have a job queued or running; wait for it "
                           "to finish before starting another") from None
        except JobHistoryFullError:
            raise ApiError(409, "job_submission_refused",
                           "History is full — delete old entries on the History tab "
                           "(Delete all history or Forget entry)") from None
        except (JobError, OSError):
            raise ApiError(409, "job_submission_refused", "job queue/history is full or its output storage is unavailable") from None
        return ApiResponse(202, self.public_job(
            record, history=None if history_cap is None
            else _remote.history_block(self.controller.finished_for(owner), history_cap)))

    def _require_references(self, references, *, owner=None):
        from qwen_engine import reference_store
        root = self.runtime.references_dir
        if root is None:  # unreachable via build_config pairing; belt and braces
            raise ApiError(422, "reference_unavailable",
                           "the server reference store is not configured")
        for ordinal, reference_id in enumerate(references, start=1):
            try:
                reference_store.read_blob(root, reference_id)
                if owner is not None and owner not in reference_store.owners_for(root, reference_id):
                    raise reference_store.ReferenceStoreError("reference id is not stored")
            except reference_store.ReferenceStoreError:
                # Another account's id is the SAME refusal an unknown one gets:
                # the ordinal is all the message may carry.
                raise ApiError(422, "reference_unavailable",
                               f"reference {ordinal} is not available in the "
                               "server reference store") from None

    def list_response(self, *, owner=None, history_cap=None):
        """This owner's job ids. ``owner=None`` means the whole history (the
        local owner's page, unchanged)."""
        records = self.controller.snapshots(owner=owner)
        document = {"schema": JOBS_SCHEMA, "jobs": [r.job_id for r in records]}
        if history_cap is not None:
            # The SAME count the status document and the submit warning use:
            # terminal jobs only, so a live entry never makes the page warn
            # about an eviction the reaper will not do.
            document["history"] = _remote.history_block(
                self.controller.finished_for(owner), history_cap)
        return ApiResponse(200, document)

    def job_response(self, job_id, *, action=None, owner=None, history_cap=None):
        try:
            if action == "cancel":
                record = self.controller.cancel(job_id, owner=owner)
            else:
                record = self.controller.get(job_id, owner=owner)
            history = None if history_cap is None else _remote.history_block(
                self.controller.finished_for(owner), history_cap)
            return ApiResponse(200, self.public_job(record, history=history))
        except (JobError, OSError):
            raise ApiError(404, "job_unavailable",
                           "job is unknown or cannot perform this action") from None

    #: The two job-deletion bodies the wire accepts: the plain forget, and
    #: the explicit file-delete opt-in. Anything else is a typed refusal.
    @staticmethod
    def _delete_files_flag(body):
        payload = parse_request_json(body)
        if payload == {}:
            return False
        if (type(payload) is not dict or set(payload) != {"delete_files"}
                or type(payload["delete_files"]) is not bool):
            raise ApiError(400, "invalid_envelope",
                           'job deletion bodies are {} or {"delete_files": true|false} '
                           'and nothing else')
        return payload["delete_files"]

    def forget_response(self, job_id, body, *, owner=None, history_cap=None):
        delete_files = self._delete_files_flag(body)
        if owner is not None and owner != LOCAL_OWNER:
            # C3: a forget that names somebody else's job (or no job at all)
            # gets the SAME fixed 404 a job read gets, so a remote id cannot
            # be confirmed by trying to delete it. The local port keeps its
            # unchanged 409 for an unknown or live entry.
            try:
                self.controller.get(job_id, owner=owner)
            except (JobError, OSError):
                raise ApiError(404, "job_unavailable",
                               "job is unknown or cannot perform this action") from None
        try:
            deleted = self.controller.forget(job_id, delete_files=delete_files, owner=owner)
        except (JobError, OSError):
            raise ApiError(409, "job_unavailable",
                           "job is unknown or cannot perform this action") from None
        with self._lock:
            self._metrics.pop(job_id, None)
        document = {"schema": JOBS_SCHEMA, "forgotten": job_id,
                    "deleted_files": deleted}
        if history_cap is not None:
            document["history"] = _remote.history_block(
                self.controller.finished_for(owner), history_cap)
        return ApiResponse(200, document)

    def forget_all_response(self, body, *, owner=None):
        """Clear ``owner``'s finished history (everyone's when ``None``)."""
        delete_files = self._delete_files_flag(body)
        try:
            forgotten, deleted, skipped = self.controller.forget_all(
                delete_files=delete_files, owner=owner)
        except (JobError, OSError):
            raise ApiError(409, "job_unavailable",
                           "the job history could not be cleared on this server") from None
        # Metrics die with the records they belong to; live jobs keep theirs.
        live = {record.job_id for record in self.controller.snapshots()}
        with self._lock:
            for job_id in [j for j in self._metrics if j not in live]:
                del self._metrics[job_id]
        return ApiResponse(200, {"schema": JOBS_SCHEMA, "forgotten": forgotten,
                                 "deleted_files": deleted,
                                 "skipped_live": skipped})

    def image(self, job_id, *, owner=None):
        try:
            data = self.controller.read_artifact(job_id, owner=owner)
            self.image_content_type(data)
            return data
        except (JobError, OSError):
            raise ApiError(404, "image_unavailable", "completed image is unavailable or has changed") from None

    def image_filename(self, job_id, *, owner=None):
        try:
            artifact = self.controller.get(job_id, owner=owner).artifact
            if artifact is None or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", artifact.filename) is None:
                raise JobError("unsafe image filename")
            return artifact.filename
        except JobError:
            raise ApiError(404, "image_unavailable", "completed image is unavailable") from None

    @staticmethod
    def image_content_type(data):
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "image/webp"
        raise JobError("not a supported image")

    def close(self):
        return self.controller.close(timeout_s=60)

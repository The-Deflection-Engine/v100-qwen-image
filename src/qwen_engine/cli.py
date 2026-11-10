"""The v1 CLI front-end (plan R1/R2, task 1.3) — argparse + exit-code contract.

Guarantees:
* ``--help``, ``--dry-run`` and plain import never touch torch/diffusers/
  transformers: the real adapter (not written yet) is the only heavy-import
  path, reached only by ``--run`` after asset verification.
* ``--dry-run`` prints deterministic, key-sorted, timestamp-free JSON.
* Expected errors print a single ``error:``-prefixed line (or per-field
  lines) and exit with the contract code — never a traceback; unexpected
  errors are logged with context (traceback into the log sink) and exit 1.

Exit codes: 0 ok · 2 invalid/malformed config · 3 memory refusal (reserved,
M5.3) · 4 missing/unverified asset · 5 hardware absent (reserved, M3) ·
1 unexpected.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from qwen_engine import __version__
from qwen_engine._jsonsafe import strict_json_loads
from qwen_engine._local_io import read_bounded_file
from qwen_engine.adapters.base import (
    ATTENTION_BACKEND_CHOICES,
    OFFLOAD_CHOICES,
    QUANT_CHOICES,
)
from qwen_engine.config import ConfigError, FieldError, GenerationConfig, Recipe
from qwen_engine.core import generate, plan, require_run_config
from qwen_engine.errors import (
    EngineError,
    ExitCode,
    HardwareAbsentError,
    MemoryGuardError,
)
from qwen_engine.log import PhaseLogger, file_sink
from qwen_engine.local_bundle import (TURBO_REFERENCE_PROFILE, LocalBundleError,
                                      VerificationReusePolicy,
                                      prime_resolution_memo,
                                      remember_verified_resolution,
                                      reusable_resolution)
from qwen_engine.manifest import ManifestError, load_manifest, manifest_status, resolve_manifest

_OVERRIDES = {
    "prompt": "prompt",
    "negative": "negative_prompt",
    "width": "width",
    "height": "height",
    "steps": "steps",
    "true_cfg_scale": "true_cfg_scale",
    "seed": "seed",
    "quant": "quant",
    "offload": "offload",
    "attention_backend": "attention_backend",
    "output_dir": "output_dir",
    "cache_dir": "cache_dir",
    "log_dir": "log_dir",
    "benchmarks_dir": "benchmarks_dir",
    "models_dir": "models_dir",
    "manifest": "manifest_path",
    "q8_manifest": "q8_manifest_path",
    "output_format": "output_format",
    "recipe": "recipe",
}
MAX_CONFIG_BYTES = 1 << 20


def build_parser() -> argparse.ArgumentParser:
    from qwen_engine.config import AttentionBackend, OffloadMode, Quant

    p = argparse.ArgumentParser(
        prog="qwen-engine",
        description=(
            "Qwen-Image-2.1 engine (V100 16GB/32GB). Verifies pinned model assets "
            "from models/manifest.json; never downloads."
        ),
    )
    p.add_argument("--version", action="version", version=f"qwen-engine {__version__}")
    p.add_argument("--config", type=Path, default=None,
                   help="JSON config file (fields = GenerationConfig)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the deterministic key-sorted plan JSON and exit 0")
    p.add_argument("--run", action="store_true",
                   help="actually generate (verifies manifest assets first; no downloads)")
    p.add_argument("--tiled-decode", action=argparse.BooleanOptionalAction, default=None,
                   help="VAE tiling: required at 2048x2048, where an untiled "
                        "decode does not fit; uses fp32 1024 px tiles with a "
                        "768 px stride. At 1024x1024 the default untiled decode "
                        "is cleaner (the transformer is released first)")
    p.add_argument("--attest-unbounded-host", action="store_true",
                   help="operator attestation that this process tree has no cgroup "
                        "memory limit, so the CPU encoder gate uses the freshly read "
                        "host MemAvailable alone (default: the gate refuses when the "
                        "cgroup scope cannot be proven)")
    p.add_argument("--calibration-record", type=Path, default=None, metavar="PATH",
                   help="opt-in measured-memory collector: sample device-wide NVML "
                        "and owned-process-tree peaks per phase and append exact "
                        "calibration observations to PATH (requires --run). A "
                        "mid-run measurement failure keeps the generated image but "
                        "exits 1 without writing a record")
    p.add_argument("--performance-report", type=Path, default=None, metavar="PATH",
                   help="opt-in synchronized substage metrics without prompt contents; "
                        "write a new JSON file (requires --run; separate from calibration)")
    # Field overrides (take precedence over --config):
    p.add_argument("--prompt")
    p.add_argument("--negative", default=None)
    p.add_argument("--width", type=int)
    from qwen_engine.presets import PRESETS
    p.add_argument("--preset", choices=tuple(PRESETS),
                   help="aspect/size preset; explicit dimensions override it; 2K enables tiled decode")
    p.add_argument("--height", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--true-cfg-scale", dest="true_cfg_scale", type=float, default=None,
                   help="true CFG scale (>= 1.0; guidance active only with a "
                        "negative prompt AND scale > 1)")
    p.add_argument("--seed", type=int)
    p.add_argument("--output-format", choices=("jpg", "png", "webp"),
                   help="saved image format (default: jpg; JPEG flattens alpha on white)")
    p.add_argument("--transparent", dest="transparent",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="native transparency: wraps the positive prompt with "
                        "the RGBA/transparency prompt and requires an "
                        "alpha-capable format (PNG/WEBP). With --transparent "
                        "and no explicit --output-format, a jpg/default "
                        "format switches to PNG automatically; an explicit "
                        "--output-format jpg is refused. --no-transparent "
                        "overrides a config-file true and keeps the format")
    p.add_argument("--quant", choices=QUANT_CHOICES)
    p.add_argument("--recipe", choices=("base", "turbo"), default=None,
                   help="weights/sampling recipe: 'turbo' is the Turbo "
                        "transformer + base encoder/VAE with the shipped "
                        "eight-step sigma schedule (steps must be 8, CFG 1, "
                        "fp16 only); default 'base'")
    p.add_argument("--offload", choices=OFFLOAD_CHOICES)
    p.add_argument("--attention-backend", dest="attention_backend",
                   choices=ATTENTION_BACKEND_CHOICES)
    for flag in ("output_dir", "cache_dir", "log_dir", "benchmarks_dir",
                 "models_dir"):
        p.add_argument(f"--{flag.replace('_', '-')}", dest=flag, default=None,
                       help=("encoder cache/scratch directory, required for --run "
                             "(e.g. cache; created automatically)"
                             if flag == "cache_dir" else
                             f"override config field {flag!r} (default: unset -> no writes)"))
    p.add_argument("--manifest", type=Path, default=None,
                   help="model manifest path override (default: unset -> run refuses)")
    p.add_argument("--q8-manifest", type=Path, default=None,
                   help="explicit verified Q8 GGUF sidecar manifest (required with --quant q8)")
    p.add_argument("--reference", dest="references", action="append", default=None,
                   metavar="PATH",
                   help="reference image file (repeatable, ordered, at most 10): the "
                        "bytes are copied verbatim into the reference store and the "
                        "run carries only content ids")
    p.add_argument("--references-dir", dest="references_dir", type=Path, default=None,
                   help="reference store directory (default: <cache_dir>/references)")
    return p


def _ingest_references(paths: list[Any], references_dir: Any,
                       data: dict[str, Any]) -> None:
    """Store CLI reference files as content blobs and point the config at ids.

    Files are copied verbatim (no re-encode: identity must cover the bytes the
    user chose); the config keeps ids + store directory only, so job envelopes
    never carry server paths. Validation stays in the store: magic sniff,
    decode caps, size caps, store fullness — every refusal is an exit-2
    config error, never a traceback."""
    from qwen_engine import reference_store
    if references_dir is None:
        cache_dir = data.get("cache_dir")
        if cache_dir is None:
            raise ConfigError(FieldError(
                "references_dir",
                "--reference requires --references-dir (or a cache_dir; the "
                "store defaults to <cache_dir>/references)"))
        references_dir = Path(str(cache_dir)) / "references"
    store_dir = Path(str(references_dir))
    try:
        store_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConfigError(FieldError(
            "references_dir", f"store directory is not usable: {store_dir}: "
                              f"{exc.strerror or exc}")) from exc
    ids = []
    for position, raw_path in enumerate(paths, start=1):
        path = Path(str(raw_path))
        try:
            blob = read_bounded_file(path, reference_store.MAX_REFERENCE_BYTES)
        except OSError as exc:
            raise ConfigError(FieldError(
                "references", f"reference {position} is not a readable file: "
                              f"{path}: {exc.strerror or exc}")) from exc
        except (ValueError, RuntimeError) as exc:
            raise ConfigError(FieldError(
                "references", f"reference {position} exceeds the store byte cap: "
                              f"{path}: {exc}")) from exc
        try:
            ids.append(reference_store.put_bytes(store_dir, blob))
        except EngineError as exc:
            raise ConfigError(FieldError(
                "references", f"reference {position} refused: {exc}")) from exc
    data["references"] = ids
    data["references_dir"] = str(store_dir)


def _load_config(args: argparse.Namespace) -> GenerationConfig:
    data: dict[str, Any] = {}
    if args.config is not None:
        try:
            raw = read_bounded_file(args.config, MAX_CONFIG_BYTES)
        except OSError as exc:
            raise ConfigError(
                f'config: config file not readable: {args.config}: {exc.strerror or exc}'
            ) from exc
        except (ValueError, RuntimeError) as exc:
            raise ConfigError(f"config: config input invalid: {args.config}: {exc}") from exc
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigError(
                f"config: config file is not valid UTF-8: {args.config}: {exc}"
            ) from exc
        try:
            loaded = strict_json_loads(text)
        except (ValueError, RecursionError) as exc:
            # JSONDecodeError subclasses ValueError; the integer-string digit
            # limit raises plain ValueError and pathologically deep nesting
            # RecursionError, both before any schema validation runs.
            raise ConfigError(f"config: config file is not valid JSON: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise ConfigError("config: config file must contain a JSON object")
        data = dict(loaded)
    if getattr(args, "preset", None) is not None:
        from qwen_engine.presets import preset_config
        data.update(preset_config(args.preset))
    for flag, field in _OVERRIDES.items():
        value = getattr(args, flag, None)
        if value is not None:
            data[field] = str(value) if isinstance(value, Path) else value
    if args.tiled_decode is not None:
        data["vae_tiled_decode"] = args.tiled_decode
    if getattr(args, "transparent", None) is not None:
        data["transparent_background"] = args.transparent
        if args.transparent:
            # Auto-flip only when --output-format was NOT explicitly given and
            # the effective format is not already alpha-capable. Explicit
            # --output-format jpg (or a config-file jpg) is left untouched and
            # refused by config validation — never a hidden rewrite.
            explicit_format = args.output_format is not None
            current = str(data.get("output_format", "jpg")).lower()
            if not explicit_format and current not in ("png", "webp"):
                data["output_format"] = "png"
    reference_paths = list(getattr(args, "references", None) or [])
    if reference_paths:
        _ingest_references(reference_paths, getattr(args, "references_dir", None), data)
    elif getattr(args, "references_dir", None) is not None:
        data["references_dir"] = str(args.references_dir)
    return GenerationConfig.from_dict(data)


def _logger_for(config: GenerationConfig) -> PhaseLogger:
    if config.log_dir is not None:
        return PhaseLogger(sink=file_sink(Path(config.log_dir) / "engine.jsonl"))
    return PhaseLogger()  # nothing configured -> nothing written


def _verified_report(manifest, root, reuse_policy, *, profile=None):
    """One tree's verification, honoring the optional reuse policy; the
    returned report is always a *this-process-successful* resolver result —
    either freshly hashed or the recorded identical-bytes report. On a
    fresh full hash that came out OK under a policy, the record is written;
    failures are recorded as "nothing" (and erase any stale record)."""
    if reuse_policy is None:
        return resolve_manifest(manifest, root)
    kwargs = {} if profile is None else {"profile": profile}
    recorded = reusable_resolution(manifest, root, reuse_policy, **kwargs)
    if recorded is not None:
        return recorded
    report = resolve_manifest(manifest, root)
    remember_verified_resolution(manifest, root, report, reuse_policy,
                                 **kwargs)
    return report


def _verify_assets(config: GenerationConfig, *, reuse_policy=None):
    """No-download contract (R3): refuse to run on any unresolved asset.
    Returns the :class:`ResolutionReport` so the caller can hand the verified,
    role-grouped paths to the adapter.

    ``reuse_policy`` (default ``None`` = today's behaviour: content-hash
    every artifact on every call) opts a long-lived process into the
    stat-fingerprint record of :mod:`qwen_engine.local_bundle`: a cached
    Turbo tree is re-used, not re-hashed, while the policy is on, every
    file's (dev, ino, size, mode, mtime_ns, ctime_ns) is unchanged, and the
    record is inside ``max_age_s``. Read
    :class:`~qwen_engine.local_bundle.VerificationReusePolicy` before
    enabling: stat-unchanged is a WEAKER guarantee than the content hash.
    Failed verifications are never recorded and never reused."""
    if reuse_policy is not None and not isinstance(reuse_policy,
                                                   VerificationReusePolicy):
        raise ValueError(
            "reuse_policy must be None or a VerificationReusePolicy, got "
            f"{reuse_policy!r}")
    if config.manifest_path is None:
        raise ManifestError(
            "no model manifest configured (manifest_path is null): assets are "
            "unverifiable and the resolver never downloads — set manifest_path "
            "(e.g. models/manifest.json)"
        )
    manifest = load_manifest(config.manifest_path)
    root = (Path(config.models_dir) if config.models_dir is not None
            else Path(config.manifest_path).parent)
    report = _verified_report(manifest, root, reuse_policy)
    if not report.ok:
        for err in report.errors:
            print(f"asset error: {err}", file=sys.stderr)
        raise ManifestError(
            f"{len(report.errors)} model artifact(s) unresolved; nothing "
            f"downloaded (fetching is a separate operator action)"
        )
    # Fresh-verification priming (measured 2026-10-25): adapter.load's
    # resolve_local_bundle re-hashed every byte the line just above hashed
    # (~19 s base + ~9 s turbo, every pass paid twice per run). Project the
    # honest report through the SAME trusted projection and remember it for
    # the adapter's reverify seconds later in this process. Best effort by
    # design: a manifest outside the reviewed profile will be refused by the
    # adapter's own trusted resolve with the same typed error it raises
    # today — priming is a reuse, never a gate, and never stores for a
    # refusal.
    try:
        prime_resolution_memo(manifest, root, report)
    except LocalBundleError:
        pass
    if config.recipe is Recipe.TURBO:
        # Turbo's fail-closed path (turbo-plan §3.3): its own manifest and
        # root, resolved exactly like the base assets, then role-merged on
        # top — dit/scheduler/pipeline come from the Turbo tree while
        # text_encoder/vae/processor stay base. A missing turbo manifest is
        # a refusal, never an implicit fallback to the base weights.
        from qwen_engine.adapters.qwen21_diffusers import turbo_bundle_paths
        turbo_manifest_path, turbo_root = turbo_bundle_paths(config.manifest_path)
        try:
            turbo_manifest = load_manifest(turbo_manifest_path)
        except ManifestError as exc:
            raise ManifestError(
                f"recipe 'turbo' needs the separate turbo manifest: {exc}") from None
        turbo_report = _verified_report(turbo_manifest, turbo_root,
                                        reuse_policy,
                                        profile=TURBO_REFERENCE_PROFILE)
        if not turbo_report.ok:
            for err in turbo_report.errors:
                print(f"turbo asset error: {err}", file=sys.stderr)
            raise ManifestError(
                f"{len(turbo_report.errors)} turbo artifact(s) unresolved; "
                f"nothing downloaded (fetching is a separate operator action)")
        try:
            prime_resolution_memo(turbo_manifest, turbo_root, turbo_report,
                                  profile=TURBO_REFERENCE_PROFILE)
        except LocalBundleError:
            pass  # the adapter's trusted resolve owns the refusal
        report = replace(report, resolved={**report.resolved,
                                           **turbo_report.resolved})
    return report


def _make_adapter(*, trust_unbounded_host: bool = False) -> Any:
    """Instantiate the real adapter; the only path to torch/diffusers/transformers.

    Only the *absence of the adapter module itself* is a packaging error; a
    :class:`ModuleNotFoundError` for anything else (e.g. torch missing *inside*
    the adapter) keeps its own identity — a packaging fault must never
    masquerade as a missing dependency, and vice versa.

    ``trust_unbounded_host`` is forwarded verbatim as the operator's own
    cgroup-scope attestation; it changes no other gate.
    """
    import importlib

    try:
        module = importlib.import_module("qwen_engine.adapters.qwen21_diffusers")
    except ModuleNotFoundError as exc:
        if exc.name != "qwen_engine.adapters.qwen21_diffusers":
            raise
        raise EngineError(
            "no pipeline adapter module available: "
            "qwen_engine.adapters.qwen21_diffusers is missing from this "
            "installation (packaging error, not a missing dependency)"
        ) from exc
    return module.Qwen21DiffusersAdapter(trust_unbounded_host=trust_unbounded_host)


def _run_generate(config: GenerationConfig, logger: PhaseLogger,
                  assets: Any = None, adapter: Any = None,
                  trust_unbounded_host: bool = False,
                  calibration_record: Path | None = None,
                  performance_report: Path | None = None) -> int:
    # Refuse an unconfigured output BEFORE the adapter exists at all: an
    # adapter import/load failure is an exit-1 boundary and must not mask
    # the config-error-2 for a run with nowhere to save.
    require_run_config(config)
    if performance_report is not None:
        if calibration_record is not None:
            raise ConfigError("performance and calibration collectors reset CUDA peaks; run them separately")
        from qwen_engine.performance import PerformanceRecorder, stage
        recorder = PerformanceRecorder(config, human_sink=lambda line: print(line, file=sys.stderr))
        try:
            handle = Path(performance_report).open("x", encoding="utf-8")
        except OSError as exc:
            raise ConfigError("performance_report must name a new writable file") from exc
        with handle, recorder.activate():
            status = "failed"
            try:
                with stage("total_job"):
                    code = _run_generate(config, logger, assets, adapter, trust_unbounded_host)
                status = "succeeded"
                return code
            finally:
                recorder.write(handle, status)
    adapter = adapter if adapter is not None else _make_adapter(
        trust_unbounded_host=trust_unbounded_host)
    collector = None
    if calibration_record is not None:
        # Measurement setup happens before the load phase so its samples
        # cover the whole run. A setup failure refuses the run: an explicit
        # measurement request is never silently ignored.
        from qwen_engine.calibration import CalibrationCollector

        collector = CalibrationCollector(config, record_path=calibration_record)
        # Bind before sampling: an adapter that cannot report measured facts
        # refuses the run before any device state is touched.
        collector.bind(adapter)
        collector.start()
    try:
        result = generate(
            config, adapter, assets or {}, logger,
            phase_callback=collector.on_phase if collector is not None else None)
    except BaseException:
        if collector is not None:
            collector.abort()
        raise
    # The verified save is already published: a measurement failure below
    # must keep the image and fail only the record (exit 1 via EngineError).
    print(json.dumps(result.to_dict(), sort_keys=True))
    if collector is not None:
        observations = collector.finish()
        _safe_log(logger, "calibration_record", path=str(collector.path),
                  observations=len(observations))
    return int(ExitCode.OK)


def _safe_log(logger: PhaseLogger, name: str, **fields: Any) -> None:
    """Log an event without ever changing the CLI's exit path: a broken log
    sink gets one short diagnostic line and is otherwise ignored."""
    try:
        logger.event(name, **fields)
    except Exception as log_exc:
        print(f"warning: log sink failed: {type(log_exc).__name__}: {log_exc}",
              file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # usage errors: argparse prints + exit 2
    if args.run and args.dry_run:
        print("error: --run and --dry-run are mutually exclusive", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return int(ExitCode.CONFIG)
    if (args.calibration_record is not None or args.performance_report is not None) and not args.run:
        print("error: a measurement report requires --run", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return int(ExitCode.CONFIG)
    logger = PhaseLogger()
    try:
        config = _load_config(args)
        logger = _logger_for(config)
        if args.dry_run:
            status = manifest_status(config.manifest_path, config.models_dir)
            document = plan(config, manifest=status)
            if config.references:
                # Echo ids, sizes and order (design §2.2); blobs are never
                # resolved — sizes come from the freshly ingested input files,
                # and a config-file-only run reports null sizes.
                paths = list(args.references or [])
                echoed = []
                for ordinal, reference_id in enumerate(config.references, start=1):
                    entry: dict[str, Any] = {"ordinal": ordinal, "id": reference_id}
                    entry["bytes"] = None
                    if ordinal <= len(paths):
                        try:
                            entry["bytes"] = Path(str(paths[ordinal - 1])).stat().st_size
                        except OSError as exc:
                            raise ConfigError(FieldError(
                                "references", f"reference {ordinal} vanished: {exc}")) from exc
                    echoed.append(entry)
                document["references"] = echoed
            print(json.dumps(document, sort_keys=True, indent=2))
            return int(ExitCode.OK)
        if not args.run:
            parser.print_help()
            return int(ExitCode.OK)
        # Asset problems are checked first and keep exit 4 even when the
        # config is *also* incomplete. A missing output_dir is then a
        # config error (2), checked BEFORE the adapter is created/imported
        # — the absent M4 adapter must not turn it into exit 1.
        report = _verify_assets(config)  # -> ManifestError -> exit 4
        if args.attest_unbounded_host:
            print(
                "notice: --attest-unbounded-host: attesting that this process "
                "tree has no cgroup memory limit; the encoder gate will apply "
                "only a freshly read host MemAvailable (recorded in run evidence)",
                file=sys.stderr)
            _safe_log(logger, "host_scope_attested",
                      attestation="unbounded-host", source="operator")
        return _run_generate(config, logger, assets=report.resolved,
                             trust_unbounded_host=args.attest_unbounded_host,
                             calibration_record=args.calibration_record,
                             performance_report=args.performance_report)
    except ConfigError as exc:
        for err in exc.errors:
            print(f"config error: {err}", file=sys.stderr)
        return int(ExitCode.CONFIG)
    except MemoryGuardError as exc:
        print(f"memory refusal: {exc}", file=sys.stderr)
        return int(ExitCode.MEMORY_REFUSAL)
    except HardwareAbsentError as exc:
        print(f"hardware: {exc}", file=sys.stderr)
        return int(ExitCode.HARDWARE)
    except ManifestError as exc:
        print(f"asset error: {exc}", file=sys.stderr)
        return int(ExitCode.ASSET)
    except EngineError as exc:
        _safe_log(logger, "engine_error", error=str(exc))
        print(f"error: {exc}", file=sys.stderr)
        return int(ExitCode.UNEXPECTED)
    except Exception:
        tb = traceback.format_exc()
        _safe_log(logger, "unexpected_error", error=tb.splitlines()[-1],
                  traceback=tb)
        print(f"error: unexpected {sys.exc_info()[0].__name__}: "
              f"{sys.exc_info()[1]}", file=sys.stderr)
        return int(ExitCode.UNEXPECTED)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

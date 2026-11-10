"""Measured-memory calibration: the exact bridge and the opt-in collector.

Two layers live here, both ML-free at import time:

* **The bridge** turns a fully measured :class:`CalibrationObservation` into
  the typed objects :mod:`qwen_engine.memory` consumes (a ``PolicyDescriptor``
  and ``CalibrationEvidence``). A record missing *any* exactness field is
  refused with the missing names — never partially trusted, never guessed.
  Legacy ``calibration_observation.v1`` records (no device/runtime identity,
  no device-wide footprint) therefore stay loadable but ineligible.

* **The collector** runs inside a real generation process, opt-in via
  ``--calibration-record PATH``. It samples the device-wide used memory
  (NVML, matched by CUDA UUID) and the owned process tree's RSS while the
  engine executes, snapshots Torch's own phase peaks, and observes the
  generation runtime identity plus the artifact-set digest. It writes one
  observation per generation phase through :func:`bench.add_observation`
  only after every field has been measured; any missing measurement aborts
  the write (the generated image itself is unaffected).

Nothing here probes hardware to grant fit, touches drivers, or fabricates a
number: an unavailable sampler or identity is a refusal, and a null field is
"not measured", never 0.
"""
from __future__ import annotations

import ctypes
import importlib
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from qwen_engine.bench import (
    EXACT_OBSERVATION_FIELDS,
    CalibrationObservation,
    add_observation,
)
from qwen_engine.errors import EngineError
from qwen_engine.hardware_probe import _Memory, _uuid, is_v100_memory
from qwen_engine.local_bundle import reference_bundle_fingerprints
from qwen_engine.manifest import load_manifest
from qwen_engine.memory import (
    CalibrationEvidence,
    DeviceIdentity,
    PolicyDescriptor,
)

GiB = 1024**3


class CalibrationError(EngineError):
    """Measurement unavailable/incoherent; no calibration record is written."""


#: Fields whose absence always disqualifies a record. ``negative_prompt_tokens``
#: is legitimately absent when CFG is inactive and ``step_index`` outside a
#: single denoise step; :func:`missing_exact_fields` applies those rules.
_ALWAYS_REQUIRED = tuple(
    name for name in EXACT_OBSERVATION_FIELDS
    if name not in ("negative_prompt_tokens", "step_index"))


def missing_exact_fields(observation: CalibrationObservation) -> tuple[str, ...]:
    """Names of exactness fields this record does not carry (empty = exact).

    The conditional fields are checked against the record's own semantics:
    an active CFG needs a measured negative length, and ``denoise-step``
    needs its in-range step index. Everything else must be non-``None``.
    """
    if type(observation) is not CalibrationObservation:
        raise CalibrationError("a CalibrationObservation is required")
    missing = [name for name in _ALWAYS_REQUIRED
               if getattr(observation, name) is None]
    if observation.cfg_active is True and observation.negative_prompt_tokens is None:
        missing.append("negative_prompt_tokens")
    if observation.phase == "denoise-step" and observation.step_index is None:
        missing.append("step_index")
    return tuple(sorted(set(missing)))


def _require_exact(observation: CalibrationObservation) -> None:
    missing = missing_exact_fields(observation)
    if missing:
        raise CalibrationError(
            "calibration observation is not exact evidence; missing: "
            + ", ".join(missing))


def device_identity(observation: CalibrationObservation) -> DeviceIdentity:
    """The record's measured device identity; refuses an incomplete one."""
    _require_exact(observation)
    return DeviceIdentity(
        uuid=observation.device_uuid,
        architecture=observation.device_architecture,
        driver_version=observation.driver_version,
        cuda_build=observation.cuda_build,
    )


def policy_descriptor(observation: CalibrationObservation) -> PolicyDescriptor:
    """Exact ``PolicyDescriptor`` for one measured observation.

    ``phase`` is carried verbatim: ``denoise`` is the whole-loop peak and
    ``denoise-step`` a single indexed step; the policy never equates them.
    """
    _require_exact(observation)
    width, height = observation.resolution
    try:
        return PolicyDescriptor(
            phase=observation.phase,
            width=width, height=height,
            steps=observation.steps,
            prompt_tokens=observation.prompt_tokens,
            negative_prompt_tokens=observation.negative_prompt_tokens,
            cfg_active=observation.cfg_active,
            backend=observation.backend,
            quant_format=observation.quant,
            residency=observation.residency,
            vae_tiled_decode=observation.vae_tiled_decode,
            encoder_dtype=observation.encoder_dtype,
            execution_dtype=observation.execution_dtype,
            use_kv_cache=observation.use_kv_cache,
            model_revision=observation.model_revision,
            artifact_set_sha256=observation.artifact_set_sha256,
            runtime_identity_sha256=observation.runtime_identity_sha256,
            step_index=observation.step_index,
            embedding_cache_hit=observation.embedding_cache_hit,
        )
    except EngineError as exc:
        raise CalibrationError(
            f"observation cannot form an exact policy descriptor: {exc}") from exc


def calibration_evidence(observation: CalibrationObservation) -> CalibrationEvidence:
    """Typed evidence the memory policy can compare; refuses incompleteness."""
    descriptor = policy_descriptor(observation)
    try:
        return CalibrationEvidence(
            descriptor=descriptor,
            device=device_identity(observation),
            observed_at=observation.observed_at,
            peak_gpu_reserved_bytes=observation.peak_gpu_reserved_bytes,
            peak_host_rss_bytes=observation.peak_host_rss_bytes,
            peak_gpu_allocated_bytes=observation.peak_gpu_allocated_bytes,
            peak_gpu_used_bytes=observation.peak_gpu_used_bytes,
            peak_host_tree_bytes=observation.peak_host_tree_bytes,
        )
    except EngineError as exc:
        raise CalibrationError(
            f"observation cannot form exact calibration evidence: {exc}") from exc


# ---------------------------------------------------------------------------
# Device-wide NVML sampler (one long-lived session, UUID matched)
# ---------------------------------------------------------------------------

_DRIVER_RE = re.compile(r"[0-9]{1,4}(?:\.[0-9]{1,4}){1,3}")


class _NvmlDevice:
    """One NVML session bound to the CUDA device's UUID.

    The handle is looked up by UUID, never by assuming NVML/CUDA index
    ordering. ``used_bytes`` includes the CUDA context and every non-Torch
    allocation on the device (this process and any other).
    """

    def __init__(self, uuid: str, expected_total_bytes: int):
        try:
            valid_uuid = type(uuid) is str and _uuid(uuid) == uuid
        except Exception:
            valid_uuid = False
        if not valid_uuid:
            raise CalibrationError("invalid CUDA UUID for NVML binding")
        try:
            library = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError:
            raise CalibrationError(
                "NVML is unavailable; device-wide peaks cannot be measured") from None
        specifications = {
            "nvmlInit_v2": [],
            "nvmlShutdown": [],
            "nvmlDeviceGetHandleByUUID": [ctypes.c_char_p,
                                          ctypes.POINTER(ctypes.c_void_p)],
            "nvmlDeviceGetUUID": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char),
                                  ctypes.c_uint],
            "nvmlSystemGetDriverVersion": [ctypes.POINTER(ctypes.c_char),
                                           ctypes.c_uint],
            "nvmlDeviceGetMemoryInfo": [ctypes.c_void_p,
                                        ctypes.POINTER(_Memory)],
        }
        for name, arguments in specifications.items():
            try:
                function = getattr(library, name)
            except AttributeError:
                raise CalibrationError(
                    f"NVML symbol unavailable: {name}") from None
            function.argtypes, function.restype = arguments, ctypes.c_int
        self._library = library
        self._handle = ctypes.c_void_p()
        self._initialized = False
        self._closed = False
        self._call("nvmlInit_v2")
        self._initialized = True
        try:
            self._call("nvmlDeviceGetHandleByUUID", uuid.encode("ascii"),
                       ctypes.byref(self._handle))
            if not self._handle.value:
                raise CalibrationError("NVML device handle is missing")
            identity = ctypes.create_string_buffer(96)
            self._call("nvmlDeviceGetUUID", self._handle, identity, len(identity))
            try:
                observed_uuid = _uuid(self._text(identity))
            except Exception:
                observed_uuid = None
            if observed_uuid != uuid:
                raise CalibrationError("CUDA/NVML UUID mismatch")
            driver = ctypes.create_string_buffer(80)
            self._call("nvmlSystemGetDriverVersion", driver, len(driver))
            self.driver_version = self._text(driver)
            if _DRIVER_RE.fullmatch(self.driver_version) is None:
                raise CalibrationError("invalid NVML driver version")
            memory = _Memory()
            self._call("nvmlDeviceGetMemoryInfo", self._handle,
                       ctypes.byref(memory))
            self.total_bytes = int(memory.total)
            # CUDA reports *usable* memory (the driver reserves a few hundred
            # MiB), so exact equality with NVML's raw total is wrong. The UUID
            # binds the device; the totals must still agree within a bound.
            if (not 15 * GiB <= self.total_bytes <= 17 * GiB
                    or not 0 < expected_total_bytes <= self.total_bytes
                    or self.total_bytes - expected_total_bytes > GiB):
                raise CalibrationError(
                    "NVML total is inconsistent with the CUDA device's total "
                    "memory")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _text(buffer) -> str:
        if b"\0" not in buffer.raw:
            raise CalibrationError("NVML returned an unterminated string")
        try:
            value = buffer.value.decode("ascii")
        except UnicodeError:
            raise CalibrationError("NVML returned a non-ASCII string") from None
        if not value:
            raise CalibrationError("NVML returned an empty string")
        return value

    def _call(self, name: str, *args) -> None:
        result = getattr(self._library, name)(*args)
        if type(result) is not int or result != 0:
            raise CalibrationError(f"NVML call failed: {name}")

    def used_bytes(self) -> int:
        if self._closed:
            raise CalibrationError("NVML session is closed")
        memory = _Memory()
        self._call("nvmlDeviceGetMemoryInfo", self._handle, ctypes.byref(memory))
        total, free, used = int(memory.total), int(memory.free), int(memory.used)
        if (total != self.total_bytes or free < 0 or used < 0
                or free + used != total):
            raise CalibrationError("incoherent NVML memory snapshot")
        return used

    def close(self) -> None:
        """Best-effort balanced shutdown; process exit is the fallback."""
        if self._closed:
            return
        self._closed = True
        if self._initialized:
            try:
                self._library.nvmlShutdown()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Sampling windows
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Window:
    peak_gpu_used: int
    peak_host_rss: int
    peak_host_tree: int
    samples: int


@dataclass(frozen=True)
class _PhaseRecord:
    phase: str
    observed_at: float
    peak_gpu_allocated: int | None
    peak_gpu_reserved: int | None
    window: _Window


_FACT_KEYS = ("prompt_tokens", "negative_prompt_tokens", "cfg_active",
              "encoder_dtype", "execution_dtype", "use_kv_cache",
              "embedding_cache_hit")


def validate_adapter_facts(facts: Any, config) -> dict[str, Any]:
    """Refuse adapter facts that are absent, typed wrong, or incoherent.

    A hard gate rather than ``dict.get`` defaults: an unmeasured cache
    decision or a token length that disagrees with the config's CFG predicate
    must stop the record, never become a plausible-looking value.
    """
    if not isinstance(facts, dict):
        raise CalibrationError("adapter conditioning facts were not measured")
    missing = [key for key in _FACT_KEYS if key not in facts]
    if missing:
        raise CalibrationError(
            "adapter calibration facts are incomplete; missing: "
            + ", ".join(missing))
    if (type(facts["prompt_tokens"]) is not int or facts["prompt_tokens"] < 1
            or type(facts["use_kv_cache"]) is not bool
            or type(facts["embedding_cache_hit"]) is not bool
            or type(facts["cfg_active"]) is not bool
            or facts["cfg_active"] != config.cfg_active
            or facts["encoder_dtype"] not in ("fp16", "bf16", "fp32")
            or facts["execution_dtype"] not in ("fp16", "fp32")):
        raise CalibrationError(
            "adapter calibration facts are not coherent with the run")
    negative = facts["negative_prompt_tokens"]
    if facts["cfg_active"]:
        if type(negative) is not int or negative < 1:
            raise CalibrationError(
                "active CFG needs a measured negative prompt length")
    elif negative is not None:
        raise CalibrationError(
            "inactive CFG must not report a negative prompt length")
    return facts


class _Sampler:
    """One sampling thread: device-wide NVML used + owned-tree RSS.

    RSS values are *current* readings sampled on an interval, not kernel
    high-water marks; a shorter spike than the interval can be missed. The
    collector's configured interval (50 ms by default) states the coverage
    limit; it is not a claim of continuous observation.
    """

    def __init__(self, nvml: _NvmlDevice, interval_s: float):
        if not 0.005 <= interval_s <= 1.0:
            raise CalibrationError("sampling interval must be in [0.005, 1.0] s")
        try:
            import psutil  # pinned in requirements.lock; lazy for ML-free import
        except ImportError:
            raise CalibrationError(
                "psutil is required to sample the owned process tree") from None

        self._psutil = psutil
        self._process = psutil.Process(os.getpid())
        self._nvml = nvml
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._window = _Window(0, 0, 0, 0)

    def _tree_bytes(self) -> int:
        total = int(self._process.memory_info().rss)
        for child in self._process.children(recursive=True):
            try:
                total += int(child.memory_info().rss)
            except self._psutil.Error:
                continue  # the child exited mid-walk; its peak was sampled before
        return total

    def _read(self) -> tuple[int, int, int]:
        return (self._nvml.used_bytes(),
                int(self._process.memory_info().rss),
                self._tree_bytes())

    def _sample(self) -> None:
        with self._lock:
            gpu, rss, tree = self._read()
            window = self._window
            self._window = _Window(
                max(window.peak_gpu_used, gpu),
                max(window.peak_host_rss, rss),
                max(window.peak_host_tree, tree),
                window.samples + 1,
            )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self._sample()
            except BaseException as exc:  # never leak out of the thread
                self._error = exc
                return

    def start(self) -> None:
        with self._lock:
            gpu, rss, tree = self._read()
            self._window = _Window(gpu, rss, tree, 1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def reset(self) -> None:
        """Start a new phase window, seeded with the current absolute values."""
        with self._lock:
            gpu, rss, tree = self._read()
            self._window = _Window(gpu, rss, tree, 1)

    def snapshot(self) -> _Window:
        if self._error is not None:
            raise CalibrationError(
                f"memory sampler failed: {type(self._error).__name__}: "
                f"{self._error}") from self._error
        with self._lock:
            if self._window.samples < 1:
                raise CalibrationError("memory sampler produced no samples")
            return replace(self._window)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise CalibrationError("memory sampler did not stop")


# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------

#: Reviewed device profile this collector may record. A different GPU refuses:
#: the memory policy only trusts the sm_70 / CUDA 12.6 device anyway.
LOGICAL_DEVICE_INDEX = 0
ARCHITECTURE = "sm_70"
CUDA_BUILD = "12.6"


def _require_reviewed_device(properties: Any) -> None:
    """Refuse every device except a reviewed V100 (16GB or 32GB, sm_70)."""
    major = getattr(properties, "major", None)
    minor = getattr(properties, "minor", None)
    total = getattr(properties, "total_memory", None)
    if ((major, minor) != (7, 0)
            or "V100" not in str(getattr(properties, "name", ""))
            or not is_v100_memory(total)):
        raise CalibrationError(
            "measurement requires a reviewed Tesla V100 (16GB or 32GB)")


class CalibrationCollector:
    """Opt-in, in-process measured-memory collector.

    Lifecycle: ``start()`` (before generation), ``on_phase`` as the core's
    phase callback, ``finish()`` after a successful generation, or
    ``abort()`` on any failure. ``bind(adapter)`` supplies the adapter whose
    exact conditioning facts are read once encoding has completed.
    """

    def __init__(self, config, *, record_path: str | Path,
                 manifest_path: str | Path | None = None,
                 models_dir: str | Path | None = None,
                 interval_s: float = 0.05):
        from qwen_engine.config import GenerationConfig

        if type(config) is not GenerationConfig:
            raise CalibrationError("a GenerationConfig is required for measurement")
        if config.attention_backend.value not in ("auto", "sdpa"):
            raise CalibrationError(
                "the reviewed loader records only the native sdpa backend")
        path = Path(record_path)
        if path.exists() and path.is_dir():
            raise CalibrationError("calibration record path is a directory")
        self.path = path
        self._config = config
        self._manifest_path = Path(manifest_path or config.manifest_path or "")
        self._models_dir = Path(models_dir or config.models_dir or "")
        if not self._manifest_path.is_file():
            raise CalibrationError("calibration measurement needs a manifest file")
        manifest = load_manifest(self._manifest_path)
        revisions = sorted({artifact.revision for artifact in manifest.artifacts})
        if len(revisions) != 1:
            raise CalibrationError(
                "calibration observation needs exactly one pinned model revision")
        self._revision = revisions[0]
        self._artifact_set_sha256 = reference_bundle_fingerprints(manifest).bundle
        if config.quant.value == "q8":
            from qwen_engine.q8_artifact import resolve_q8_artifact
            from qwen_engine.adapters.q8_loader import artifact_set_identity
            if config.q8_manifest_path is None:
                raise CalibrationError("Q8 calibration requires an explicit sidecar manifest")
            q8 = resolve_q8_artifact(config.q8_manifest_path)
            self._artifact_set_sha256 = artifact_set_identity(self._artifact_set_sha256, q8.identity_sha256)
        self._interval_s = interval_s
        self._adapter: Any = None
        self._facts: dict[str, Any] | None = None
        self._torch: Any = None
        self._nvml: _NvmlDevice | None = None
        self._sampler: _Sampler | None = None
        self._runtime_sha256: str | None = None
        self._device: DeviceIdentity | None = None
        self._records: list[_PhaseRecord] = []
        self._current_phase: str | None = None
        self._failure: BaseException | None = None
        self._finished = False

    # -- lifecycle ---------------------------------------------------------

    def bind(self, adapter: Any) -> None:
        if not callable(getattr(adapter, "calibration_facts", None)):
            raise CalibrationError(
                "adapter does not expose measured calibration facts")
        self._adapter = adapter

    def start(self) -> None:
        if self._sampler is not None or self._finished:
            raise CalibrationError("collector is single-use")
        from qwen_engine.encoder_identity import (
            observe_installed_packages,
            validate_installed_packages,
        )

        validate_installed_packages(observe_installed_packages())
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise CalibrationError("CUDA is unavailable; nothing to measure")
        properties = torch.cuda.get_device_properties(LOGICAL_DEVICE_INDEX)
        _require_reviewed_device(properties)
        if str(torch.version.cuda) != CUDA_BUILD:
            raise CalibrationError(
                f"measurement requires the reviewed CUDA {CUDA_BUILD} build")
        try:
            uuid = _uuid(str(properties.uuid))
        except Exception:
            raise CalibrationError(
                "CUDA device UUID is unavailable or malformed") from None
        self._nvml = _NvmlDevice(uuid, int(properties.total_memory))
        try:
            if (torch.cuda.get_arch_list()
                    and ARCHITECTURE not in torch.cuda.get_arch_list()):
                raise CalibrationError(
                    "runtime has no reviewed sm_70 kernel build")
            from qwen_engine.runtime_identity import observe_generation_runtime

            runtime = observe_generation_runtime()
            if not runtime.matches_cuda_device(
                    uuid=uuid, logical_device_index=LOGICAL_DEVICE_INDEX):
                raise CalibrationError(
                    "generation runtime identity is not bound to the measured "
                    "device")
            self._runtime_sha256 = runtime.sha256
            self._device = DeviceIdentity(uuid, ARCHITECTURE,
                                          self._nvml.driver_version, CUDA_BUILD)
            self._torch = torch
            torch.cuda.reset_peak_memory_stats()
            self._sampler = _Sampler(self._nvml, self._interval_s)
            self._sampler.start()
        except BaseException:
            # Setup is atomic: a refused run must not leave NVML open or a
            # sampling thread behind, and a failure here never writes.
            try:
                self._stop_sampling()
            except CalibrationError:
                pass
            raise

    def abort(self) -> None:
        """Stop sampling and discard everything; never writes a record."""
        self._failure = self._failure or CalibrationError("measurement aborted")
        try:
            self._stop_sampling()
        except CalibrationError:
            pass  # abort must not mask the failure that caused it

    def finish(self) -> tuple[CalibrationObservation, ...]:
        """Validate every phase, then append one observation per phase."""
        if self._failure is not None:
            self._stop_sampling()
            raise CalibrationError(
                f"calibration measurement failed: {self._failure}")
        if self._sampler is None or self._finished:
            raise CalibrationError("collector was not started or already finished")
        try:
            self._close_phase()
            self._stop_sampling()
            observations = self._build_observations()
            # Completeness is checked for the WHOLE batch before the first
            # append, so an incomplete measurement cannot leave a prefix.
            for observation in observations:
                _require_exact(observation)
            for observation in observations:
                add_observation(self.path, observation)
            self._finished = True
            return observations
        except BaseException as exc:
            self._failure = exc
            try:
                self._stop_sampling()
            except CalibrationError:
                pass
            raise

    def _stop_sampling(self) -> None:
        """A sampler that will not stop is a measurement failure, not a note."""
        sampler, self._sampler = self._sampler, None
        if sampler is not None:
            sampler.stop()
        nvml, self._nvml = self._nvml, None
        if nvml is not None:
            nvml.close()

    # -- phase windows -----------------------------------------------------

    def on_phase(self, name: str) -> None:
        """Core phase callback; a measurement failure never kills generation."""
        if self._failure is not None or self._sampler is None:
            return
        try:
            if name == "denoise" and self._facts is None:
                if self._adapter is None:
                    raise CalibrationError(
                        "adapter facts were not bound before encoding")
                self._facts = self._adapter.calibration_facts()
            self._close_phase()
            self._start_phase(name)
        except BaseException as exc:
            self._failure = exc

    def _start_phase(self, name: str) -> None:
        if name not in ("load", "encode", "denoise", "decode", "save"):
            raise CalibrationError(f"unexpected generation phase: {name!r}")
        self._sampler.reset()
        if self._torch.cuda.is_initialized():
            self._torch.cuda.reset_peak_memory_stats()
        self._current_phase = name

    def _close_phase(self) -> None:
        if self._current_phase is None:
            return
        window = self._sampler.snapshot()
        allocated = reserved = None
        if self._torch.cuda.is_initialized():
            allocated = int(self._torch.cuda.max_memory_allocated())
            reserved = int(self._torch.cuda.max_memory_reserved())
        self._records.append(_PhaseRecord(
            phase=self._current_phase,
            observed_at=time.time(),
            peak_gpu_allocated=allocated,
            peak_gpu_reserved=reserved,
            window=window,
        ))
        self._current_phase = None

    # -- record construction ----------------------------------------------

    def _build_observations(self) -> tuple[CalibrationObservation, ...]:
        if not self._records:
            raise CalibrationError("no generation phase was measured")
        if self._runtime_sha256 is None or self._device is None:
            raise CalibrationError("runtime/device identity was not observed")
        width, height = self._config.width, self._config.height
        backend = "sdpa" if self._config.attention_backend.value == "auto" \
            else self._config.attention_backend.value
        facts = validate_adapter_facts(self._facts, self._config)
        observations = []
        for record in self._records:
            observations.append(CalibrationObservation(
                observed_at=record.observed_at,
                phase=record.phase,
                model_revision=self._revision,
                backend=backend,
                quant=self._config.quant.value,
                residency=self._config.offload.value,
                vae_tiled_decode=self._config.vae_tiled_decode,
                # This loader encodes a real negative branch; it never
                # duplicates the positive prompt as a batch.
                cfg_duplicated=False,
                tokens=self._config.latent_tokens,
                resolution=(width, height),
                peak_gpu_allocated_bytes=record.peak_gpu_allocated,
                peak_gpu_reserved_bytes=record.peak_gpu_reserved,
                peak_host_rss_bytes=record.window.peak_host_rss,
                prompt_tokens=facts["prompt_tokens"],
                steps=self._config.steps,
                negative_prompt_tokens=facts["negative_prompt_tokens"],
                cfg_active=self._config.cfg_active,
                encoder_dtype=facts["encoder_dtype"],
                execution_dtype=facts["execution_dtype"],
                use_kv_cache=facts["use_kv_cache"],
                step_index=None,
                embedding_cache_hit=facts["embedding_cache_hit"],
                artifact_set_sha256=self._artifact_set_sha256,
                runtime_identity_sha256=self._runtime_sha256,
                device_uuid=self._device.uuid,
                device_architecture=self._device.architecture,
                driver_version=self._device.driver_version,
                cuda_build=self._device.cuda_build,
                peak_gpu_used_bytes=record.window.peak_gpu_used,
                peak_host_tree_bytes=record.window.peak_host_tree,
            ))
        return tuple(observations)

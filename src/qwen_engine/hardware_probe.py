"""Explicit fresh-worker V100 diagnostic, never planning/import side effects.

Pinned Torch + a tiny real FP16 CUDA kernel + UUID-matched NVML whole-device
counters. No drivers/services/clocks/modes changed, no model weights loaded.
Driver calls/context initialization are not hard-deadline bounded: invoke in
an owned diagnostic worker, not a server request or a long-lived supervisor.
"""
from __future__ import annotations

import ctypes
import math
import re
import time
from dataclasses import dataclass

from qwen_engine.encoder_identity import observe_installed_packages, validate_installed_packages
from qwen_engine.errors import HardwareAbsentError
from qwen_engine.mem_preflight import read_memory_snapshot
from qwen_engine.memory import DeviceIdentity, ResourceEvidence

GiB = 1024**3

#: Total memory a reviewed Tesla V100 reports, as inclusive byte ranges: the
#: 16GB cards report just under 16 GiB, the 32GB cards (SXM2, PCIe, V100S) just
#: under 32 GiB. Anything else is not a card this engine was reviewed for.
V100_MEMORY_RANGES = ((15 * GiB, 16 * GiB), (31 * GiB, 32 * GiB))


def is_v100_memory(total: object) -> bool:
    """True when ``total`` bytes is the memory size of a reviewed V100."""
    return type(total) is int and any(low <= total <= high for low, high in V100_MEMORY_RANGES)


class HardwareProbeError(HardwareAbsentError):
    """Fixed diagnostic; unavailable/incoherent measurements never grant fit."""


def _uuid(value):
    if type(value) is not str:
        raise HardwareProbeError("invalid_gpu_uuid")
    raw = value[4:] if value.startswith("GPU-") else value
    if re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", raw) is None:
        raise HardwareProbeError("invalid_gpu_uuid")
    return "GPU-" + raw.lower()


class _Memory(ctypes.Structure):
    # NVIDIA's nvmlMemory_t (v1): used includes reserved + allocated memory.
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _nvml_snapshot(uuid):
    try:
        library = ctypes.CDLL("libnvidia-ml.so.1")
        specifications = {
            "nvmlInit_v2": [], "nvmlShutdown": [],
            "nvmlDeviceGetHandleByUUID": [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)],
            "nvmlDeviceGetUUID": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint],
            "nvmlSystemGetDriverVersion": [ctypes.POINTER(ctypes.c_char), ctypes.c_uint],
            "nvmlDeviceGetMemoryInfo": [ctypes.c_void_p, ctypes.POINTER(_Memory)],
        }
        for name, arguments in specifications.items():
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, ctypes.c_int
    except (OSError, AttributeError):
        raise HardwareProbeError("nvml_interface_unavailable") from None

    def call(name, *args):
        result = getattr(library, name)(*args)
        if type(result) is not int or result != 0:
            raise HardwareProbeError("nvml_query_failed")

    def text(buffer):
        if b"\0" not in buffer.raw:
            raise HardwareProbeError("nvml_string_unterminated")
        try:
            result = buffer.value.decode("ascii")
        except UnicodeError:
            raise HardwareProbeError("nvml_string_invalid") from None
        if not result or any(not 33 <= ord(c) <= 126 for c in result):
            raise HardwareProbeError("nvml_string_invalid")
        return result

    call("nvmlInit_v2")
    failed = False
    try:
        handle = ctypes.c_void_p()
        call("nvmlDeviceGetHandleByUUID", uuid.encode("ascii"), ctypes.byref(handle))
        if not handle.value:
            raise HardwareProbeError("nvml_device_handle_missing")
        identity, driver = ctypes.create_string_buffer(96), ctypes.create_string_buffer(80)
        call("nvmlDeviceGetUUID", handle, identity, len(identity))
        call("nvmlSystemGetDriverVersion", driver, len(driver))
        if _uuid(text(identity)) != uuid:
            raise HardwareProbeError("cuda_nvml_uuid_mismatch")
        version = text(driver)
        if re.fullmatch(r"[0-9]{1,4}(?:\.[0-9]{1,4}){1,3}", version) is None:
            raise HardwareProbeError("invalid_driver_version")
        memory = _Memory()
        call("nvmlDeviceGetMemoryInfo", handle, ctypes.byref(memory))
        if (not is_v100_memory(int(memory.total)) or memory.free > memory.total
                or memory.used > memory.total or memory.free + memory.used != memory.total):
            raise HardwareProbeError("incoherent_or_unreviewed_gpu_memory")
        return version, int(memory.total), int(memory.free), int(memory.used)
    except BaseException:
        failed = True
        raise
    finally:
        try:
            call("nvmlShutdown")
        except Exception:
            if not failed:
                raise
            # Do not hide the primary failure/interrupt. Fresh worker exit
            # releases process state; no successful snapshot is returned.


def _torch():
    validate_installed_packages(observe_installed_packages())
    import torch
    return torch


def _kernel(torch, index):
    device = torch.device("cuda", index)
    with torch.inference_mode():
        inputs = torch.ones((16, 16), dtype=torch.float16, device=device)
        output = inputs @ inputs
        torch.cuda.synchronize(index)
        if (output.device != device or output.dtype != torch.float16
                or tuple(output.shape) != (16, 16)
                or not bool(torch.equal(output.cpu(), torch.full((16, 16), 16, dtype=torch.float16)))):
            raise HardwareProbeError("cuda_fp16_kernel_result_invalid")


@dataclass(frozen=True)
class HardwareObservation:
    resources: ResourceEvidence
    logical_device_index: int
    device_name: str
    gpu_used_bytes: int

    def __post_init__(self):
        if (type(self.resources) is not ResourceEvidence or self.resources.device is None
                or not self.resources.cuda_kernel_verified
                or type(self.logical_device_index) is not int or not 0 <= self.logical_device_index < 128
                or type(self.device_name) is not str or not 1 <= len(self.device_name) <= 128
                or "V100" not in self.device_name
                or self.resources.device.architecture != "sm_70" or self.resources.device.cuda_build != "12.6"
                or _uuid(self.resources.device.uuid) != self.resources.device.uuid
                or type(self.gpu_used_bytes) is not int or self.gpu_used_bytes < 0
                or self.resources.gpu_total_bytes is None or self.resources.gpu_available_bytes is None
                or not is_v100_memory(self.resources.gpu_total_bytes)
                or self.gpu_used_bytes + self.resources.gpu_available_bytes != self.resources.gpu_total_bytes):
            raise HardwareProbeError("invalid_hardware_observation")


def probe_v100(*, logical_device_index: int = 0) -> HardwareObservation:
    """Explicit tiny-kernel diagnostic; not checkpoint numerics or calibration.

    Device selection uses the executing CUDA process's logical index; NVML is
    matched by its UUID, never by assuming CUDA/NVML index ordering. The oldest
    component-read time is retained. Host budget discovery remains fail-closed.
    Nothing here installs drivers, exposes a public endpoint or enables jobs.
    """
    if type(logical_device_index) is not int or not 0 <= logical_device_index < 128:
        raise HardwareProbeError("invalid_logical_device_index")
    try:
        captured = time.time()
        if type(captured) not in (int, float) or captured <= 0 or not math.isfinite(captured):
            raise HardwareProbeError("hardware_capture_time_unavailable")
        host = read_memory_snapshot()
        torch = _torch()
        if torch.version.cuda != "12.6" or not torch.cuda.is_available():
            raise HardwareProbeError("reviewed_cuda_device_unavailable")
        if logical_device_index >= torch.cuda.device_count():
            raise HardwareProbeError("logical_device_unavailable")
        properties = torch.cuda.get_device_properties(logical_device_index)
        if (type(properties.major) is not int or type(properties.minor) is not int
                or (properties.major, properties.minor) != (7, 0)
                or type(properties.name) is not str or not 1 <= len(properties.name) <= 128
                or "V100" not in properties.name
                or not is_v100_memory(properties.total_memory)):
            raise HardwareProbeError("required_v100_profile_unavailable")
        uuid = _uuid(str(properties.uuid))
        if "sm_70" not in torch.cuda.get_arch_list():
            raise HardwareProbeError("runtime_sm70_kernel_build_unavailable")
        _kernel(torch, logical_device_index)
        driver, total, available, used = _nvml_snapshot(uuid)
        resources = ResourceEvidence(captured, host, DeviceIdentity(uuid, "sm_70", driver, "12.6"),
                                     True, total, available)
        return HardwareObservation(resources, logical_device_index, properties.name, used)
    except HardwareProbeError:
        raise
    except Exception:
        raise HardwareProbeError("hardware_probe_unavailable") from None

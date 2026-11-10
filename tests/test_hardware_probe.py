"""Pure query/kernel spies; synthetic records do not prove a real V100."""
import ctypes
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from qwen_engine import hardware_probe as module
from qwen_engine.mem_preflight import MemorySnapshot

UUID = "GPU-01234567-0123-0123-0123-012345678901"


@pytest.fixture
def probe(monkeypatch):
    calls = []
    properties = SimpleNamespace(major=7, minor=0, name="Tesla V100-PCIE-16GB",
                                 total_memory=16 * module.GiB, uuid=UUID)
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                           get_device_properties=lambda index: properties,
                           get_arch_list=lambda: ["sm_70", "sm_75"])
    torch = SimpleNamespace(version=SimpleNamespace(cuda="12.6"), cuda=cuda)
    host = MemorySnapshot(32 * module.GiB, "unknown")
    monkeypatch.setattr(module.time, "time", lambda: 100.0)
    monkeypatch.setattr(module, "read_memory_snapshot", lambda: calls.append("host") or host)
    monkeypatch.setattr(module, "_torch", lambda: calls.append("torch") or torch)
    monkeypatch.setattr(module, "_kernel", lambda torch, index: calls.append(("kernel", index)))
    def nvml(uuid):
        calls.append(("nvml", uuid))
        return "560.35.03", 16 * module.GiB, 12 * module.GiB, 4 * module.GiB
    monkeypatch.setattr(module, "_nvml_snapshot", nvml)
    return SimpleNamespace(torch=torch, properties=properties, host=host, calls=calls)


def test_oldest_read_time_real_kernel_then_uuid_matched_counters(probe):
    value = module.probe_v100(logical_device_index=1)
    assert probe.calls == ["host", "torch", ("kernel", 1), ("nvml", UUID)]
    assert value.resources.captured_at == 100 and value.resources.host is probe.host
    assert value.resources.cuda_kernel_verified and value.resources.device.uuid == UUID
    assert value.gpu_used_bytes == 4 * module.GiB and value.logical_device_index == 1
    assert value.resources.host.budget()[0] is None  # GPU success cannot grant a complete host budget


@pytest.mark.parametrize("index", [-1, 128, True, 0.0, "0", None])
def test_invalid_index_precedes_any_probe(probe, index):
    with pytest.raises(module.HardwareProbeError, match="invalid_logical"):
        module.probe_v100(logical_device_index=index)
    assert probe.calls == []


@pytest.mark.parametrize("timestamp", [0, -1, True, None, float("nan"), float("inf")])
def test_bad_clock_precedes_host_and_ml(probe, monkeypatch, timestamp):
    monkeypatch.setattr(module.time, "time", lambda: timestamp)
    with pytest.raises(module.HardwareProbeError):
        module.probe_v100()
    assert probe.calls == []


@pytest.mark.parametrize("case", ["runtime", "absent", "index", "arch_build", "arch", "name", "size", "uuid", "bool"])
def test_unsupported_device_refuses_before_kernel_or_nvml(probe, case):
    if case == "runtime":
        probe.torch.version.cuda = "13.0"
    elif case == "absent":
        probe.torch.cuda.is_available = lambda: False
    elif case == "index":
        probe.torch.cuda.device_count = lambda: 0
    elif case == "arch_build":
        probe.torch.cuda.get_arch_list = lambda: ["sm_75"]
    elif case == "arch":
        probe.properties.major = 8
    elif case == "name":
        probe.properties.name = "TITAN V"
    elif case == "size":
        probe.properties.total_memory = 24 * module.GiB  # no V100 has 24GB
    elif case == "uuid":
        probe.properties.uuid = "not a UUID"
    else:
        probe.properties.minor = False
    with pytest.raises(module.HardwareProbeError):
        module.probe_v100()
    assert probe.calls == ["host", "torch"]


def test_kernel_failure_precedes_nvml_and_interrupt_propagates(probe, monkeypatch):
    def failure(*args):
        raise RuntimeError("private driver diagnostic")
    monkeypatch.setattr(module, "_kernel", failure)
    with pytest.raises(module.HardwareProbeError, match="unavailable") as raised:
        module.probe_v100()
    assert "private" not in str(raised.value) and probe.calls == ["host", "torch"]
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(module, "_kernel", interrupt)
    with pytest.raises(KeyboardInterrupt):
        module.probe_v100()


@pytest.mark.parametrize("uuid", [UUID, UUID[4:], UUID.upper().replace("GPU-", "GPU-")])
def test_uuid_canonicalization(uuid):
    assert module._uuid(uuid) == UUID


@pytest.mark.parametrize("uuid", [None, True, "", "MIG-" + UUID[4:], UUID + "x", UUID + "\n", "GPU-abc"])
def test_invalid_uuid(uuid):
    with pytest.raises(module.HardwareProbeError):
        module._uuid(uuid)


@pytest.mark.parametrize("field,value", [("gpu_used_bytes", True), ("gpu_used_bytes", -1),
    ("gpu_used_bytes", 1), ("logical_device_index", True), ("device_name", "another card"), ("resources", None)])
def test_observation_structural_validation(probe, field, value):
    observed = module.probe_v100()
    with pytest.raises(module.HardwareProbeError):
        replace(observed, **{field: value})


class Function:
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, *args):
        return self.fn(*args)


@pytest.fixture
def nvml(monkeypatch):
    calls = []
    values = SimpleNamespace(uuid=UUID.encode(), driver=b"560.35.03", total=16 * module.GiB,
                             free=12 * module.GiB, used=4 * module.GiB, failed=None, handle=123)
    def function(name):
        def call(*args):
            calls.append(name)
            if values.failed == name:
                return 1
            if name == "nvmlDeviceGetHandleByUUID":
                assert args[0] == UUID.encode()
                args[1]._obj.value = values.handle
            elif name == "nvmlDeviceGetUUID":
                args[1].value = values.uuid
            elif name == "nvmlSystemGetDriverVersion":
                args[0].value = values.driver
            elif name == "nvmlDeviceGetMemoryInfo":
                args[1]._obj.total, args[1]._obj.free, args[1]._obj.used = values.total, values.free, values.used
            return 0
        return Function(call)
    names = ["nvmlInit_v2", "nvmlShutdown", "nvmlDeviceGetHandleByUUID", "nvmlDeviceGetUUID",
             "nvmlSystemGetDriverVersion", "nvmlDeviceGetMemoryInfo"]
    library = SimpleNamespace(**{name: function(name) for name in names})
    def load(name):
        assert name == "libnvidia-ml.so.1"
        return library
    monkeypatch.setattr(module.ctypes, "CDLL", load)
    return SimpleNamespace(calls=calls, values=values, library=library, names=names)


def test_nvml_explicit_abi_memory_layout_and_balanced_reference(nvml):
    assert ctypes.sizeof(module._Memory) == 24
    assert module._nvml_snapshot(UUID) == ("560.35.03", 16 * module.GiB, 12 * module.GiB, 4 * module.GiB)
    assert nvml.calls == ["nvmlInit_v2", "nvmlDeviceGetHandleByUUID", "nvmlDeviceGetUUID",
                          "nvmlSystemGetDriverVersion", "nvmlDeviceGetMemoryInfo", "nvmlShutdown"]
    assert nvml.library.nvmlDeviceGetMemoryInfo.restype is ctypes.c_int
    assert nvml.library.nvmlDeviceGetMemoryInfo.argtypes[-1]._type_ is module._Memory


@pytest.mark.parametrize("failed", ["nvmlInit_v2", "nvmlDeviceGetHandleByUUID", "nvmlDeviceGetUUID",
    "nvmlSystemGetDriverVersion", "nvmlDeviceGetMemoryInfo", "nvmlShutdown"])
def test_nvml_failure_is_not_success_and_shutdown_balanced(nvml, failed):
    nvml.values.failed = failed
    with pytest.raises(module.HardwareProbeError, match="query_failed"):
        module._nvml_snapshot(UUID)
    assert ("nvmlShutdown" in nvml.calls) is (failed != "nvmlInit_v2")


@pytest.mark.parametrize("field,value", [("uuid", b"GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"),
    ("uuid", b"\xff"), ("driver", b"driver diagnostic"), ("driver", b""), ("handle", 0),
    ("total", 32 * module.GiB), ("free", 17 * module.GiB), ("used", 5 * module.GiB)])
def test_nvml_incoherent_or_wrong_device_refuses_and_closes(nvml, field, value):
    setattr(nvml.values, field, value)
    with pytest.raises(module.HardwareProbeError):
        module._nvml_snapshot(UUID)
    assert nvml.calls[-1] == "nvmlShutdown"


def test_missing_nvml_symbol_precedes_initialization(nvml):
    del nvml.library.nvmlDeviceGetMemoryInfo
    with pytest.raises(module.HardwareProbeError, match="interface_unavailable"):
        module._nvml_snapshot(UUID)
    assert nvml.calls == []


def test_nvml_shutdown_fault_never_masks_primary_interrupt(nvml):
    def interrupt(*args):
        raise KeyboardInterrupt()
    nvml.library.nvmlDeviceGetMemoryInfo = Function(interrupt)
    nvml.values.failed = "nvmlShutdown"
    with pytest.raises(KeyboardInterrupt):
        module._nvml_snapshot(UUID)
    assert nvml.calls[-1] == "nvmlShutdown"


@pytest.mark.parametrize("case", ["ok", "device", "dtype", "shape", "value"])
def test_kernel_requires_actual_device_dtype_shape_synchronized_value(case):
    calls = []
    @contextmanager
    def inference():
        calls.append("inference_enter")
        try:
            yield
        finally:
            calls.append("inference_exit")
    output = SimpleNamespace(device=("cuda", 1), dtype="fp16", shape=(16, 16))
    output.cpu = lambda: calls.append("copy_cpu") or output
    if case in ("device", "dtype", "shape"):
        setattr(output, case, {"device": ("cpu", 1), "dtype": "fp32", "shape": (16, 1)}[case])
    class Inputs:
        def __matmul__(self, other):
            assert other is self
            calls.append("matmul")
            return output
    def ones(shape, *, dtype, device):
        assert shape == (16, 16) and dtype == "fp16" and device == ("cuda", 1)
        calls.append("ones_cuda_fp16")
        return Inputs()
    def full(shape, value, *, dtype):
        assert shape == (16, 16) and value == 16 and dtype == "fp16"
        calls.append("reference_cpu_fp16")
        return "expected"
    torch = SimpleNamespace(device=lambda kind, index: (kind, index), float16="fp16", ones=ones, full=full,
        inference_mode=inference, cuda=SimpleNamespace(synchronize=lambda index: calls.append(("synchronize", index))),
        equal=lambda actual, expected: actual is output and expected == "expected" and case != "value")
    if case == "ok":
        module._kernel(torch, 1)
    else:
        with pytest.raises(module.HardwareProbeError, match="kernel_result_invalid"):
            module._kernel(torch, 1)
    assert calls[:4] == ["inference_enter", "ones_cuda_fp16", "matmul", ("synchronize", 1)]
    assert calls[-1] == "inference_exit"


@pytest.mark.parametrize("total, accepted", [
    (16945512448, True),            # V100 16GB as the driver reports it
    (34089730048, True),            # V100 32GB (SXM2 / PCIe / V100S)
    (15 * module.GiB, True), (16 * module.GiB, True),
    (31 * module.GiB, True), (32 * module.GiB, True),
    (15 * module.GiB - 1, False), (16 * module.GiB + 1, False),
    (24 * module.GiB, False), (32 * module.GiB + 1, False),
    (80 * module.GiB, False), (True, False), (16.0 * module.GiB, False), (None, False),
])
def test_only_the_16gb_and_32gb_v100_memory_sizes_are_reviewed(total, accepted):
    assert module.is_v100_memory(total) is accepted


def test_a_32gb_v100_passes_the_probe(probe, monkeypatch):
    probe.properties.name = "Tesla V100-SXM2-32GB"
    probe.properties.total_memory = 34089730048
    monkeypatch.setattr(module, "_nvml_snapshot",
                        lambda uuid: ("560.35.03", 34089730048, 30 * module.GiB,
                                      34089730048 - 30 * module.GiB))
    value = module.probe_v100()
    assert value.resources.gpu_total_bytes == 34089730048

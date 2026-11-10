# Explicit V100 probe groundwork

`hardware_probe.probe_v100()` is an explicit fresh-worker diagnostic, not
startup/server/planning behavior. Its import is ML-free. Pinned package checks
precede Torch import. It selects a CUDA logical index, requires the reviewed
CUDA12.6 build, sm_70 compiled kernels and a Tesla V100 (16GB or 32GB), then runs
a tiny FP16 matrix multiplication and verifies its synchronized CPU result.
Enumeration or architecture strings alone cannot establish the kernel check.
This is not Qwen checkpoint numerics or quality acceptance.

NVML is accessed through fixed typed ctypes bindings to `libnvidia-ml.so.1`;
no extra dependency, shell command, caller-selected library or backend override.
The handle is selected by the CUDA device UUID, not by assuming CUDA-visible
indices equal NVML physical indices. UUID and driver strings, memory ranges,
and total/free/used coherence are checked. The v1 `nvmlMemory_t` used counter
includes reserved and allocated device memory; it is a whole-device point
observation, **not a peak** or just a Torch allocator number.

The observation retains the oldest component-read time and the actual
fail-closed host MemorySnapshot. A successful GPU diagnostic cannot turn
unknown complete cgroup scope into a host-memory pass, supply calibration,
or enable model generation. Exact fresh peaks/headroom/identity remain needed.
The observation record's constructor checks data coherence only; direct
construction is not physical hardware proof.

Only query APIs and the tiny kernel are used. No driver install, clocks,
ECC/mode changes, reset, global allocator flush or service/config mutation.
CUDA context creation and NVML target lookup can initialize driver/device
state; NVML may initialize additional GPUs while resolving a UUID. Use an
owned isolated diagnostic worker: driver calls are not hard-deadline bounded,
and process exit owns CUDA-context teardown. NVML initialization is balanced
with shutdown after every owned successful initialization, even on query
failure or interrupt. Shutdown faults refuse successful observations; primary
failures are preserved and worker exit is required for failed library cleanup.

## CPU acceptance, 2026-10-13

61 guarded pure tests cover typed device/profile/kernel ordering, fixed NVML
library and ctypes ABI, whole-device counter coherence, UUID matching, all
query/initialization/shutdown failures, missing symbols, fixed errors and
interrupt preservation, plus FP16 kernel device/dtype/shape/synchronized-value
checks. Positive cases are **fabricated query/kernel spies**,
not V100 evidence. `tools/hardware_refusal_cpu_check.py` proves actual CUDA
unavailability/refusal in this container under offline fixture-only guards;
it forbids a kernel/NVML call and fails if CUDA becomes available.

After installation, actual success-path execution, native driver ABI, CUDA
UUID mapping, ECC-adjusted counters, kernel/architecture support and device
selection must be reviewed on the real card. PCIe generation/width and transfer
throughput, model BF16-storage→FP16 numerics, full peak calibration and quality
are separate pending checks. No default loader or live job endpoint consumes
this diagnostic yet. This module does not remove the CPU encoder's RAM blocker.

Primary API references checked 2026-10-13:

- [NVIDIA NVML memory v1 layout and counter meanings](https://docs.nvidia.com/deploy/nvml-api/latest/api/structnvmlMemory__t.html).
- [NVIDIA device UUID/handle/memory queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html).
- [NVIDIA initialization/shutdown reference ownership](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlInitializationAndCleanup.html).
- [NVIDIA driver-version query](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlSystemQueries.html).

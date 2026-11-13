# Configuration

The CLI and embedded API use the same `Configuration` object. Explicit values
win over defaults; launcher identity (`RANK`, `WORLD_SIZE`, and friends) is
resolved only when an external process group is being used.

## Starting ComfyUI

```console
comfyui                         # serve is the default command
comfyui serve --listen 0.0.0.0
comfyui --cuda-device 0,1 --tensor-parallel-size 2
comfyui run-workflow image_flux2_text_to_image --all
comfyui models list
comfyui workflows list
comfyui nodes list
comfyui env
```

In Python, pass a configuration directly. The embedded client selects thread or
process execution based on its global-state requirements; no `PYTHONPATH`
modification is required:

```python
from comfy.client.embedded_comfy_client import Comfy
from comfy.component_model.configuration import Configuration

configuration = Configuration(cuda_device="0,1", tensor_parallel_size=2)
app = Comfy(configuration=configuration)
```

## Memory and device selection

`guess_settings` is enabled by default. It selects a suitable device, dtype,
attention implementation, and DynamicVRAM policy from the hardware and the
requested model, and on Linux defaults to tensor parallelism across identical
NVIDIA GPUs (see [distributed inference](distributed.md#defaults)). An
explicit CLI or `Configuration` value always wins.

Use `--reserve-vram` to leave a fixed amount available for the desktop or
other workloads. `--novram` is an explicit compatibility escape hatch, not the
normal recommendation; DynamicVRAM can eject dependencies when memory is
needed and works across pipeline stages.

Pinned host memory cannot be reclaimed, so it has one budget, shared by the
DynamicVRAM host buffers and the legacy model patcher. The effective memory
limit is host RAM or, in a container or systemd scope, the lowest cgroup
`memory.max`, `memory.high`, or v1 `memory.limit_in_bytes` of the process's
cgroup and its ancestors. Pinning never takes the reserve,
`--pinned-memory-reserve` GB (default: the larger of 4 GB and 10% of the
limit). The reserve is checked at every pin against memory available at that
moment: `MemAvailable`, capped by each limited cgroup's limit minus its working
set, so memory held by other processes, VMs, or the rest of a pod counts. When
a pin does not fit, idle pins are evicted first; otherwise the weights are used
unpinned. `--max-pinned-memory` caps the total below the limit minus the
reserve.

### Unified memory (Apple Silicon, DGX Spark, AMD APUs)

On Apple MPS and on integrated CUDA or ROCm GPUs (torch reports
`is_integrated`) the GPU allocates from the same RAM as the CPU. ComfyUI
treats these devices as unified memory:

- Free and total device memory are system memory: `MemAvailable`, capped by
  the cgroup as above, plus memory torch has cached for reuse. On a GB10,
  `cudaMemGetInfo` reports `MemFree`, which ignores reclaimable page cache.
- The device is `SHARED`: weights load straight to the device from a
  copy-on-write mapping of the safetensors file, with no anonymous CPU copy,
  and models are never offloaded to the CPU, which would only copy within the
  same RAM. Unloading a model points its weights back at the file mapping,
  so the memory is returned and the next load reads the file again.
- Pinned memory is off, and the pinned-memory reserve is also kept free of
  device allocations, which the OS can neither reclaim nor swap.

`--unified-memory-allocator` / `--no-unified-memory-allocator`
(`COMFYUI_UNIFIED_MEMORY_ALLOCATOR`) controls the CUDA allocator. By default
it is on when every visible device is an integrated NVIDIA GPU with
concurrent managed access, such as the DGX Spark GB10, and off for discrete
GPUs, ROCm, MPS and CPU, and also off for pipeline, tensor and sequence
parallel runs: NGC PyTorch 2.15 aborts when a NCCL process group is destroyed
under a pluggable allocator. When on, every CUDA allocation is managed memory
(`cudaMallocManaged`, preferred location the GPU): pageable RAM the kernel
can reclaim and swap, instead of pinned device allocations. Freed managed
blocks are reused only after a device synchronization and are returned to
the driver whenever ComfyUI empties its caches.

DynamicVRAM is off on unified-memory devices: on a GB10 its VMM pages held
memory at its 7.8 GB floor until the GPU driver's own allocations failed and
the next kernel faulted (Xid 31).

## Distributed configuration

The model-parallel flags are ordinary configuration values:

```console
comfyui --cuda-device 0,1 --tensor-parallel-size 2
comfyui --cuda-device 0,1 --pipeline-parallel-size 2
comfyui --cuda-device 0,1 --ulysses-degree 2
comfyui --cuda-device 0,1 --ring-degree 2
```

For `torchrun` or another launcher, canonical `RANK`, `WORLD_SIZE`,
`LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` are read
alongside common MPI/PMI/Slurm aliases. Do not configure normal application
behavior by inventing additional environment variables.

## Tracing

ComfyUI emits OpenTelemetry spans for workflow execution and sampling. The
OTLP/JSONL exporter destination is configured through the CLI. Sampler spans
exclude custom-node import and checkpoint-load time.

## Embedded applications

The embedded entry point and the server use the same configuration surface:

```python
from comfy.client.embedded_comfy_client import Comfy
from comfy.component_model.configuration import Configuration

app = Comfy(configuration=Configuration(guess_settings=True))
```

Keep model paths, device choices, and feature flags in `Configuration`. CLI startup
and an existing host process can initialize global state differently; see
[embedded isolation](embedded.md#client-lifetime-and-isolation) for the execution
boundary and allocator limitations.

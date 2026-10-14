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

In Python, pass a configuration directly; no setup call, subprocess, or
`PYTHONPATH` modification is required:

```python
from comfy import Comfy
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
from comfy import Comfy
from comfy.component_model.configuration import Configuration

app = Comfy(configuration=Configuration(guess_settings=True))
```

Keep model paths, device choices, and feature flags in `Configuration` so an
embedded run and a CLI run have identical behavior.

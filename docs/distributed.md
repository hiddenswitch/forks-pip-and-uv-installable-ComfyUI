# Performance and multi-GPU inference

Choose a mode for your model and workload. More GPUs do not guarantee a faster
image: latency depends on checkpoint layout, activation size, memory pressure,
attention backend, and interconnect. Pipeline parallelism primarily adds capacity.
For more independent jobs per second, see [worker deployment](deployment.md).

## Choose a mode

| Goal | Mode | Tradeoff |
|---|---|---|
| Share transformer computation | Tensor parallelism (`--tensor-parallel-size`) | Splits projections across GPUs; collectives can outweigh compute savings |
| Fit larger models | Pipeline parallelism (`--pipeline-parallel-size`) | Places contiguous blocks on different devices; stages execute sequentially for each sample |
| Partition sequence attention | Ulysses / Ring (`--ulysses-degree`, `--ring-degree`) | Requires a supported attention layout and communication backend |
| Run independent jobs concurrently | RabbitMQ workers | Each worker needs its own model capacity; does not split a model |

TP, PP, and sequence modes are currently exclusive; hybrid TP+PP and TP+sequence
topologies are not implemented. Normal memory management still controls residency
and offloading. Parallelism does not require `--novram`.

## Defaults

Hardware guessing is enabled by default. On Linux, when all visible NVIDIA GPUs
are the same product, it selects TP across the largest power of two of them:
two GPUs use two ranks, four use four, three use two. `--cuda-device` and
`CUDA_VISIBLE_DEVICES` restrict visibility first. Explicit parallelism flags,
their `COMFYUI_*` environment variables, or an external launcher group suppress
that guess. Windows does not guess TP because its Torch builds lack NCCL.

Use `--tensor-parallel-size 1` to disable TP. To restrict execution to one GPU,
also select one device: `--cuda-device 0`. A model outside the supported TP
families falls back with a warning to the PP or ordinary loader. A supported
model with an invalid requested topology reports an error.

`comfyui env check` reports device connectivity, peer versus host-mediated copies,
and supported parallel sizes. Use its findings when selecting devices.

## Supported model families

| Model family | TP | PP | Ulysses/Ring |
|---|:---:|:---:|:---:|
| Flux2 Dev | Yes | Yes | Yes |
| Ideogram 4 | Yes | — | — |
| Krea 2 | Yes | — | — |
| MiniMax H3 | Yes | Yes | Yes |
| Qwen Image / Layered | — | Yes | Yes |
| Kandinsky 6 Pro custom node | Yes | — | — |
| HunyuanImage3 custom node | Yes, with expert parallelism | Yes | — |

TP requires safetensors checkpoints and CUDA/NCCL on one host, using the internal
multiprocessing executor. Kandinsky 6 TP requires unquantized weights. Native
quantized TP preserves the quantizer's input domain by gathering complete inputs
before output-channel shards; checkpoint layout helpers handle packed projections,
scalar/channel/blocked scales, and tiled SVDQuant weights. Hunyuan's streaming
loader preserves per-expert quantization data.

PP uses DynamicVRAM-aware block partitioning. Hunyuan PP supports diffusion with
supplied captions; autoregressive prompt rewriting and Spectrum acceleration are
not supported by its pipeline stages. TP supports the separate autoregressive path.
Family support does not imply every checkpoint, custom patch, or quantizer works.

## Examples

No special workflow node is needed. Apply the mode to the normal runner:

```console
comfyui serve --cuda-device 0,1 --tensor-parallel-size 2
comfyui serve --cuda-device 0,1 --pipeline-parallel-size 2
comfyui serve --cuda-device 0,1 --ulysses-degree 2
comfyui serve --cuda-device 0,1 --ring-degree 2
```

The same options apply to `run-workflow` and workers. Select a workflow from a
supported family before enabling a mode.

External launch supports PP only. Canonical launcher variables are `RANK`,
`WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`.
Common PMI, Open MPI, MVAPICH, and Slurm aliases are also accepted. Launcher
variables identify processes; CLI/Configuration values select model parallelism.
Multi-node TP is not supported.

## Memory and limitations

Keep a server or embedded client alive across prompts to reuse resident models
and node caches. Separate CLI processes cannot share those caches. A changed
model, LoRA, execution configuration, or memory pressure can invalidate residency.

DynamicVRAM is shared with upstream. This fork additionally accounts for native
quantized payloads and LoRA materialization, coordinates model-manager access,
and budgets host-memory pinning against available memory and container limits.
`--max-pinned-memory` caps pinned memory in GiB; `--pinned-memory-reserve` leaves
host headroom. Model-parallel processes share the pinning budget. Quantized
storage reduces weight capacity requirements but does not guarantee faster math.
See [quantization contracts](../QUANTIZATION.md).

Pipeline boundaries begin with checkpoint geometry and are refined against
DynamicVRAM geometry on each device. Text encoders and other dependencies can
be ejected by the normal memory manager. Reduced offloading can improve latency,
but sequential pipeline stages do not execute different layers of one sample
simultaneously. Slow checkpoint storage can dominate startup and streaming.

Start with the default memory policy. `--novram` disables DynamicVRAM and smart
memory in the CLI; use it deliberately, rather than adding it to every command.
Optional attention kernels and `--compile` have hardware, shape, and warm-up
costs. Compare your own complete workload with consistent settings; there is no
universal speedup multiplier.

## Distributed prompt queue

Complete RabbitMQ, worker, frontend, and shared-storage instructions are in
[Frontend and worker deployment](deployment.md). Queue distribution increases
independent-job throughput and is separate from the model-parallel modes above.

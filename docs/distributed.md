# Distributed inference

ComfyUI selects distributed execution from the normal `Configuration`. No
special workflow node is required: choose devices and a parallelism mode on
the command line, or pass the same values to `Comfy(configuration=...)`.

## Choose a mode

| Goal | Mode | Use when |
|---|---|---|
| Faster transformer blocks | Tensor parallelism (`--tensor-parallel-size`) | The model fits when weights and activations are partitioned across GPUs |
| Fit a model that does not fit on one GPU | Pipeline parallelism (`--pipeline-parallel-size`) | DynamicVRAM needs to place contiguous blocks on different devices |
| Faster sequence attention | Ulysses or Ring (`--ulysses-degree` / `--ring-degree`) | The model family has xDiT sequence-parallel support |

These modes are currently exclusive. Hybrid TP+PP and TP+sequence topologies
are not implemented yet. The normal ComfyUI memory manager remains in charge
of residency and offloading; parallelism does not imply `--novram`.

## Defaults

With `guess_settings` on (the default), a Linux machine whose visible NVIDIA
GPUs are all the same product defaults to tensor parallelism across the largest
power of two of them: two identical GPUs run `--tensor-parallel-size 2`, four
run 4, three run 2. `--cuda-device` and `CUDA_VISIBLE_DEVICES` narrow the group
first. Any explicit `--tensor-parallel-size`, `--pipeline-parallel-size`,
`--ulysses-degree`, or `--ring-degree` value, the matching `COMFYUI_*`
environment variable, or a launcher process group turns the guess off;
`--tensor-parallel-size 1` restores the single-device loader. Windows never
guesses tensor parallelism because its torch builds ship without NCCL.

A checkpoint outside the tensor-parallel families loads through pipeline
parallelism when its family supports that and multiple devices are visible,
and through the ordinary single-device loader otherwise; the fallback is
logged as a warning rather than failing the load.

`comfyui env check` measures device-to-device copy bandwidth between every
pair of visible CUDA devices, reports whether each pair uses peer access or
goes through host memory, and prints the largest tensor-parallel and
pipeline-parallel sizes this machine supports.

## Supported model families

| Model family | TP | PP | Ulysses/Ring |
|---|:---:|:---:|:---:|
| Flux2 Dev | ✅ | ✅ | ✅ |
| Ideogram 4 | ✅ | — | — |
| Krea 2 | ✅ | — | — |
| MiniMax H3 | ✅ | ✅ | ✅ |
| Qwen Image / Layered | — | ✅ | ✅ |
| Kandinsky 6 Pro (pip custom node) | ✅ | — | — |
| HunyuanImage3 (`comfyui-hunyuanimage3` node) | ✅, with EP | ✅ | — |

TP currently uses safetensors checkpoints and CUDA/NCCL. PP supports the
listed block layouts and uses DynamicVRAM-aware partitioning. Sequence
parallelism is provided by the xDiT operations backend. Unsupported models
continue through the ordinary single-device loader.

For quantized native TP models, row projections gather the complete input
before applying output-channel shards, preserving the quantizer's input
domain. Checkpoint sharding handles packed projection sections, scalar and
channel scales, blocked scales, and tiled SVDQuant weights through shared
layout helpers. HunyuanImage3's streaming loader uses the same helpers for
dense projections and preserves per-expert quantization data. Kandinsky 6
TP currently requires unquantized weights.

## Examples

```console
# Two GPUs, transparent tensor parallelism.
comfyui --cuda-device 0,1 --tensor-parallel-size 2

# Two GPUs, memory-aware pipeline partitioning.
comfyui --cuda-device 0,1 --pipeline-parallel-size 2

# xDiT sequence parallelism.
comfyui --cuda-device 0,1 --ulysses-degree 2
comfyui --cuda-device 0,1 --ring-degree 2
```

External launch currently supports PP only. TP uses the internal multiprocessing
executor with GPUs on one host; a multi-node TP JobSet is not supported.

For an externally launched PP process group, use canonical launcher values
`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, and
`MASTER_PORT` (for example with `torchrun`). Common PMI, Open MPI, MVAPICH,
and Slurm aliases are also accepted. Model-parallel choices still come from
`Configuration`/CLI; launcher variables identify the process.

## Getting useful performance from multiple GPUs

Tensor parallelism shares supported model computations across GPUs, which can
reduce generation time. Pipeline parallelism primarily helps fit larger models
and reduce CPU offloading. Neither mode guarantees a speedup: model size,
quantization, available VRAM, and communication bandwidth determine the result.

Start with `comfyui env check` to inspect GPU connectivity. Compare the same
workflow, checkpoint, precision, and sampling settings on one GPU and your
chosen topology. Keep model loading separate from sampling when comparing
latency. A fast interconnect helps; transfers through host memory can erase the
benefit of additional GPUs.

For more independent jobs per second, use [worker queues](workers.md) instead
of dividing one generation across devices.

## Memory and limitations

Pipeline stages execute sequentially for each sample. Partitioning can reduce
offloading enough to improve latency, but it does not by itself make both GPUs
compute different layers of that sample simultaneously.

Hunyuan PP currently supports diffusion with supplied captions; its autoregressive
prompt rewriting and Spectrum acceleration are not supported by the pipeline
stages. TP supports the separate autoregressive path.

Pipeline boundaries are selected from checkpoint geometry and then refined
using measured DynamicVRAM geometry on every participating device. Text
encoders and other dependencies can be ejected by the same general memory
manager used on one GPU; they are not hard-coded to a particular model.

TP, PP, and sequence parallelism require the selected devices and process
group to be visible to the chosen backend. If a mode cannot be constructed,
ComfyUI reports the reason and does not silently change the requested topology.

## Worker queues

See [worker deployment](workers.md) to distribute independent workflows across machines.

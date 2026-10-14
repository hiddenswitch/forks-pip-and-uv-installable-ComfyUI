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

## Recorded performance

These historical measurements use two NVLink-connected NVIDIA RTX A5000
24 GiB GPUs; the baseline uses one of those GPUs. They are sampler averages
or complete workflow timings as labeled, **not second-step timings**. GPU 0
also served desktop graphics. Results describe the recorded checkpoint and
workload, not a guarantee for every model variant or memory setting.

| Model / workload | GPU configuration | Mode | Attention | Measurement | One GPU | Two GPUs | Speedup |
|---|---|---|---|---|---:|---:|---:|
| Krea 2 Turbo, INT8 ConvRot, 1024², 8 steps | 2 × RTX A5000 24 GiB, NVLink | TP=2 | SageAttention | Mean sampler seconds/step | 1.170 | 0.876 | 1.34× |
| Ideogram 4, INT8 ConvRot conditional + unconditional, 1024², 20 steps | 2 × RTX A5000 24 GiB, NVLink | TP=2 | SageAttention | Mean sampler seconds/step | 2.754 | 1.823 | 1.51× |
| Flux2 Dev, full INT8 ConvRot, 1024², 20 steps | 2 × RTX A5000 24 GiB, NVLink | TP=2 | SageAttention | Mean sampler seconds/step | 4.018 | 1.662 | 2.42× |
| MiniMax H3, pruned INT8 ConvRot, 864×480×124, 20 steps | 2 × RTX A5000 24 GiB, NVLink | TP=2 | SageAttention | Mean sampler seconds/step | 9.497 | 6.272 | 1.51× |
| HunyuanImage3 Instruct W4A8, fixed caption, 1024², 50 steps | 2 × RTX A5000 24 GiB, NVLink | EP + TP=2 | Comfy Kitchen INT8 | Complete sampler seconds | 316.99 | 201.00 | 1.58× |
| Kandinsky 6 Pro distilled T2AV, 864×480×121, 10 steps | 2 × RTX A5000 24 GiB, NVLink | TP=2 | Not recorded in summary | Cold workflow seconds | 822.04 | 573.38 | 1.43× |
| Kandinsky 6 Pro distilled I2AV | 2 × RTX A5000 24 GiB, NVLink | TP=2 | Not recorded in summary | Cold workflow seconds | 846.17 | 570.27 | 1.48× |
| Kandinsky 6 Pro standard T2AV, generation without SR | 2 × RTX A5000 24 GiB, NVLink | TP=2 | Not recorded in summary | Cold workflow seconds | 7172.74 | 7817.27 | 0.92× |
| Kandinsky 6 Pro standard I2AV, generation without SR | 2 × RTX A5000 24 GiB, NVLink | TP=2 | Not recorded in summary | Cold workflow seconds | 7391.41 | 8344.53 | 0.89× |

The first four rows were retained in [the earlier benchmark table](https://github.com/hiddenswitch/pip-and-uv-installable-ComfyUI/blob/94c047692aec7ebbdd2dca5b11309f3061f1031f/docs/distributed.md).
The complete-run evidence is in
[Hunyuan diffusion validation](../release-artifacts/hunyuan-diffusion-tp-fix-validation.json),
[Kandinsky distilled T2AV](../release-artifacts/kandinsky-pro-distil-benchmark.json),
[distilled I2AV](../release-artifacts/kandinsky6_pro_distilled_i2av-final-benchmark.json),
[standard T2AV](../release-artifacts/kandinsky6_pro_standard_t2av-cold-validation.json),
and [standard I2AV](../release-artifacts/kandinsky6_pro_standard_i2av-cold-validation.json).
The complete-run rows each compare one cold process per configuration; the OS
file cache was not flushed. Hunyuan and distilled Kandinsky reserve 2 GiB of
extra VRAM headroom. Hunyuan W4A8 means 4-bit weights and 8-bit activations.
Standard Kandinsky Pro was slower with TP=2 in these samples.

The following PP measurements use the same 1024² HunyuanImage3 Instruct W4A8
fixed-caption workflow reduced to two Euler steps, CFG 2.5, seed 1234. They time
the **second completed step**, with PyTorch SDPA observed at runtime.

| Model | GPU configuration | Mode / transport | Checkpoint storage | Step 2 (s) | it/s |
|---|---|---|---|---:|---:|
| HunyuanImage3 Instruct W4A8 | 2 × RTX 3090 24 GiB, one per host | PP=2, NCCL / USB4 ibverbs | Node-local disk | 4.937 | 0.203 |
| HunyuanImage3 Instruct W4A8 | 2 × RTX 3090 24 GiB, one per host | PP=2, NCCL / USB4 ibverbs | SeaweedFS network mount | 18.323 | 0.055 |

[Recorded PP evidence](../release-artifacts/hunyuan-pp2-usb4-validation.json)
includes revisions and runtime details. Each row is one fresh process; the OS
file cache was not flushed. The runs use different neighboring host pairs and
have no matching single-GPU baseline, so they do not establish a PP speedup.
Two-step output validates execution, not image quality. A separate 50-step
fresh-process PP=2 run completed and its saved image visually matched the fox,
grass, golden light, and bokeh in the supplied caption; this is one inspected
sample, not a general quality comparison.

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

## Distributed prompt queue

Prompt distribution is separate from model parallelism. A frontend serves the
normal ComfyUI API and submits complete prompts to RabbitMQ; one or more workers
consume those prompts and execute them. Websocket progress and results are
forwarded through the frontend.

Start RabbitMQ on a machine reachable by every frontend and worker. Use a
dedicated account because RabbitMQ does not permit remote use of its default
`guest` account:

```console
docker run --rm --name comfyui-rabbitmq -p 5672:5672 \
  -e RABBITMQ_DEFAULT_USER=comfyui \
  -e RABBITMQ_DEFAULT_PASS=change-me \
  rabbitmq:4
```

Replace the example password in both the container configuration and connection
URI. Start any number of workers with the same queue URI:

```console
comfyui worker \
  --distributed-queue-connection-uri amqp://comfyui:change-me@10.1.0.100
```

All ordinary configuration options are available to a worker. For example, set
its workspace and GPU selection normally:

```console
comfyui worker \
  --cwd /mnt/comfy-shared \
  --cuda-device 0 \
  --distributed-queue-connection-uri amqp://comfyui:change-me@10.1.0.100
```

Start the API/frontend role with queue forwarding enabled:

```console
comfyui serve \
  --listen 0.0.0.0 \
  --cwd /mnt/comfy-shared \
  --distributed-queue-connection-uri amqp://comfyui:change-me@10.1.0.100 \
  --distributed-queue-frontend
```

Workers and frontends must see the same input and output contents. The paths do
not need to be textually identical, but the frontend's input/output directories
must expose the files referenced by workers. A shared `--cwd` is the simplest
layout. Models may instead be stored locally on every worker or configured with
`--extra-model-paths-config`; they do not need to be readable by a frontend that
does not execute prompts.

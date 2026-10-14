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
| HunyuanImage 3.0 (`comfyui-hunyuanimage3` node) | — | ✅ | — |
| Ideogram 4 | ✅ | — | — |
| Krea 2 | ✅ | — | — |
| MiniMax H3 | ✅ | ✅ | ✅ |
| Qwen Image / Layered | — | ✅ | ✅ |

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

For an externally launched process group, use canonical launcher values
`RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`, `MASTER_ADDR`, and
`MASTER_PORT` (for example with `torchrun`). Common PMI, Open MPI, MVAPICH,
and Slurm aliases are also accepted. Model-parallel choices still come from
`Configuration`/CLI; launcher variables identify the process.

## Collected TP benchmark

These are sampler spans, not process startup or checkpoint-loading time. The
workload used two NVLink-connected RTX A5000 24GB GPUs, `--fast
fp16_accumulation`, SageAttention, and the stated INT8 ConvRot checkpoint.

| Model | Resolution | Frames | Steps | Weights | Attention | TP=1 | TP=2 | Speedup |
|---|---:|---:|---:|---|---|---:|---:|---:|
| Krea 2 Turbo | 1024×1024 | — | 8 | INT8 ConvRot | SageAttention | 1.170 s/it | 0.876 s/it | 1.34× |
| Ideogram 4 | 1024×1024 | — | 20 | INT8 ConvRot conditional + unconditional | SageAttention | 2.754 s/it | 1.823 s/it | 1.51× |
| Flux2 Dev | 1024×1024 | — | 20 | Full INT8 ConvRot | SageAttention | 4.018 s/it | 1.662 s/it | 2.42× |
| MiniMax H3 | 864×480 | 124 | 20 | Pruned INT8 ConvRot | SageAttention | 9.497 s/it | 6.272 s/it | 1.51× |

To reproduce a measurement, run the same workflow once for warm-up, enable
sampler OpenTelemetry JSONL output, and compare the sampler span rather than
end-to-end startup. See [configuration](configuration.md#tracing-and-benchmarking).

## Memory and limitations

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

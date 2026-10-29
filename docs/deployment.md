# Frontend and worker deployment

A frontend serves the ComfyUI API and forwards complete prompts to RabbitMQ.
Workers consume jobs and execute them; progress and results return through the
frontend. Scale workers for throughput. To split one model across devices, use
[model parallelism](distributed.md) inside a worker instead.

## Start the broker

Run RabbitMQ on a host reachable by every worker and frontend. Use a dedicated
account: RabbitMQ's default guest account does not allow remote connections.
Replace `change-me` in both the broker configuration and the connection URI.

```console
docker run --rm --name comfyui-rabbitmq -p 5672:5672 \
  -e RABBITMQ_DEFAULT_USER=comfyui \
  -e RABBITMQ_DEFAULT_PASS=change-me \
  rabbitmq:4
```

## Start workers

Install the same required nodes on each worker, then start each with the same
queue URI. Choose separate GPUs when running independent workers on one host:

```console
comfyui worker \
  --cwd /mnt/comfy-shared \
  --cuda-device 0 \
  --distributed-queue-connection-uri amqp://comfyui:change-me@10.1.0.100
```

Worker commands accept ordinary runtime configuration. A worker may use multiple
GPUs for a supported model, but that reserves those devices for its execution.

## Start the frontend

```console
comfyui serve \
  --listen 0.0.0.0 \
  --cwd /mnt/comfy-shared \
  --distributed-queue-connection-uri amqp://comfyui:change-me@10.1.0.100 \
  --distributed-queue-frontend
```

## Shared storage

Workers and frontends must expose the same input and output contents. Paths need
not be textually identical, but the frontend must be able to serve files named
in worker results. A shared workspace is the simplest arrangement. Broker access
alone does not transfer input/output files or install model dependencies.

Model files may live on local worker disks or in paths configured with
`--extra-model-paths-config`; a frontend that only forwards prompts does not need
the model weights. Keep node versions and workflow definitions compatible across
the fleet. For accelerator images and volume mounts, see [containers](docker.md).

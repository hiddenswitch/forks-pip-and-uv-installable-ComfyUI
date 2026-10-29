# Scale workflow throughput with workers


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

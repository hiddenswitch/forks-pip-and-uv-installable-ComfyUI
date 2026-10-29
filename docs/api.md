# Application integration

Choose [embedded execution](embedded.md) to run workflows inside a Python
application without an HTTP server. Use a remote client when an existing ComfyUI
server owns execution. For a shell script, start with [headless workflows](workflows.md).

## Remote Python client

Install the [package](installing.md), start `comfyui serve`, and save a complete
workflow to `workflow.json`. Both UI and API workflow formats are accepted by
the fork's `/api/v1/prompts` endpoint.

```python
import asyncio
import json
from pathlib import Path
from comfy.client.aio_client import AsyncRemoteComfyClient


async def main():
    workflow = json.loads(Path("workflow.json").read_text(encoding="utf-8"))
    async with AsyncRemoteComfyClient("http://localhost:8188") as client:
        result = await client.queue_prompt_api(workflow)
        print(result.prompt_id)
        print(result.urls)
        print(result.outputs)


if __name__ == "__main__":
    asyncio.run(main())
```

The context manager closes the HTTP session. `queue_prompt_api()` returns a
structured response; `queue_prompt()` instead requests PNG bytes. Use
`queue_and_forget_prompt_api()` and `poll_prompt_until_done()` when submission and
completion must be separate. Polling has a finite timeout; choose it for your job
duration. Returned paths describe the server's filesystem, while URLs are intended
for remote retrieval.

## REST

Send the complete workflow JSON as the body, not a fragment with placeholder nodes:

```console
curl --fail-with-body http://localhost:8188/api/v1/prompts \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  --data-binary @workflow.json
```

For an image-producing workflow, request a PNG response:

```console
curl --fail-with-body http://localhost:8188/api/v1/prompts \
  -H 'Content-Type: application/json' \
  -H 'Accept: image/png' \
  --data-binary @workflow.json --output output.png
```

`Prefer: respond-async` requests asynchronous submission. The response includes a
prompt ID that can be polled at `/api/v1/prompts/{prompt_id}`. This fork's endpoint
is distinct from the upstream-compatible `/prompt` route and its envelope.

## OpenAPI Spec for Vanilla API, Typed Clients

[openapi.yaml](../openapi.yaml) is the canonical specification for both
upstream-compatible routes and the fork's additional APIs. Generate typed clients
from it and use its request/response schemas rather than private server internals.

## Embedded

[Using ComfyUI as a library](embedded.md) covers client lifetime, progress,
previews, workflow conversion, and execution isolation, with a complete workflow
example.

## RabbitMQ / AMQP Support

[Queue deployment](deployment.md) describes separate frontend and worker
processes, broker configuration, and shared storage.

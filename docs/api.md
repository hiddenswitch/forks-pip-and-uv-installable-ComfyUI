# Using ComfyUI as an API / Programmatically

There are multiple ways to use this ComfyUI package to run workflows programmatically:

### Embedded

Start ComfyUI by creating an ordinary Python object. This does not create a web server. It runs ComfyUI as a library, like any other package you are familiar with:

```python
from comfy.client.embedded_comfy_client import Comfy

async with Comfy() as client:
    outputs = await client.queue_prompt(prompt)
    save_image_node_id = next(key for key in prompt if prompt[key]["class_type"] == "SaveImage")
    print(outputs[save_image_node_id]["images"][0]["abs_path"])
```

See [script_examples/basic_api_example.py](examples/script_examples/basic_api_example.py) for a complete example.

Preview can be retrieved too:

```python
prompt_dict = copy.deepcopy(_PROMPT_FROM_WEB_UI)
prompt_dict["6"]["inputs"]["text"] = "masterpiece best quality man"
...
preview_frames = []
async with Comfy() as client:
    task = client.queue_with_progress(prompt)
    async for notification in task.progress():
        if notification.event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
            image_data: PreviewImageWithMetadataMessage = notification.data
            unencoded_preview, _ = image_data
            
            preview_frames.append(unencoded_preview.pil_image.copy())
```

See more in Colab: https://colab.research.google.com/drive/1Gd9F8iYRJW-LG8JLiwGTKLAcXLJ5eH78#scrollTo=mP_72JH6v1BK


### Remote

Visit the ComfyUI interface and design your workflow. You can export it using either **Save** (UI format) or **Save (API Format)**. Both formats are accepted — UI workflows are automatically converted to API format.

You can use the built-in Python client library by installing this package without its dependencies.

```shell
uv pip install aiohttp
uv pip install --no-deps --extra-index-url https://nodes.appmana.com/simple/ comfyui
```

Then the following idiomatic pattern is available:

```python
from comfy.client.aio_client import AsyncRemoteComfyClient

client = AsyncRemoteComfyClient(server_address="http://localhost:8188")
png_image_bytes = await client.queue_prompt(prompt)
with open("image.png", "wb") as f:
    f.write(png_image_bytes)
```

See [script_examples/remote_api_example.py](examples/script_examples/remote_api_example.py) for a complete example.

##### REST API

First, install this package using the [Installation Instructions](installing.md). Then, run `comfyui`.

Visit the ComfyUI interface and design your workflow. Export it using either **Save** (UI format) or **Save (API Format)** — both are accepted.

Then, send a request to `api/v1/prompts`. Here are some examples:

**`curl`**:

```shell
curl -X POST "http://localhost:8188/api/v1/prompts" \
     -H "Content-Type: application/json" \
     -H "Accept: image/png" \
     -o output.png \
     -d '{
       "prompt": {
         # ... (include the rest of the workflow)
       }
     }'
```

**Python**:

```python
import requests

url = "http://localhost:8188/api/v1/prompts"
headers = {
    "Content-Type": "application/json",
    "Accept": "image/png"
}
workflow = {
    "4": {
        "inputs": {
            "ckpt_name": "sd_xl_base_1.0.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    # ... (include the rest of the workflow)
}

payload = {"prompt": workflow}

response = requests.post(url, json=payload, headers=headers)
```

**Javascript (Browser)**:

```javascript
async function generateImage() {
    const prompt = "a man walking on the beach";
    const workflow = {
        "4": {
            "inputs": {
                "ckpt_name": "sd_xl_base_1.0.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        // ... (include the rest of the workflow)
    };

    const response = await fetch('http://localhost:8188/api/v1/prompts', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'image/png'
        },
        body: JSON.stringify({prompt: workflow})
    });

    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);
    const img = document.createElement('img');
    img.src = imageUrl;
    document.body.appendChild(img);
}

generateImage().catch(console.error);
```

You can use the OpenAPI specification file to learn more about all the supported API methods.

### OpenAPI Spec for Vanilla API, Typed Clients

Use a typed, generated API client for your programming language and access ComfyUI server remotely as an API.

You can generate the client from [openapi.yaml](../openapi.yaml). The root
spec is the canonical OpenAPI document for both upstream-compatible ComfyUI
routes and this fork's additional REST APIs.

### RabbitMQ / AMQP Support

Submit jobs directly to a distributed work queue. This package supports AMQP message queues like RabbitMQ. You can submit workflows to the queue, including from the web using RabbitMQ's STOMP support, and receive realtime progress updates from multiple workers. See [distributed inference](distributed.md#distributed-prompt-queue) for the queue architecture.

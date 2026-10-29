# Using ComfyUI as a Library

ComfyUI can run workflows as a library without starting an HTTP server. The
default executor uses a thread; when global configuration requires isolation,
the client can select a subprocess instead.

See the [README](../README.md) for installation and getting started.

## Installing

```shell
uv pip install --torch-backend=auto --extra-index-url https://nodes.appmana.com/simple/ comfyui
```

`--torch-backend=auto` installs the correct `torch`, `torchvision`, and `torchaudio` for your platform. Pin the installed Torch build with resolver constraints when preserving an existing accelerator stack.

## Running a Workflow

Save a workflow from the ComfyUI web UI as a JSON file. You can use either the standard Save (UI format) or Save -> API Format — `queue_prompt` accepts both and automatically converts UI workflows to API format. Load JSON with `json.loads()`; JSON booleans and null are not Python literals. The equivalent Python dictionary looks like:

```python
from comfy.client.embedded_comfy_client import Comfy
from comfy.cli_args import default_configuration
import copy

WORKFLOW = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"},
    },
    "2": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 512, "height": 512, "batch_size": 1},
    },
    "3": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "bad hands", "clip": ["1", 1]},
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "masterpiece best quality girl", "clip": ["1", 1]},
    },
    "5": {
        "class_type": "KSampler",
        "inputs": {
            "seed": 8566257,
            "steps": 20,
            "cfg": 8,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1,
            "model": ["1", 0],
            "positive": ["4", 0],
            "negative": ["3", 0],
            "latent_image": ["2", 0],
        },
    },
    "6": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
    },
    "7": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI_API", "images": ["6", 0]},
    },
}


async def run_example():
    prompt = copy.deepcopy(WORKFLOW)
    prompt["4"]["inputs"]["text"] = "masterpiece best quality man"
    prompt["5"]["inputs"]["seed"] = 5

    # auto-detect GPU type, VRAM mode, and attention backend
    config = default_configuration()
    config.guess_settings = True

    async with Comfy(configuration=config) as client:
        outputs = await client.queue_prompt(prompt)

        save_image_node_id = next(
            key for key in prompt if prompt[key]["class_type"] == "SaveImage"
        )
        return outputs[save_image_node_id]["images"][0]["abs_path"]
```

Run it with:

```python
import asyncio

path = asyncio.run(run_example())
print(path)
```

Registered, accessible models referenced in the workflow can be downloaded
automatically. Unknown filenames need a local model or a registered source.

## Building Workflows Programmatically

Use `GraphBuilder` to construct workflows in code instead of pasting JSON:

```python
from comfy_execution.graph_utils import GraphBuilder


def build_graph(positive_prompt_text="masterpiece best quality girl"):
    builder = GraphBuilder()

    checkpoint_loader = builder.node(
        "CheckpointLoaderSimple",
        ckpt_name="v1-5-pruned-emaonly.safetensors",
    )

    empty_latent = builder.node(
        "EmptyLatentImage",
        width=512,
        height=512,
        batch_size=1,
    )

    negative_prompt = builder.node(
        "CLIPTextEncode",
        text="bad hands",
        clip=checkpoint_loader.out(1),
    )

    positive_prompt = builder.node(
        "CLIPTextEncode",
        text=positive_prompt_text,
        clip=checkpoint_loader.out(1),
    )

    k_sampler = builder.node(
        "KSampler",
        seed=8566257,
        steps=20,
        cfg=8,
        sampler_name="euler",
        scheduler="normal",
        denoise=1,
        model=checkpoint_loader.out(0),
        positive=positive_prompt.out(0),
        negative=negative_prompt.out(0),
        latent_image=empty_latent.out(0),
    )

    vae_decode = builder.node(
        "VAEDecode",
        samples=k_sampler.out(0),
        vae=checkpoint_loader.out(2),
    )

    builder.node(
        "SaveImage",
        filename_prefix="ComfyUI_API",
        images=vae_decode.out(0),
    )

    return builder


builder = build_graph()
prompt = builder.finalize()
```

The `finalize()` output is identical to the API format JSON — pass it to `client.queue_prompt(prompt)`.

## Converting UI Workflows to API Format

`queue_prompt` accepts both API and UI format workflows and converts automatically. You can also convert explicitly:

```python
import json
from comfy.component_model.workflow_convert import is_ui_workflow, convert_ui_to_api

workflow = json.loads(open("my_workflow.json").read())

if is_ui_workflow(workflow):
    api_workflow = convert_ui_to_api(workflow)
    print(json.dumps(api_workflow, indent=2))
```

This is useful for batch-converting saved UI workflows to API format, or for inspecting the conversion output.

## Streaming Progress and Previews

Use `queue_with_progress` to receive preview images during inference:

Progress callbacks currently work with the thread executor. The process-executor
path does not forward them, so do not rely on preview/progress events when using
process isolation. Final workflow results remain available.

```python
import copy
from comfy.client.embedded_comfy_client import Comfy
from comfy.api.components.schema.prompt import Prompt
from comfy.component_model.queue_types import BinaryEventTypes


async def run_with_previews():
    prompt = Prompt.validate(copy.deepcopy(WORKFLOW))

    async with Comfy() as client:
        task = client.queue_with_progress(prompt)

        async for notification in task.progress():
            if notification.event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
                image_data, metadata = notification.data
                # image_data.pil_image is a PIL Image of the current denoising step
                print(f"Preview: {image_data.pil_image.size}")

        result = await task.get()
        save_image_node_id = next(
            key for key, value in prompt.items() if value.class_type == "SaveImage"
        )
        return result.outputs[save_image_node_id]["images"][0]["abs_path"]
```

## Running the Web UI Programmatically

You can also start the full ComfyUI web server from Python:

```python
from comfy.cmd.main import entrypoint

entrypoint()
```

Or in Google Colab with a tunnel:

```python
from comfy.app.colab import start_server_in_colab

start_server_in_colab()
```

## Configuring Performance Options

Use `default_configuration()` to create a `Configuration` object, then set attributes or call `.update()` to configure performance options before passing it to `Comfy()`:

```python
from comfy.client.embedded_comfy_client import Comfy
from comfy.cli_args import default_configuration
from comfy.cli_args_types import PerformanceFeature

config = default_configuration()

# Enable SageAttention (requires sageattention package installed)
config.use_sage_attention = True

# Enable cuBLAS ops for faster matrix multiplications (NVIDIA Ampere+ GPUs)
config.fast = {PerformanceFeature.CublasOps}

# Use the normal hardware-aware DynamicVRAM policy
config.guess_settings = True

# Run VAE in full precision (recommended for AMD GPUs)
config.fp32_vae = True

# Disable custom nodes for faster startup and isolation
config.disable_all_custom_nodes = True

config.update({
    "use_sage_attention": True,
    "guess_settings": True,
    "fast": {PerformanceFeature.CublasOps},
})

async with Comfy(configuration=config) as client:
    outputs = await client.queue_prompt(prompt)
```

Available `PerformanceFeature` values for `config.fast`:

- `PerformanceFeature.CublasOps` — Use cuBLAS for supported operations. Recommended for NVIDIA Ampere (RTX 30xx) and newer GPUs.
- `PerformanceFeature.Fp16Accumulation` — Use FP16 accumulation. May reduce quality.
- `PerformanceFeature.Fp8MatrixMultiplication` — Use FP8 matrix multiplication.
- `PerformanceFeature.AutoTune` — Enable PyTorch autotuning.

## Running Multiple Workflows

To run a list of workflows programmatically, iterate over them with the same `Comfy` client:

```python
import json
from pathlib import Path
from comfy.client.embedded_comfy_client import Comfy
from comfy.api.components.schema.prompt import Prompt
from comfy.cli_args import default_configuration

config = default_configuration()
config.guess_settings = True

workflow_dir = Path("./my_workflows")

async with Comfy(configuration=config) as client:
    for workflow_path in sorted(workflow_dir.glob("*.json")):
        workflow = json.loads(workflow_path.read_text())
        outputs = await client.queue_prompt(workflow)
        print(f"{workflow_path.name}: {outputs}")
```

## Headless Workflow Execution with `run-workflow`

See [headless workflows](workflows.md) for CLI execution, workflow-specific help,
parameter overrides, JSON output, and dependency preparation.

## Client lifetime and isolation

Keep one `Comfy` client open across related prompts to reuse eligible resident
weights and caches. The default thread path serializes execution even if its
executor has multiple workers. It shares the host process's global library state
and does not replace the host's CUDA allocator.

The client can select `ProcessPoolExecutor` when configuration differs from
already initialized global state. To request isolation explicitly:

```python
from comfy.client.embedded_comfy_client import Comfy
from comfy.distributed.process_pool_executor import ProcessPoolExecutor


async def run_isolated(workflow):
    with ProcessPoolExecutor(max_workers=1) as executor:
        async with Comfy(executor=executor) as client:
            return await client.queue_prompt(workflow)
```

Call this from an application with a guarded `if __name__ == "__main__":` entry
point so child processes can import it safely. Workers are reused across prompts;
process isolation does not mean a new process per workflow. Global configuration
changes can recycle an owned executor. An externally supplied executor remains
the caller's responsibility and is not automatically replaced.

Exiting the client waits for pending work and cleans up its resources. Closing the
process executor ends its workers; that is the boundary that releases their process
state. For per-test isolation, create and close an executor per test.

## Adding Known Models for Automatic Download

Use `add_known_models()` to register models that should be downloaded automatically from Hugging Face when a workflow references them. This makes models appear in the UI dropdown and triggers on-demand downloads.

```python
from comfy.model_downloader import add_known_models
from comfy.model_downloader_types import HuggingFile, CivitFile

add_known_models("checkpoints", HuggingFile(
    "stabilityai/stable-diffusion-xl-base-1.0",
    "sd_xl_base_1.0.safetensors"
))

add_known_models("loras", HuggingFile(
    "ByteDance/Hyper-SD",
    "Hyper-SDXL-12steps-CFG-lora.safetensors"
))

# CivitAI uses model_id and model_version_id
add_known_models("checkpoints", CivitFile(
    model_id=133005,
    model_version_id=357609,
    filename="juggernautXL_v9Rundiffusionphoto2.safetensors"
))

# save_with_filename renames generic filenames on disk
add_known_models("controlnet", HuggingFile(
    "jschoormans/controlnet-densepose-sdxl",
    "diffusion_pytorch_model.safetensors",
    save_with_filename="controlnet-densepose-sdxl.safetensors"
))

add_known_models("diffusion_models",
    HuggingFile("black-forest-labs/FLUX.1-schnell", "flux1-schnell.safetensors"),
    HuggingFile("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors"),
)
```

The first argument is the folder name (matching ComfyUI's model directory structure: `checkpoints`, `loras`, `controlnet`, `vae`, `clip`, `diffusion_models`, `upscale_models`, etc.).

### How Model Downloads Work

When a workflow references a registered model filename, ComfyUI resolves local
files and downloads missing Hugging Face files through `hf_hub_download`.
The Hugging Face cache is used by default; model lookup can return a cached path
without requiring a separate copy in each workspace.

To use traditional flat file downloads instead of the Hugging Face cache, pass `--force-hf-local-dir-mode` at startup. This saves files directly to `models/huggingface/<repo_id>/`.

### Authenticating for Gated Repositories

Some models (like `black-forest-labs/FLUX.1-dev` or `stabilityai/stable-diffusion-3-medium`) require accepting terms on Hugging Face before downloading. If you try to download a gated model without authentication, ComfyUI raises a `GatedRepoError` with instructions.

To authenticate:

1. Visit the model's Hugging Face page and accept the terms.
2. Set your token using one of:

```bash
export HF_TOKEN=hf_your_token_here
# or
hf auth login
```

The downloader uses Hugging Face credentials from the environment or stored login.
Access still depends on the repository's approval and license requirements.

### Disabling Automatic Downloads

To prevent automatic model downloads (e.g., in air-gapped environments):

```python
config = default_configuration()
config.disable_known_models = True
```

Or via CLI:

```bash
uv run comfyui --disable-known-models
```

## Memory management

DynamicVRAM is the normal embedded memory-management path. It tracks model
dependencies and active devices, ejecting inactive weights when memory is
needed and restoring them transparently. Configure a desktop reservation with
`Configuration(reserve_vram=...)`; use `novram=True` only as an explicit
compatibility fallback when DynamicVRAM cannot satisfy an unusual workload.

## Automated Testing

See [Testing Workflows](testing.md) for pytest integration, image output verification, and snapshot testing.

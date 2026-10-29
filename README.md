# ComfyUI

An installable, production-focused fork of [ComfyUI](https://github.com/comfyanonymous/comfyui) intended for long term support (LTS) from [AppMana](https://appmana.com) and [Hidden Switch](https://hiddenswitch.com).

> **For AI coding assistants (Claude, Codex, Cursor, etc.)**: start with [`llms.txt`](./llms.txt). It walks through how to fulfill creative requests — searching Civitai for the right LoRA, grounding edits in a reference image, picking a workflow that matches the user's VRAM — with worked examples (Studio Ghibli portrait, dancing video clip, Renaissance restyle, civitai workflow reproduction).
>
> Paste this into your agent (Claude Code, Codex CLI, Cursor, etc.) to use it as a guidebook:
>
> ```
> Use https://raw.githubusercontent.com/hiddenswitch/forks-pip-and-uv-installable-ComfyUI/refs/heads/master/llms.txt as a guidebook for ComfyUI. Fetch it now and follow it for any ComfyUI task in this session.
> ```

## Used By

Used in production by **Scopely**, a game studio, **Livepeer** and **Nunchaku Tech**. Used by innovators at **Ferrero Group**, **Hyundai** and **Nike**.

If you need to develop an application or plugin around ComfyUI, this fork stays compatible and up-to-date with upstream, fixing numerous bugs and adding features. It also packages tacit knowledge about running diffusion models and art workflows, distributed inference, deployment on Kubernetes, and other production tasks.

## Why use this fork?

**ComfyUI, ready for your workstation, scripts, and production applications.**
Keep the familiar workflow editor and ecosystem while gaining a Python package,
powerful automation tools, and integrated multi-GPU execution.

- **Install instead of assembling.** Install ComfyUI with `pip` or `uv` from our package index. Install supported custom nodes and acceleration extensions as packages too.
- **Put more GPUs to work.** Integrated tensor, pipeline, and sequence parallelism can accelerate supported models or fit models that exceed one GPU’s memory. No special workflow node is required. See [multi-GPU inference](docs/distributed.md) for supported models and hardware.
- **Spend less time reloading models.** Improved DynamicVRAM handling reuses resident models across prompts and keeps compatible LoRA-patched weights resident when they fit. Host-memory budgets account for container limits.
- **Turn workflows into commands.** Run workflow files, URLs, or templates without starting a server. Inspect each workflow’s parameters with `--help`, override prompts and seeds, and use `--all` to install missing nodes and download known models.
- **Build with ComfyUI as a library.** Execute workflows and stream previews through an async Python client, with optional process isolation. See [embedding](docs/embedded.md).
- **Scale beyond one worker.** Separate the frontend from RabbitMQ workers to process independent workflows across machines. Use the same API and workflow format.
- **Bring your existing workflows and nodes.** The compatibility layer loads ordinary ecosystem nodes; the package facade makes supported nodes installable through a Python index.
- **Create more than standard images.** Work with high-bit-depth images, HDR media, native language-model workflows, and raster/SVG conversion. These complement the model and media capabilities inherited from upstream.
- **Deploy tested builds.** Published CUDA and ROCm images reuse the exact images validated by the hardware test suites.

See [the fork’s differences from upstream](docs/differences.md) for details.

## Install

Pick the block for your platform and accelerator and paste it into a terminal.

- The **workspace directory** is where ComfyUI stores the `.venv`, downloaded models, outputs, and `custom_nodes/`. You can move it anywhere; the examples use `ComfyUI_Workspace`.
- The **virtual environment** (`.venv`) is an isolated Python install inside the workspace. Activate it before running `comfyui`.
- **`uv`** is the Python package manager used here. It creates the venv and installs ComfyUI plus the correct PyTorch wheels.
- **GPU settings** are chosen in two places: `uv pip install --torch-backend=...` selects the PyTorch wheel, and `comfyui serve --guess-settings` auto-detects runtime settings such as VRAM mode and attention backend. CUDA users should use `--torch-backend=auto`.
- **PyTorch 2.7 or newer** is required. NVIDIA 20-series and newer GPUs require a CUDA 13.0-or-newer PyTorch build.

### Windows + CUDA

```powershell
irm https://astral.sh/uv/install.ps1 | iex
$env:Path = "$HOME\.local\bin;$env:Path"
New-Item -ItemType Directory -Force "$HOME\Documents\ComfyUI_Workspace"
cd $HOME\Documents\ComfyUI_Workspace
uv venv --python 3.12
uv pip install --torch-backend=auto --extra-index-url https://nodes.appmana.com/simple/ comfyui
uv pip install --extra-index-url https://nodes.appmana.com/simple/ triton-windows
uv pip install --extra-index-url https://nodes.appmana.com/simple/ sageattention flash-attn
.\.venv\Scripts\Activate.ps1
comfyui --help
comfyui serve --guess-settings
```

For macOS, Linux CUDA, AMD ROCm, and Intel XPU setup, see [Installation & Getting Started](docs/installing.md).

### Using ComfyUI as a Library

ComfyUI can run embedded inside your own Python application. No web server is required; configurations that need isolation can use a subprocess. Use the `Comfy` async context manager to execute workflows directly:

```python
from comfy.client.embedded_comfy_client import Comfy

async with Comfy() as client:
    outputs = await client.queue_prompt(workflow_dict)
    # All models unloaded and VRAM released on exit
```

Build workflows programmatically with `GraphBuilder`, or paste JSON from the web UI (both API and UI format workflows are accepted). Stream previews during inference with `queue_with_progress`.

See [Embedded / Library Usage](docs/embedded.md) for complete examples.

## Documentation

Full documentation is available in [docs/index.md](docs/index.md).

### Core
- [Installation & Getting Started](docs/installing.md)
- [Hardware Compatibility](docs/compatibility.md)
- [Configuration](docs/configuration.md)
- [Troubleshooting](docs/troubleshooting.md)

### Features & Workflows
- [Run workflows from the CLI](docs/workflows.md)
- [Large Language Models](docs/llm.md)
- [Media Features](docs/other_features.md)

### Extending ComfyUI
- [Custom Nodes](docs/custom_nodes.md) (Installing & Authoring)
- [Embedded / Library Usage](docs/embedded.md) (Python, GraphBuilder, Streaming)
- [Testing Workflows](docs/testing.md) (pytest, Image Snapshots)
- [API Usage](docs/api.md) (REST, WebSocket)

### Deployment
- [Multi-GPU inference](docs/distributed.md)
- [Worker queues](docs/workers.md)
- [Docker & Containers](docs/docker.md)

### Development
- [Linting](docs/linting.md)

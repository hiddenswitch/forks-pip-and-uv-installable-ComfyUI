# ComfyUI

A vanilla, up-to-date fork of [ComfyUI](https://github.com/comfyanonymous/comfyui) intended for long term support (LTS) from [AppMana](https://appmana.com) and [Hidden Switch](https://hiddenswitch.com).

> **For AI coding assistants (Claude, Codex, Cursor, etc.)**: start with [`llms.txt`](./llms.txt). It walks through fulfilling creative requests: choosing compatible models and LoRAs, grounding edits in reference images, inspecting workflow parameters, running generation, and checking the resulting artifacts.
>
> Paste this into your agent (Claude Code, Codex CLI, Cursor, etc.) to use it as a guidebook:
>
> ```
> Use https://raw.githubusercontent.com/hiddenswitch/pip-and-uv-installable-ComfyUI/refs/heads/master/llms.txt as a guidebook for ComfyUI. Fetch it now and follow it for any ComfyUI task in this session.
> ```

## Used By

Used in production by **Scopely**, a game studio, **Livepeer** and **Nunchaku Tech**. Used by innovators at **Ferrero Group**, **Hyundai** and **Nike**.

Take ComfyUI from a creative workspace to an application, a command-line job, or a fleet of workers. This fork brings Python packaging, headless workflows, embedded execution, and integrated multi-GPU inference together while tracking upstream ComfyUI.

## Key Features and Differences

- **Install an application, not a checkout.** Get the `comfyui` command from a Python package index, reuse an existing workspace, and add accelerator extensions when needed.
- **Run a workflow directly.** Discover its parameters with workflow-specific help, override prompts and node inputs, and consume JSON results. Opt into missing-node installation and model downloads with `--all`.
- **Put multiple GPUs to work.** Integrated tensor, pipeline, and sequence parallelism support selected model families. Hardware-aware defaults and explicit controls let you choose between latency, model capacity, and independent-job throughput.
- **Keep useful weights resident.** Reuse a running client across prompts. Native quantized storage, LoRA materialization accounting, and bounded host-memory pinning help avoid unnecessary transfers under memory pressure.
- **Build around the workflow engine.** The embedded Python client accepts API and UI workflow dictionaries, provides progress streaming in its thread executor, and selects subprocess isolation for configuration that changes global model-management state.
- **Bring ecosystem nodes along.** Install custom nodes as packages through the facade, load conventional `custom_nodes/` folders through compatibility shims, or host your own package index. Node-specific dependencies and platform limits still apply.
- **Scale throughput independently.** RabbitMQ workers execute queued jobs behind a separate frontend; shared storage keeps inputs and outputs accessible across hosts.
- **Connect existing infrastructure.** Discover local models, resolve supported model URLs, use object-storage integrations, and deploy packaged NVIDIA or AMD containers.

You also get upstream ComfyUI capabilities such as native text generation, DynamicVRAM, and HDR/high-bit-depth media nodes. These are part of the product, not exclusive fork features. See the [workflow guide](docs/workflows.md) and [performance guide](docs/distributed.md) for practical limits.

## Install

Pick the block for your platform and accelerator and paste it into a terminal.

- The **workspace directory** is where ComfyUI stores the `.venv`, downloaded models, outputs, and `custom_nodes/`. You can move it anywhere; the examples use `ComfyUI_Workspace`.
- The **virtual environment** (`.venv`) is an isolated Python install inside the workspace. Activate it before running `comfyui`.
- **`uv`** is the Python package manager used here. It creates the venv and installs ComfyUI plus the correct PyTorch wheels.
- **GPU settings** are chosen in two places: `uv pip install --torch-backend=...` selects the PyTorch wheel, and `comfyui serve --guess-settings` auto-detects runtime settings such as VRAM mode and attention backend. CUDA users should use `--torch-backend=auto`.
- Choose a PyTorch build that supports your accelerator and driver. [Hardware compatibility](docs/compatibility.md) separates CI configurations from general package requirements.

### Windows + CUDA

```powershell
irm https://astral.sh/uv/install.ps1 | iex
$env:Path = "$HOME\.local\bin;$env:Path"
New-Item -ItemType Directory -Force "$HOME\Documents\ComfyUI_Workspace"
cd $HOME\Documents\ComfyUI_Workspace
uv venv --python 3.12
uv pip install --torch-backend=auto --extra-index-url https://nodes.appmana.com/simple/ comfyui
.\.venv\Scripts\Activate.ps1
comfyui --help
comfyui serve --guess-settings
```

For macOS, Linux + CUDA, Linux + ROCm, and Intel XPU, see [Installation & Getting Started](docs/installing.md). Accelerator support depends on the host driver and installed backend.

### Run a saved workflow

Inspect a workflow exported from the editor, prepare its dependencies, and run it
without starting a server:

```console
comfyui run-workflow workflow.json --help
comfyui run-workflow workflow.json --dry-run
comfyui run-workflow workflow.json --all --prompt "a small observatory in a snowy forest" --seed 42
```

Results are JSON on stdout. See [headless workflows](docs/workflows.md) for
parameter overrides, model discovery, and repeated generation.

### Using ComfyUI as a Library

ComfyUI can run inside your Python application without a web server. The default client uses a thread executor; configurations that change global model-management settings select an isolated subprocess. Reuse one context across jobs to retain useful caches:

```python
from comfy.client.embedded_comfy_client import Comfy

async with Comfy() as client:
    outputs = await client.queue_prompt(workflow_dict)
    # Context exit cleans up models and the executor owned by this client.
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
- [Interactive and Headless Workflows](docs/workflows.md)
- [Large Language Models](docs/llm.md)
- [Other Features](docs/other_features.md) (SVG)

### Extending ComfyUI
- [Install Custom Nodes](docs/custom_nodes.md)
- [Author Custom Nodes](docs/custom_node_authoring.md)
- [Host a Package Index](docs/package_index.md)
- [Embedded / Library Usage](docs/embedded.md) (Python, GraphBuilder, Streaming)
- [Testing Workflows](docs/testing.md) (pytest, Image Snapshots)
- [API Usage](docs/api.md) (REST, WebSocket)

### Deployment
- [Distributed / Multi-GPU](docs/distributed.md)
- [Frontend and Worker Deployment](docs/deployment.md)
- [Docker & Containers](docs/docker.md)

### Development
- [Linting](docs/linting.md)

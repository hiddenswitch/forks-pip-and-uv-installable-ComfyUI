# ComfyUI

A vanilla, up-to-date fork of [ComfyUI](https://github.com/comfyanonymous/comfyui) intended for long term support (LTS) from [AppMana](https://appmana.com) and [Hidden Switch](https://hiddenswitch.com).

> **For AI coding assistants (Claude, Codex, Cursor, etc.)**: start with [`llms.txt`](./llms.txt). It walks through how to fulfill creative requests — searching Civitai for the right LoRA, grounding edits in a reference image, picking a workflow that matches the user's VRAM — with worked examples (Studio Ghibli portrait, dancing video clip, Renaissance restyle, civitai workflow reproduction).
>
> Paste this into your agent (Claude Code, Codex CLI, Cursor, etc.) to use it as a guidebook:
>
> ```
> Use https://raw.githubusercontent.com/hiddenswitch/pip-and-uv-installable-ComfyUI/refs/heads/master/llms.txt as a guidebook for ComfyUI. Fetch it now and follow it for any ComfyUI task in this session.
> ```

## Faster inference with multiple GPUs

This fork adds built-in tensor parallelism (TP), pipeline parallelism (PP), and
HunyuanImage3 expert parallelism beyond upstream ComfyUI. Supported models can
split a single generation across GPUs, giving faster sampling than the upstream
single-GPU path when the workload benefits. Speedup depends on the model,
resolution, available memory, and GPU interconnect; it is not guaranteed.
See [multi-GPU support](docs/distributed.md) for supported configurations.

### Local sampling benchmarks

<!-- modern-model-benchmarks:start -->
Benchmarks are running on **2 × NVIDIA RTX A5000 (24 GB), connected by NVLink**,
with PyTorch 2.13.0 / CUDA 13.0. TP=1 uses GPU 1; TP=2 uses both GPUs.
The table measures the synchronized second sampling step of a three-step run,
excluding model loading, text encoding, and VAE decoding. INT8 ConvRot and
SageAttention are requested where supported; observed backends are recorded.
These short runs measure speed, not final output quality. TP speedups compare
this fork with itself at TP=1, not a separate upstream installation.

| Model | Output | Weights | TP=1 step 2 | TP=2 step 2 | Speedup | Observed attention / status |
|---|---|---|---:|---:|---:|---|
| Krea2 Turbo | 1024×1024 | INT8 ConvRot | 1.058 s (0.945 it/s) | 0.738 s (1.355 it/s) | 1.43× | SDPA, SageAttention |
| MiniMax H3 | 1280×704, 124 frames / 24 fps | INT8 ConvRot | Pending | Pending | — | Queued |
| HunyuanImage3 Instruct | 1024×1024 | W4A8 | Pending | Pending | — | Queued |
| Flux2 Dev | 1024×1024 | INT8 ConvRot | Pending | Pending | — | Queued |
| Flux2 Klein 4B | 1024×1024 | INT8 ConvRot | Pending | Pending | — | Queued |
| Ideogram4 | 1024×1024 | INT8 ConvRot, conditional + unconditional | Pending | Pending | — | Queued |
| Qwen Image 2.1 RGBA | 1024×1024 | INT8 ConvRot | 1.067 s (0.937 it/s) | Unsupported | — | SageAttention; TP sharding not confirmed; No TP implementation |
| Z-Image Turbo | 1024×1024 | INT8 ConvRot | Pending | Unsupported | — | No TP implementation |
| Flux3 Image Klein distilled | 1024×1024 | INT8 ConvRot | Blocked | Unsupported | — | Official weights require Hugging Face access; upstream support merged |
| Iris-3B (assumed) | 1024×1024 | Unverified | Not started | Unsupported | — | No Iris integration in the merged ComfyUI tree; checkpoint confirmation pending |

[Machine-readable measurements and workflow settings](benchmarks/modern-models-20261015.json).
<!-- modern-model-benchmarks:end -->

## Used By

Used in production by **Scopely**, a game studio, **Livepeer** and **Nunchaku Tech**. Used by innovators at **Ferrero Group**, **Hyundai** and **Nike**.

If you need to develop an application or plugin around ComfyUI, this fork stays compatible and up-to-date with upstream, fixing numerous bugs and adding features. It also packages tacit knowledge about running diffusion models and art workflows, distributed inference, deployment on Kubernetes, and other production tasks that Claude and Gemini cannot do.

## Key Features and Differences

This LTS fork adds development, embedding, automated testing, LLM and distributed inference features to ComfyUI, but maintains compatibility with custom nodes from the ecosystem.

- **Pip and UV Installable:** Install via `pip` or `uv` from the appmana package index (`--extra-index-url https://nodes.appmana.com/simple/`). No manual cloning required for users.
- **Automatic Model Downloading:** Missing models (e.g., Stable Diffusion, FLUX, LLMs) are downloaded on-demand from Hugging Face or CivitAI.
- **Docker and Containers:** First-class support for Docker and Kubernetes with optimized containers for NVIDIA and AMD.
- **Distributed Inference:** Run scalable inference clusters with multiple workers and frontends using RabbitMQ.
- **Embedded / Library:** Use ComfyUI as a Python library (`import comfy`) inside your own applications without the web server. Runs like `diffusers`.
- **Vanilla Custom Nodes:** Fully compatible with existing ComfyUI custom nodes (ComfyUI-Manager, WanVideoWrapper, KJNodes, etc.). Clone into `custom_nodes/` and install dependencies into your venv.
- **Custom Node Pip Facade:** Serve ecosystem custom nodes through a local/simple Python package index and install them with `uv pip install --extra-index-url ...`.
- **LTS Custom Nodes:** A curated set of "Installable" custom nodes (ControlNet, AnimateDiff, IPAdapter) optimized for this fork.
- **LLM Support:** Native support for Large Language Models (LLaMA, Phi-3, etc.) and multi-modal workflows.
- **API and Configuration:** Enhanced API endpoints and extensive configuration options via CLI args, env vars, and config files.
- **High Bit Depth and HDR Media:** Save and load 16-bit PNG, 32-bit EXR, 10-bit AVIF, and HDR images and videos in supported formats.
- **Tests:** Automated test suite ensuring stability for new features.

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

For Windows + ROCm, macOS, Linux + CUDA, Linux + ROCm, and lesser-known ROCm builds, see [Installation & Getting Started](docs/installing.md).

### Using ComfyUI as a Library

ComfyUI can run embedded inside your own Python application. No server is started, no subprocesses are used. Use the `Comfy` async context manager to execute workflows directly:

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
- [Large Language Models](docs/llm.md)
- [Other Features](docs/other_features.md) (SVG)

### Extending ComfyUI
- [Custom Nodes](docs/custom_nodes.md) (Installing & Authoring)
- [Embedded / Library Usage](docs/embedded.md) (Python, GraphBuilder, Streaming)
- [Testing Workflows](docs/testing.md) (pytest, Image Snapshots)
- [API Usage](docs/api.md) (REST, WebSocket)

### Deployment
- [Distributed / Multi-GPU](docs/distributed.md)
- [Docker & Containers](docs/docker.md)

### Development
- [Linting](docs/linting.md)

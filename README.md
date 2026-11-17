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

You also get upstream ComfyUI capabilities such as native text generation, DynamicVRAM, and HDR/high-bit-depth media nodes. These are part of the product, not exclusive fork features. See the [supported models and benchmarks guide](docs/models.md), [workflow guide](docs/workflows.md), and [performance guide](docs/distributed.md) for practical limits.

## Supported Models (`This Fork` vs. `Upstream`)

The table below highlights models and execution modes supported in **This Fork** compared with **Upstream**, along with peak VRAM / unified-memory requirements (`Resident` / `DynamicVRAM offloaded floor`) and measured `2× GB10` (`TP=2`) sampling speed (`s/it`) for models whose full unoffloaded pipeline exceeds 128 GiB. For the full 113-model catalog and trace-separated `INT8 ConvRot` benchmarks across `8G` (RX 7600), `12G` (RTX 3060), `24G` (RTX 3090), `2×24G` (2× RTX A5000), `1× GB10`, `2× GB10`, and `M3 Ultra`, see [Supported Models and Hardware Benchmarks](docs/models.md).

| Model / Family | Generation Args | This Fork | Upstream |
| --- | --- | --- | --- |
| **MAGI-2 Preview (35.9B MoE)**<br>`magi2_preview_int8_convrot` + `qwen3.5_27b_int8_convrot` + `magi2_turbo_vae` + `stable-audio-open-1.0` | `896×512`, 25 frames (`2.0 s` @ 12.5 fps) + stereo audio (`11,472` tokens) | **Yes** (`TP=2/4`, `PP≥2`, fused Triton `flash_mh_moe`)<br>• Full `BF16` pipeline: **~136 GiB** (`INT8 ConvRot` `TP=2`: **33.3 GiB** R0 / **48.1 GiB** R1)<br>• **`2× GB10` (`TP=2`)**: **`36.5 s/it` avg (`28.0–31.0 s/it` steady-state)** | **No** |
| **MAGI-2 Preview + Refiner (35.9B MoE + 6.9B)**<br>`magi2_preview_int8_convrot` + `magi2_refiner_int8_convrot` | `896×512` $\rightarrow$ `1920×1088`, 25 frames (`2.0 s` @ 12.5 fps) + stereo audio | **Yes** (`TP=2/4/8`, `PP≥2`, fused Triton `flash_mh_moe`)<br>• Full `BF16` pipeline: **~154 GiB** (`INT8 ConvRot` `TP=2`: **~42 GiB / rank**)<br>• **`2× GB10` (`TP=2`)**: **`36.5 s/it` preview + `~9.4 s/it` refiner** | **No** |
| **HunyuanImage 3.0 Instruct (80B MoE)**<br>`hunyuan_image_3_instruct_w4a8` | `1024×1024`, 20 steps | **Yes** (`TP` + Expert Parallelism, `PP≥2`, `W4A8` streaming)<br>• Full `BF16` pipeline: **~164 GiB** (`2× GB10`: **`~6.8 s/it`**; `W4A8` `TP=2`: **~22 GiB/GPU**) | **No** |
| **Kandinsky 6 Pro**<br>`Kandinsky-6.0-Pro-5s` | `832×480`, 121 frames (`5.0 s`) | **Yes** (Native `TP=2/4` + `MagCache` transport, **~66 GiB** total / **~34 GiB/GPU** at `TP=2`) | **No** |
| **PAN 2 (27B AR Chunked World Model)**<br>`pan2_int8_convrot` + `qwen3.5_9b_int8_convrot` | `832×480`, 21 frames (5 latent frames), CFG 5.0 | **Yes** (Single-GPU + **`TP=2/4`** + **`PP≥2`**)<br>• Peak VRAM: **~36.5 GiB** `INT8 ConvRot` (**~19.2 GiB/GPU** at `TP=2`; **12 GiB** offloaded) | **Single-GPU only** |
| **MiniMax H3 Video+Audio (`FL2VA` / `Ref2VA`)**<br>`minimax_h3_fl2va_pruned_int8_convrot` + `qwen3vl_32b_minimax_h3_int8_convrot` | `1280×704`, 124 frames (`24 fps`) + stereo audio | **Yes** (Single-GPU + **`TP=2/4`** + **`PP≥2`** + **`Ulysses/Ring`**)<br>• Full `BF16` pipeline: **~132 GiB** (`2× GB10` `TP=2`: **`~19.5 s/it`**; `INT8 ConvRot` `TP=2`: **~22.8 GiB/GPU**; **12 GiB** offloaded) | **Single-GPU only** |
| **MiniMax H3 Image**<br>`minimax_h3_image_pruned_int8_convrot` | `1024×1024`, 6 steps | **Yes** (Single-GPU + **`TP=2/4`** + **`PP≥2`**, **~53 GiB** resident / **~21.5 GiB/GPU** at `TP=2` / **12 GiB** offloaded) | **Single-GPU only** |
| **Flux.2 Dev (32B)**<br>`flux2-dev-int8-convrot-simple` | `1024×1024`, 20 steps (`Flux2Scheduler`) | **Yes** (Single-GPU + **`TP=2/4`** + **`PP≥2`** + **`Ulysses/Ring`**, **~46 GiB** `INT8 ConvRot` / **~21.4 GiB/GPU** at `TP=2` / **12 GiB** offloaded) | **Single-GPU only** |
| **Ideogram 4**<br>`ideogram4_int8_convrot` (cond + uncond) | `1024×1024`, 20 steps (`Ideogram4Scheduler`) | **Yes** (Single-GPU + **`TP=2/4`**, **~28.5 GiB** resident / **16.4 GiB/GPU** at `TP=2`, **`3.16 s/it`** on `2× A5000`; **8–12 GiB** offloaded) | **Single-GPU only** |
| **Krea 2 Turbo**<br>`krea2_turbo_int8_convrot` | `1024×1024`, 4–8 steps | **Yes** (Single-GPU + **`TP=2/4`**, **~18.2 GiB** resident / **9.2 GiB/GPU** at `TP=2`, **`0.738 s/it`** on `2× A5000`; **8 GiB** offloaded) | **Single-GPU only** |
| **Qwen-Image / Layered / 2.1**<br>`qwen_image_2.1_int8_convrot` / `qwen-image-Q4_K_M.gguf` | `1024×1024` / `1328×1328` | **Yes** (Single-GPU + **`PP≥2`** + **`Ulysses/Ring`** + **Native `GGUF`**, **~25.4 GiB** `INT8 ConvRot` / **8–12 GiB** offloaded) | **Single-GPU safetensors only** |
| **Native `SVDQuant W4A4` & `AWQ W4A16`** | `1024×1024` (`Flux.1`, `Qwen-Image`) | **Yes** (`comfy_kitchen` Nunchaku 4-bit + low-rank SVD, **6.5–11.8 GiB** VRAM) | **No** |
| **Native `TransformersLoader` LLMs/VLMs & SVG (`StarVector`/`VTracer`, `TextDiffuser`)** | Text / VLM / Raster $\leftrightarrow$ SVG | **Yes** (`Phi-4`, `Qwen2-VL`, `LLaVA`, `Florence-2`, `PaliGemma`, `Pixtral-12B`, **1.5–28 GiB** VRAM) | **No** |
| **All Upstream Diffusion, Video, Audio, 3D & Vision Models** (`Flux.1`, `Flux.3 Klein`, `Fibo 2.0`, `MOSS-Image`, `Mage-Flow`, `Ming-Image`, `Z-Image`, `JoyImage`, `Boogu`, `Ernie-Image`, `Chroma`, `HiDream`, `SenseNova U1.5`, `HunyuanImage 2.1`, `LTX-2/2.5`, `Wan 2.1/2.2`, `HunyuanVideo 1.5`, `Cosmos 1/2`, `Mochi`, `SeedVR2`, `YuE2`, `MiniMax Music 3`, `ACE-Step`, `Stable Audio`, `TRELLIS.2`, `Hunyuan3D`, `TripoSplat`, `SAM 3.1`, `DepthAnything 3`, `MoGe 3`, `Marigold V2`) | Image (`1024×1024`), Video (`480p`–`720p`), Audio (`30–47 s`), 3D | **Yes** (with native `INT8 ConvRot`, `--guess-settings`, and `DynamicVRAM`; **2.2–36 GiB** resident / **4–12 GiB** offloaded — see [full table](docs/models.md)) | **Yes** |

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
- [Supported Models & Hardware Benchmarks](docs/models.md)
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

# Custom Nodes

Custom Nodes can be added to ComfyUI by copying and pasting Python files into your `./custom_nodes` directory.

## Installing ComfyUI

This fork is installed from the same index as the `comfyui` package:

```bash
uv pip install --extra-index-url https://nodes.appmana.com/simple/ comfyui
```

The wheel is built and attached to each GitHub release by
`.github/workflows/build-wheel.yml`, and the index serves those release assets.

## Installing Custom Nodes with uv

Install custom nodes from the package index at `nodes.appmana.com`:

```bash
uv pip install --extra-index-url https://nodes.appmana.com/simple/ kandinsky6 kandinsky6-sr comfyui-hunyuanimage3
uv pip install --extra-index-url https://nodes.appmana.com/simple/ comfyui-wanvideowrapper
uv pip install --extra-index-url https://nodes.appmana.com/simple/ comfyui-kjnodes
uv pip install --extra-index-url https://nodes.appmana.com/simple/ comfyui-controlnet-aux
uv pip install --extra-index-url https://nodes.appmana.com/simple/ comfyui-videohelpersuite
```

All supported installation paths use uv so the active Torch backend remains
under one resolver and lockfile policy.

Kandinsky 6 and HunyuanImage3 remain pip-installed custom nodes. Their upstream
prompting references are [Kandinsky 6](prompting_guides/kandinsky6_prompting.md)
and [HunyuanImage3 Instruct](prompting_guides/hunyuan_image3_instruct_prompting.md).

### AMD TheRock packages

The facade exposes AMD’s allowlisted TheRock simple index without rehosting
large wheels. Use it when installing the split host/device packages directly:

```bash
uv pip install --index-url https://nodes.appmana.com/simple/rocm/ \
  torch==2.14.0+rocm10.1.0 torchvision==0.29.0a0+rocm10.1.0
```

The current stable uv resolver does not select TheRock’s `device-all` extras
reliably, so use the validated AppMana ROCm image or explicitly pin the device
packages listed by that release. The facade preserves future WheelNext index
metadata, but does not pretend to implement resolver-side variant selection.

### Selecting a CUDA version

The default index (`/simple/`) serves CUDA 13.0 binaries for packages like `nunchaku` and `sageattention`. To install CUDA 12.8 builds instead, use the `/simple/cu128` index:

```bash
# CUDA 13.0 (default)
uv pip install --extra-index-url https://nodes.appmana.com/simple/ sageattention
uv pip install --extra-index-url https://nodes.appmana.com/simple/ comfyui-nunchaku

# CUDA 13.0 (explicit)
uv pip install --extra-index-url https://nodes.appmana.com/simple/cu130 sageattention
uv pip install --extra-index-url https://nodes.appmana.com/simple/cu130 comfyui-nunchaku

# CUDA 12.8
uv pip install --extra-index-url https://nodes.appmana.com/simple/cu128 sageattention
uv pip install --extra-index-url https://nodes.appmana.com/simple/cu128 comfyui-nunchaku
```

All other packages (custom nodes, patched dependencies) are identical across CUDA variants.

#### Triton

`triton` and `triton-windows` are served directly from the index:

```bash
# Linux gets PyPI manylinux triton; Windows gets woct0rdho triton-windows
uv pip install --extra-index-url https://nodes.appmana.com/simple/cu130 triton
# Same Windows wheels under their own name
uv pip install --extra-index-url https://nodes.appmana.com/simple/cu130 triton-windows
```

On Windows, `triton` resolves to the `triton-windows` wheels (rewritten to the
`triton` distribution name). For CUDA 13 variants (`cu130`/`cu131`/`cu132`), the
index swaps the wheel's bundled CUDA 12.x compiler binaries
(`ptxas.exe`, `cuda.h`, `cuda.lib`) for the matching CUDA 13.x redistributables —
woct0rdho's wheels ship 12.x regardless of target, which breaks `ptxas` on the
CUDA 13 toolchain.

For Intel XPU installs, `torch` itself can depend on `triton-xpu`. That package
is hosted by PyTorch's XPU wheel index
(`https://download.pytorch.org/whl/xpu/triton-xpu/`), not PyPI. Use
`uv pip install --torch-backend=xpu ...` for XPU environments so uv includes the
PyTorch XPU index when resolving custom-node requirements.

#### Selecting a CUDA *and* torch ABI (flash-attn)

`flash-attn` / `flash-attn-3` wheels are tagged with both the CUDA and torch
version. Use a combined `cuXXXtorchY.Z` index to filter to exactly the matching
wheel:

```bash
uv pip install --extra-index-url https://nodes.appmana.com/simple/cu130torch2.12 flash-attn
uv pip install --extra-index-url https://nodes.appmana.com/simple/cu128torch2.11 flash-attn-3
```

To find and install the packages a workflow needs:

```bash
uv pip install --extra-index-url https://nodes.appmana.com/simple/ \
  -r <(comfyui workflows requirements path/to/workflow.json)
```

## Other Installation Methods

There are two kinds of custom nodes: vanilla custom nodes, which generally expect to be dropped into the `custom_nodes` directory and managed by a tool called the ComfyUI Extension manager ("vanilla" custom nodes) and this repository's opinionated, installable custom nodes ("installable").

### Installing ComfyUI Manager

ComfyUI-Manager is a popular extension to help you install and manage other custom nodes. To install it, you will need `git` on your system.

#### Manual Install

The installation process for ComfyUI-Manager requires installing its Python dependencies, cloning its code into the `custom_nodes` directory, and enabling it at startup.

1.  **Install dependencies.**
    First, ensure you have installed `comfyui` from this repository as described in the Installing section. Then, run the following command from your ComfyUI workspace directory (the one containing your `.venv` folder) to install the extra dependencies for ComfyUI-Manager:

    ```shell
    uv pip install --torch-backend=auto --upgrade --extra-index-url https://nodes.appmana.com/simple/ "comfyui[comfyui_manager]"
    ```

2.  **Clone the repository.**
    Next, you need to clone the ComfyUI-Manager repository into the `custom_nodes` directory within your ComfyUI workspace. Your workspace is the directory you created during the initial setup where you ran `uv venv` (e.g., `~/Documents/ComfyUI_Workspace`).

    If the `custom_nodes` directory does not exist in your workspace, create it first (e.g., `mkdir custom_nodes`). Then, from your workspace directory, run the following command:

    ```shell
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git ./custom_nodes/ComfyUI-Manager
    ```
    This command will place the manager's code into `custom_nodes/ComfyUI-Manager/`.

3.  **Restart ComfyUI.**
    After cloning, start `comfyui serve --enable-manager`. The manager UI is available when enabled.

### ComfyUI-Manager Command Line Options

| Flag | Description |
|------|-------------|
| `--enable-manager` | Enable ComfyUI-Manager |
| `--enable-manager-legacy-ui` | Use the legacy manager UI instead of the new UI (requires `--enable-manager`) |
| `--disable-manager-ui` | Disable the manager UI and endpoints while keeping background features like security checks and scheduled installation completion (requires `--enable-manager`) |

### Vanilla Custom Nodes

This fork imports ordinary ecosystem nodes through its compatibility layer. Nodes still need compatible Python dependencies, accelerator extensions, and frontend APIs. Private upstream APIs can require node-specific fixes.

#### Step 1: Open a Terminal in Your Workspace

Your workspace is the directory where you ran `uv venv` during installation (the one containing your `.venv` folder).

**Windows (PowerShell):**
```powershell
cd ~\Documents\ComfyUI_Workspace
.\.venv\Scripts\Activate.ps1
```

**macOS:**
```shell
cd ~/Documents/ComfyUI_Workspace
source .venv/bin/activate
```

**Linux:**
```shell
cd ~/Documents/ComfyUI_Workspace
source .venv/bin/activate
```

#### Step 2: Create the `custom_nodes` Directory (if it doesn't exist)

```shell
mkdir -p custom_nodes
```

On Windows PowerShell, use:
```powershell
if (!(Test-Path custom_nodes)) { mkdir custom_nodes }
```

#### Step 3: Clone the Custom Node and Install Dependencies

Clone the repository into `custom_nodes/` and install its Python dependencies:

```shell
git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git custom_nodes/ComfyUI-KJNodes
uv pip install -r custom_nodes/ComfyUI-KJNodes/requirements.txt
```

Some nodes may not have a `requirements.txt`. In that case, skip the `uv pip install` step.

#### Step 4: Restart ComfyUI

After cloning and installing dependencies, restart ComfyUI. The new nodes will be available in the node menu.

#### More Examples

```shell
# WAN Video Wrapper
git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git custom_nodes/ComfyUI-WanVideoWrapper
uv pip install -r custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt

# ComfyUI-Manager (also available as an installable, see above)
git clone --depth 1 https://github.com/Comfy-Org/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager
uv pip install -r custom_nodes/ComfyUI-Manager/requirements.txt
```

### Custom Nodes Authored for this Fork

Run `uv pip install "git+https://github.com/owner/repository"`, replacing the `git` repository with the installable custom nodes URL. This is just the GitHub URL.

## More guides

- [Author custom nodes](node_authoring.md)
- [Manage nodes programmatically](node_management.md)
- [Host your own package index](package_index.md)

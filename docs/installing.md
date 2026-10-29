# Getting started

ComfyUI uses `uv` for environments and dependency resolution. The examples
below never modify `PYTHONPATH` and install the application through the
package facade when binary extensions are needed.

## Linux and macOS

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/ComfyUI_Workspace"
cd "$HOME/ComfyUI_Workspace"
uv venv --python 3.12
uv pip install --torch-backend=auto --extra-index-url https://nodes.appmana.com/simple/ comfyui
source .venv/bin/activate
comfyui serve --guess-settings
```

### Optional accelerator extensions

The base installation does not require SageAttention or FlashAttention. Install
extensions only when your workflow needs them and matching wheels exist for your
Python, Torch, CUDA, and operating system. For a compatible CUDA 13.0 environment:

```console
uv pip install --extra-index-url https://nodes.appmana.com/simple/cu130/ sageattention flash-attn
```

Apple Silicon uses the MPS backend selected by `--torch-backend=auto`. Feature
availability follows the installed PyTorch MPS build.

## AMD ROCm

For a supported Linux AMD host, verify `/dev/kfd` and a render node exist, then
install from AMD’s stable TheRock index. The current AppMana image uses the
same host/device split and all-architecture closure. This example selects
RX 7600 (`gfx1102`); select the matching device packages for other GPUs:

```console
uv venv --python 3.12
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ \
  torch==2.14.0+rocm10.1.0 \
  torchvision==0.29.0a0+rocm10.1.0 \
  torchaudio==2.11.0.3+rocm10.1.0 \
  amd-torch-device-gfx1102==2.14.0+rocm10.1.0 \
  amd-torchvision-device-gfx1102==0.29.0a0+rocm10.1.0
uv pip install --extra-index-url https://nodes.appmana.com/simple/ comfyui
source .venv/bin/activate
comfyui serve --guess-settings --fp32-vae
```

The generic image includes TheRock device packages for every architecture in
that release and is validated on RX 7600/gfx1102. Do not set
`HSA_OVERRIDE_GFX_VERSION` when the installed wheel natively contains the
GPU’s architecture.

## Intel XPU

Install the XPU backend with uv and verify it before starting ComfyUI:

```console
uv venv --python 3.14.7
uv pip install --torch-backend=xpu --extra-index-url https://nodes.appmana.com/simple/ \
  comfyui "torch==2.14.1+xpu" "torchvision==0.29.1+xpu" "torchaudio==2.11.0+xpu"
source .venv/bin/activate
python -c "import torch; assert torch.xpu.is_available(); print(torch.xpu.get_device_name(0))"
comfyui serve --guess-settings
```

On Arc A380, set `TRITON_INTEL_DEVICE_ARCH=dg2` before starting ComfyUI when
using `torch.compile`. The Triton version in this stack recognizes DG2 G10 but
does not automatically select the architecture for the A380's G11 device.

## Windows

```powershell
irm https://astral.sh/uv/install.ps1 | iex
New-Item -ItemType Directory -Force "$HOME\Documents\ComfyUI_Workspace"
cd "$HOME\Documents\ComfyUI_Workspace"
uv venv --python 3.12
uv pip install --torch-backend=auto --extra-index-url https://nodes.appmana.com/simple/ comfyui
.\.venv\Scripts\Activate.ps1
comfyui serve --guess-settings
```

## Models and workflows

Known model files can be downloaded on demand. Authenticate with Hugging Face
using its supported CLI, or set `HF_TOKEN` in the process environment. Disable
automatic downloads with `--disable-known-models` when running offline.

Workflow templates are listed with:

```console
comfyui workflows list
comfyui run-workflow <template-or-path> --help
```

## Upgrading

```console
uv pip install --upgrade --extra-index-url https://nodes.appmana.com/simple/ comfyui
```

Keep accelerator-specific Torch packages pinned when upgrading the application
so uv cannot replace a tested CUDA, XPU, or TheRock stack.

## Reuse an existing workspace

Keep an existing compatible environment rather than recreating it to match an
example's Python version. Package metadata permits Python >=3.10, but dependency
and accelerator wheels can narrow that range. See [compatibility](compatibility.md)
for the configured test stacks.

Point the installed application at your existing models, inputs, outputs, and
custom-node directory:

```console
comfyui serve --cwd /path/to/ComfyUI
```

Alternatively, use `--extra-model-paths-config` or `--add-model-folder-path` to
share model directories. Upstream-style source workspaces may receive missing
package marker files from the [workspace compatibility layer](custom_node_reference.md#existing-workspace-imports).
Do not modify `PYTHONPATH` to select between competing installations.

## Memory and performance

DynamicVRAM is the normal memory-management path. It accounts for all selected
devices and can eject inactive model dependencies when memory is needed. Use
`--reserve-vram` for a desktop reservation. `--novram` remains an explicit
compatibility escape hatch and is not required for ordinary constrained runs.

On NVIDIA Ampere and newer, `--fast cublas_ops` and SageAttention may improve
sampling.

See [workflows](workflows.md), [configuration](configuration.md), [distributed inference](distributed.md),
and [troubleshooting](troubleshooting.md) for configuration and usage details.

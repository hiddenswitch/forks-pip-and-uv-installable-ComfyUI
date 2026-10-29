# Hardware and software compatibility

The package advertises Python 3.10–3.14. The project’s dependency metadata is the
source of truth for installation. It declares Python >=3.10; dependencies and
binary wheels can impose narrower constraints. The versions below describe CI
targets, not universal installation requirements or a guarantee of passing runs.

## CI backend targets

| Backend | Configured CI stack | Hardware target | Notes |
|---|---|---|---|
| NVIDIA CUDA | Windows Python 3.14.7, CUDA 13.0, PyTorch 2.14.1 | RTX 3090 | Enabled backend test job; extension and model-parallel support vary by model |
| AMD ROCm | TheRock ROCm 10.1.0, PyTorch 2.14.0 | RX 7600 (gfx1102) | Generic `device-all` image; no architecture override required |
| Intel XPU | Python 3.14.7, PyTorch 2.14.1 XPU | Arc A380 | Uses the XPU allocator path, not CUDA allocator settings |
| Apple MPS | macOS 14, Python 3.12, PyTorch 2.13 | macOS Apple Silicon | Conditional job: runs for Comfy-Org/ComfyUI or when `ENABLE_MACOS_CI=true`; not enabled unconditionally for this fork |
| CPU | Arch Python 3.14, PyTorch 2.14 CPU | Any supported platform | Useful for tests and fallback execution, not performance inference |

The AMD image uses AMD’s separated host/device packages from the stable
[TheRock wheel index](https://stable.repo.amd.com/rocm/whl-next/). The image
includes all architectures published by the selected release; the RX 7600 is
the hardware acceptance target. See [distributed inference](distributed.md)
for model-parallel support rather than duplicating that matrix here.

## Compatibility expectations

- Native ComfyUI nodes and the packaged frontend are tested together. A custom
  node that depends on a private frontend build may need its own compatibility
  pin.
- DynamicVRAM is the normal memory-management path. Use explicit low-memory
  modes only when diagnosing a constrained or unusual environment.
- Torch extensions must match the active backend and Python ABI. The package
  facade exposes named, allowlisted indexes for supported extension families;
  it does not make arbitrary binary wheels compatible.
- On AMD, verify `/dev/kfd`, a render node, and the host driver before blaming
  Python packages. The image must report gfx1102 directly and must not require
  `HSA_OVERRIDE_GFX_VERSION`.

## Reporting a compatibility issue

Include the output of `comfyui env`, the backend/device name, Python and Torch
versions, the model checkpoint family, and the smallest workflow that fails.
For distributed failures also include the selected mode, device list, and
canonical launcher rank values.

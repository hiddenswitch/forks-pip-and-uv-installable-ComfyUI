# Troubleshooting

Include `comfyui env`, the failing workflow, and the exception when reporting
an issue. [Compatibility](compatibility.md) lists the current CI targets.

An MPS operation missing support for a dtype depends on the PyTorch build and
operation. Changing Python alone does not add that kernel. Use the tested
macOS/PyTorch combination and report a minimal workflow if it still fails.

`HeaderTooLarge` while reading safetensors can indicate a truncated download
or a non-safetensors response saved under that extension. Check the file size
and checksum against its publisher before replacing the affected file.

| Symptom | Next check |
|---|---|
| Missing workflow node | Identify its registry package and install it in the same environment; see [custom nodes](custom_nodes.md). |
| Model filename cannot resolve | Register its folder or download source; use [workflow preparation](workflows.md#prepare-missing-dependencies) to inspect missing assets. |
| Gated model download fails | Accept the repository terms and authenticate with `hf auth login` or `HF_TOKEN`. |
| Accelerator extension fails to import | Compare its wheel's Python/Torch/backend requirements with `comfyui env`; optional extensions are not required for the base install. |
| Out of memory | Check selected devices, model size, image/video dimensions, batch size, and host limits; see [memory handling](distributed.md). |
| Embedded progress is absent | Process executors do not currently forward progress callbacks; see [isolation](embedded.md#client-lifetime-and-isolation). |
| Queue worker cannot read an input | Verify shared input/output paths and matching node environments on frontend and workers; see [deployment](deployment.md). |

Report reproducible failures through the [issue tracker](https://github.com/hiddenswitch/pip-and-uv-installable-ComfyUI/issues).
Include the complete error and the smallest workflow that exhibits it.

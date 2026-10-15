# Troubleshooting

Include `comfyui env`, the failing workflow, and the exception when reporting
an issue. [Compatibility](compatibility.md) lists the current CI targets.

An MPS operation missing support for a dtype depends on the PyTorch build and
operation. Changing Python alone does not add that kernel. Use the tested
macOS/PyTorch combination and report a minimal workflow if it still fails.

`HeaderTooLarge` while reading safetensors can indicate a truncated download
or a non-safetensors response saved under that extension. Check the file size
and checksum against its publisher before replacing the affected file.

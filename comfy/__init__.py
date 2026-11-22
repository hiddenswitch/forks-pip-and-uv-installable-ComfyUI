__version__ = "0.46.0.0"

# This deals with workspace issues
from comfy_compatibility.workspace import auto_patch_workspace_and_restart

auto_patch_workspace_and_restart()

# Linting Guidelines

This document describes the custom linting rules used in ComfyUI and how to resolve common linting issues.

## Linter configuration

This project uses **two linters that run sequentially**:

1. **ruff** — handles every standard Python lint rule (F-codes, E-codes, W-codes, B-codes from flake8-bugbear, T-codes from flake8-print, S307/S102 from flake8-bandit). This is the bulk of what used to be pylint's coverage. Configuration lives under `[tool.ruff]` in `pyproject.toml`.
2. **pylint** — runs *only* custom checkers in `tests/*_checker.py`: import ordering, package-init coverage, root node placement, SD clip config forwarding, and merge hygiene rules for version sync, Transformers imports, model inference coverage, packaged blueprints, CUDA allocator defaults, and workflow conversion caches. The default pylint profile is disabled via `disable = ["all"]` so pylint does no built-in inference — it's a pure plugin host. This is much faster than full pylint.

## Pragma Comments — `# noqa` vs `# pylint: disable=`

- For **ruff** rules (F401, F811, B005, etc.) use `# noqa: <CODE>[,<CODE2>]` at end of line.
- For **pylint** custom rules use `# pylint: disable=<symbol>`.
- **Do not combine them** in a single comment like `# pylint: disable=import-error, noqa: F401`. Pylint's option parser treats everything after `disable=` as pylint message symbols and emits W0012 unknown-option-value for `noqa` and `F401`. Keep them as two separate comments if both are needed:

  ```python
  import optional_pkg  # noqa: F401  # pylint: disable=absolute-import-used
  ```

## Custom Linting Rules

### W9001: SDClipModel Missing Config

**Rule:** `tests/sd_clip_model_init_checker.py`

Classes inheriting from `SDClipModel` must have `textmodel_json_config` as an explicit argument in their `__init__` method.

**Bad:**
```python
class MyClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None):
        super().__init__(device=device, textmodel_json_config={}, dtype=dtype)
```

**Good:**
```python
class MyClipModel(sd1_clip.SDClipModel):
    def __init__(self, device="cpu", textmodel_json_config=None, dtype=None):
        if textmodel_json_config is None:
            textmodel_json_config = {}
        super().__init__(device=device, textmodel_json_config=textmodel_json_config, dtype=dtype)
```

### W0001: Absolute Import Used

**Rule:** `tests/absolute_import_checker.py`

Within the `comfy` or `comfy_extras` packages, use relative imports instead of absolute imports for modules within the same package. This applies to both `from X import Y` and `import X` style imports.

**Bad:**
```python
# In comfy/ldm/lightricks/av_model.py
from comfy.ldm.lightricks.model import CrossAttention
from comfy.ldm.common_dit import rms_norm
import comfy.ldm.common_dit
```

**Good:**
```python
# In comfy/ldm/lightricks/av_model.py
from .model import CrossAttention
from ..common_dit import rms_norm
```

**Relative Import Reference:**
- `.module` - same directory
- `..module` - parent directory
- `...module` - grandparent directory

## Common Issues and Fixes

### Optional Dependencies (E0401: import-error)

For optional dependencies like `torchaudio` that may not be installed, use local imports with a pylint disable comment:

**Bad:**
```python
import torchaudio  # Top-level import causes E0401

def process_audio(waveform):
    return torchaudio.functional.resample(waveform, 44100, 16000)
```

**Good:**
```python
def process_audio(waveform):
    import torchaudio  # pylint: disable=import-error
    return torchaudio.functional.resample(waveform, 44100, 16000)
```

If the code path should fail gracefully when the dependency is missing, follow the local import with a targeted `ImportError` handler and raise the repo’s domain-specific exception or skip path instead of crashing at import time.

**Good:**
```python
def load_audio_backend():
    try:
        import torchaudio  # pylint: disable=import-error
    except ImportError as exc:
        raise TorchAudioNotFoundError("torchaudio is required for this node") from exc
    return torchaudio
```

### W8002: Root-Level `comfy_extras/nodes_*.py` File

**Rule:** `tests/root_comfy_extras_nodes_checker.py`

Files matching `comfy_extras/nodes_*.py` must not live at the `comfy_extras/` package root. Move them into `comfy_extras/nodes/` with `git mv`.

**Bad:**
```text
comfy_extras/nodes_math.py
comfy_extras/nodes_sdpose.py
```

**Good:**
```text
comfy_extras/nodes/nodes_math.py
comfy_extras/nodes/nodes_sdpose.py
```

### Dynamic Attribute Access (E1101: no-member)

When accessing attributes that are dynamically defined (e.g., in subclasses), use a pylint disable comment:

```python
class StringConvertibleEnum(Enum):
    @classmethod
    def str_to_enum(cls, value):
        if value is None:
            if hasattr(cls, "NONE"):
                return cls.NONE  # pylint: disable=no-member
```

### Variables Defined in Control Flow

Ensure variables are defined before use, even if the linter can't follow the control flow:

**Bad:**
```python
if condition:
    x, y, z = compute_values()
else:
    # x, y, z not defined here but used later
    pass

# Linter warns x, y, z may be undefined
result = process(x, y, z)
```

**Good:**
```python
x, y, z = None, None, None  # Initialize to satisfy linter
if condition:
    x, y, z = compute_values()
else:
    x, y, z = default_values()

result = process(x, y, z)
```

### Break/Continue Outside Loop (E0103)

Ensure `break` and `continue` statements are properly indented inside their loops:

**Bad:**
```python
with open(file) as f:
    for line in f:
        process(line)

    if some_condition:
        break  # Error: break outside loop
```

**Good:**
```python
with open(file) as f:
    for line in f:
        process(line)

        if some_condition:
            break  # Correctly inside the for loop
```

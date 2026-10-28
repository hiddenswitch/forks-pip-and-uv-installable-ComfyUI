# Testing Workflows

This page covers automated testing of ComfyUI workflows using pytest. For general library usage, see [Embedded / Library Usage](embedded.md).

## CI validation and publication

Develop runs the CPU, Windows, XPU, CUDA, and ROCm suites. CUDA/ROCm images
receive a full-commit-SHA tag only after their validation succeeds. Master
and release-tag pushes reuse those validated images without rebuilding or
rerunning accelerator tests. Publication fails if the exact commit has no
validated image; merge develop into master by fast-forward to preserve that
commit identity. Master additionally runs the installed-package smoke check.

## ProcessPoolExecutor for Isolation

Use `ProcessPoolExecutor` to run each workflow in a subprocess. This ensures VRAM is fully released between runs and prevents state leakage between tests:

```python
from comfy.client.embedded_comfy_client import Comfy
from comfy.cli_args import default_configuration
from comfy.distributed.process_pool_executor import ProcessPoolExecutor

# auto-detect GPU type, VRAM mode, and attention backend
config = default_configuration()
config.guess_settings = True
config.disable_all_custom_nodes = True

with ProcessPoolExecutor(max_workers=1) as executor:
    async with Comfy(configuration=config, executor=executor) as client:
        outputs = await client.queue_prompt(prompt)
```

## Directory Structure

ComfyUI's test suite uses `importlib.resources` to discover workflow JSON files from a Python package:

```
tests/
└── inference/
    ├── __init__.py
    ├── test_workflows.py
    └── workflows/
        ├── __init__.py          # makes this a Python package
        ├── sd15-basic-0.json
        ├── flux-0.json
        └── my-custom-workflow-0.json
```

The `__init__.py` inside `workflows/` is required so that `importlib.resources` can discover the JSON files. Each JSON file can be either an API-format workflow (Save -> API Format) or a standard UI workflow (Save) exported from the ComfyUI web UI. UI workflows are automatically converted to API format at runtime.

## Minimal pytest Example

```python
import importlib.resources
import json
import pytest
from comfy.client.embedded_comfy_client import Comfy
from comfy.api.components.schema.prompt import Prompt
from comfy.cli_args import default_configuration
from comfy.cli_args_types import PerformanceFeature
from comfy.distributed.process_pool_executor import ProcessPoolExecutor
from comfy.model_downloader import add_known_models
from comfy.model_downloader_types import HuggingFile
from . import workflows


def _discover_workflows():
    add_known_models("loras", HuggingFile(
        "artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
        "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors"
    ))
    return {
        f.name: f
        for f in importlib.resources.files(workflows).iterdir()
        if f.is_file() and f.name.endswith(".json")
    }


@pytest.fixture(scope="function")
async def client():
    config = default_configuration()
    config.disable_all_custom_nodes = True
    config.guess_settings = True
    config.fast = {PerformanceFeature.CublasOps}
    with ProcessPoolExecutor(max_workers=1) as executor:
        async with Comfy(configuration=config, executor=executor) as c:
            yield c


@pytest.mark.asyncio
@pytest.mark.parametrize("name, workflow_file", _discover_workflows().items())
async def test_workflow(name: str, workflow_file, client: Comfy):
    workflow = json.loads(workflow_file.read_text(encoding="utf8"))
    prompt = Prompt.validate(workflow)
    outputs = await client.queue_prompt(prompt)
    assert len(outputs) > 0
```

## Saving Image Outputs

The `SaveImage` node writes images to disk and returns their paths. Extract the output path from the workflow results to load or verify the generated image:

```python
from pathlib import Path
from PIL import Image


@pytest.mark.asyncio
async def test_generates_valid_image(client: Comfy):
    workflow = json.loads(Path("workflows/sd15-basic-0.json").read_text())
    prompt = Prompt.validate(workflow)
    outputs = await client.queue_prompt(prompt)

    save_node_id = next(
        key for key, value in prompt.items() if value.class_type == "SaveImage"
    )
    image_path = outputs[save_node_id]["images"][0]["abs_path"]

    img = Image.open(image_path)
    assert img.size == (512, 512)
    assert img.mode == "RGB"
```

## Snapshot Testing with pytest-image-diff

Use [pytest-image-diff](https://pypi.org/project/pytest-image-diff/) to compare generated images against reference snapshots. On first run, the reference image is saved. On subsequent runs, the test fails if the output differs beyond a threshold.

Install:

```shell
uv pip install pytest-image-diff
```

```python
from pathlib import Path
from PIL import Image


@pytest.mark.asyncio
async def test_image_matches_snapshot(client: Comfy, image_diff):
    workflow = json.loads(Path("workflows/sd15-basic-0.json").read_text())
    prompt = Prompt.validate(workflow)

    # Pin the seed for deterministic output
    sampler_node_id = next(
        key for key, value in prompt.items() if value.class_type == "KSampler"
    )
    prompt[sampler_node_id].inputs["seed"] = 42

    outputs = await client.queue_prompt(prompt)

    save_node_id = next(
        key for key, value in prompt.items() if value.class_type == "SaveImage"
    )
    image_path = outputs[save_node_id]["images"][0]["abs_path"]
    result = Image.open(image_path)

    image_diff(result, threshold=0.001)
```

Reference images are stored in a `image_snapshots/` directory next to your test file. Run with `--image-diff-update` to regenerate snapshots:

```shell
pytest --image-diff-update tests/inference/test_workflows.py
```

Note: deterministic output requires pinning the seed and using `--deterministic` in your ComfyUI configuration. Even then, results may vary across GPU architectures.

---

## Testing Custom Nodes

This section covers setting up inference tests for your own custom node package, using ComfyUI as a library in CI.

### Project Layout

```
my-custom-nodes/
├── pyproject.toml
├── my_nodes/
│   ├── __init__.py
│   └── ...
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_workflows.py
│   └── workflows/
│       ├── __init__.py
│       ├── my-workflow-a.json
│       └── my-workflow-b.json
└── .github/
    └── workflows/
        └── test.yaml
```

Export each workflow from the ComfyUI web UI (either Save or Save -> API Format) and place the JSON files in `tests/workflows/`. Both formats are accepted — UI workflows are converted automatically.

### pyproject.toml

Add ComfyUI and test dependencies as an optional extra:

```toml
[project]
name = "my-comfyui-nodes"
version = "1.0.0"
dependencies = [
    "torch",
]

[project.optional-dependencies]
ci = [
    "comfyui",
    "pytest",
    "pytest-asyncio",
    "pillow",
]

[project.entry-points."comfyui.custom_nodes"]
my_nodes = "my_nodes"
```

Install for testing with (the index is where the `comfyui` package resolves):

```shell
uv pip install --torch-backend=auto --extra-index-url https://nodes.appmana.com/simple/ -e ".[ci]"
```

### Registering Models

Register models your workflows need in `conftest.py` using `add_known_models`. ComfyUI downloads them on demand from Hugging Face when a workflow references the filename:

```python
import importlib.resources
import itertools

import pytest
from comfy.api.components.schema.prompt import Prompt
from comfy.cli_args import default_configuration
from comfy.cli_args_types import PerformanceFeature
from comfy.client.embedded_comfy_client import Comfy
from comfy.distributed.process_pool_executor import ProcessPoolExecutor
from comfy.model_downloader import add_known_models
from comfy.model_downloader_types import HuggingFile
# in other words, the conftest is adjacent to the workflows directory, which is a package
from . import workflows


add_known_models("diffusion_models",
    HuggingFile("my-org/my-model", "my-model-fp8.safetensors"),
)

add_known_models("loras",
    HuggingFile("my-org/my-lora", "lora.safetensors", save_with_filename="my-lora.safetensors"),
)

add_known_models("controlnet",
    HuggingFile("my-org/my-controlnet", "diffusion_pytorch_model.safetensors", save_with_filename="my-controlnet.safetensors"),
)
```

Use `save_with_filename` when the repo filename is generic (like `diffusion_pytorch_model.safetensors`) and your workflow references a more descriptive name.

### Parameterizing Across Configurations

ComfyUI's own test suite parameterizes each workflow against multiple configuration combinations (attention backends, VRAM modes, performance features). This catches regressions across different hardware paths. The pattern uses `itertools.product` to generate a matrix of config options:

```python
def _generate_config_params():
    attn_options = [
        {"use_pytorch_cross_attention": True},
        {"use_sage_attention": True},
    ]
    vram_options = [
        {"guess_settings": True},
    ]
    fast_options = [
        {"fast": {PerformanceFeature.CublasOps}},
    ]
    for attn, vram, fst in itertools.product(attn_options, vram_options, fast_options):
        config_update = {}
        config_update.update(attn)
        config_update.update(vram)
        config_update.update(fst)
        yield config_update


@pytest.fixture(
    scope="function",
    params=_generate_config_params(),
    ids=lambda p: ",".join(f"{k}={v}" for k, v in p.items()),
)
async def client(request):
    config = default_configuration()
    config.disable_all_custom_nodes = True
    config.update(request.param)
    with ProcessPoolExecutor(max_workers=1) as executor:
        async with Comfy(configuration=config, executor=executor) as c:
            yield c
```

Each workflow test then runs once per configuration combination. For simpler setups, use a single fixed configuration instead.

### test_workflows.py

Discover workflow JSON files from the `workflows/` package and parameterize over them:

```python
import importlib.resources
import json

import pytest
from comfy.api.components.schema.prompt import Prompt
from comfy.client.embedded_comfy_client import Comfy
from PIL import Image
from . import workflows


def _discover_workflows():
    return {
        f.name: f
        for f in importlib.resources.files(workflows).iterdir()
        if f.is_file() and f.name.endswith(".json")
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("name, workflow_file", _discover_workflows().items())
async def test_workflow(name: str, workflow_file, client: Comfy):
    workflow = json.loads(workflow_file.read_text(encoding="utf8"))
    prompt = Prompt.validate(workflow)
    outputs = await client.queue_prompt(prompt)

    if any(v.class_type == "SaveImage" for v in prompt.values()):
        save_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
        image_path = outputs[save_node_id]["images"][0]["abs_path"]
        img = Image.open(image_path)
        assert img.size[0] > 0 and img.size[1] > 0
    else:
        assert len(outputs) > 0
```

### Test environments

The checked-in CI workflows and platform locks define the supported runtime
matrix. [Compatibility](compatibility.md) identifies the hardware targets.
DynamicVRAM and the shared pinned-memory budget account for host and container
limits; low-memory flags are explicit overrides, not requirements for every
GPU with 16 GiB of VRAM. Gated models require credentials and accepted terms.

## Mocking folder_paths in Tests

This fork uses execution-context-scoped `folder_names_and_paths` instead of a module-level global. Do not use `unittest.mock.patch` to override `folder_paths` attributes — use `FolderNames` and `context_folder_names_and_paths` instead:

```python
from comfy.component_model.folder_path_types import FolderNames
from comfy.execution_context import context_folder_names_and_paths

fn = FolderNames()
fn["custom_nodes"] = ([str(custom_nodes_dir)], set())  # set(), not None
with context_folder_names_and_paths(fn):
    manager = CustomNodeManager()  # must be created INSIDE the context
    result = manager.build_translations()
```

Key rules:
- Use `set()` for supported_extensions, not `None`
- Create objects that read `folder_paths.folder_names_and_paths` at init time **inside** the context block
- For other mock targets, use the full module path: `patch("comfy.cmd.folder_paths.xxx")`, `patch("comfy.app.xxx")`

---

## Inference benchmark results

Inference fixtures use at least two sampling steps. The native workflow and
custom-node example suites print a final table with synchronized second-iteration
wall time, iterations per second, GPU/device, PyTorch version, requested TP/PP,
and observed attention backend. The first iteration is excluded from this
measurement. `--inference-benchmark-json=PATH` also saves the rows as JSON.

Measurements cover the native k-diffusion and UniPC iteration loops. An attention
backend is reported only when its instrumented entry point is called; custom or
remote-rank implementations can appear as `not observed`. A missing measurement
is not a zero-duration result. These short execution checks do not establish
image quality, and comparisons require the same weights, workflow, hardware,
attention configuration, and competing GPU load.

## Workflow test coverage

The custom-node example suite discovers workflows, substitutes stub media, and
reduces inference cost to check execution. Missing examples are skipped;
missing-model errors are warnings, and other execution failures fail the test.

The Playwright parity test compares Python `convert_ui_to_api()` with the
frontend's `graphToPrompt()` in headless Chromium. Frontend outputs are cached
by `comfyui-frontend-package` version. An uncached comparison requires browser
binaries; without them, the test is skipped rather than passed.

## IPython notebook tests

The `notebook-tests` extra supplies the kernel test dependencies.
`@pytest.mark.ipython` runs a module-level test body in a fresh Jupyter/IPython
kernel using pytest's Python interpreter. Kernel errors fail the test, and the
kernel is shut down afterward. No browser or notebook server is needed.

`@pytest.mark.ipython(capture_fd_output=False)` exercises notebook streams
without a file descriptor. Marked functions must be self-contained:
JSON-serializable parametrization is supported, but host pytest fixtures are
not passed into the kernel.

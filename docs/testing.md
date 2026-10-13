# Testing Workflows

This page covers automated testing of ComfyUI workflows using pytest. For general library usage, see [Embedded / Library Usage](embedded.md).

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
    config.novram = True
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
        key for key, value in prompt.items() if value.get("class_type") == "SaveImage"
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
        key for key, value in prompt.items() if value.get("class_type") == "KSampler"
    )
    prompt[sampler_node_id]["inputs"]["seed"] = 42

    outputs = await client.queue_prompt(prompt)

    save_node_id = next(
        key for key, value in prompt.items() if value.get("class_type") == "SaveImage"
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
        {"novram": True},
        {"normalvram": True},
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

### Pre-populating Models

Models are downloaded on demand during test runs, but you can pre-populate the Hugging Face cache to avoid downloads in CI:

```shell
huggingface-cli download my-org/my-model my-model-fp8.safetensors
huggingface-cli download my-org/my-lora lora.safetensors
```

Files are stored in `~/.cache/huggingface/` and symlinked into the models directory at runtime. For gated repos, authenticate first:

```shell
huggingface-cli login --token $HF_TOKEN
```

### GitHub Actions Workflow

Inference tests require a GPU. There are three options for running GPU-equipped GitHub Actions:

1. **[GitHub-hosted GPU runners](https://docs.github.com/en/actions/concepts/runners/larger-runners)** (Team/Enterprise plan, $0.07/min): Go to Settings > Actions > Runners > New GitHub-hosted runner, select the "NVIDIA GPU-Optimized Image" under the Partner tab, then choose a GPU-powered size (T4 with 16 GB VRAM). Assign it a label.
2. **Self-hosted runners**: Install the [GitHub Actions runner agent](https://github.com/actions/runner) on a machine with an NVIDIA GPU. The machine stays online and picks up jobs from your repository. Use labels like `[self-hosted, gpu]` in `runs-on`.
3. **[RunsOn](https://runs-on.com/runners/gpu/)** (any GitHub plan, ~$0.009/min for T4 on AWS spot): Deploy their CloudFormation stack to your AWS account, then use their label format in `runs-on`. Supports the full range of NVIDIA GPUs (T4, A10G, L4, L40S, A100, H100). No GitHub plan restriction.

#### Memory Constraints on CI Runners

GPU CI runners are typically memory-constrained compared to workstation hardware:

- **T4 runners have 16 GB VRAM** — `config.novram = True` is required. Without it, large models (FLUX, SD3, video models) will OOM immediately. The `novram` flag aggressively offloads weights to CPU between operations.
- **Runners with less than 32 GB RAM** (GitHub's T4 runner has 28 GB) — `config.disable_pinned_memory = True` is required. Pinned (page-locked) memory cannot be swapped, so on RAM-constrained runners the remaining unpinned memory thrashes to disk and performance collapses or the process is killed.
- **Enable swap** as a safety net. Even with `novram` and `disable_pinned_memory`, some models temporarily spike RAM usage during loading. A swap file prevents OOM kills at the cost of brief slowdowns.

Set these in your client fixture:

```python
@pytest.fixture(scope="function")
async def client():
    config = default_configuration()
    config.novram = True
    config.disable_pinned_memory = True
    config.disable_all_custom_nodes = True
    with ProcessPoolExecutor(max_workers=1) as executor:
        async with Comfy(configuration=config, executor=executor) as c:
            yield c
```

Create `.github/workflows/test.yaml`:

```yaml
name: Inference Tests
on:
  pull_request:
    types: [opened, synchronize, reopened]
    paths-ignore:
      - '**/*.md'
      - 'docs/**'
  workflow_dispatch:

concurrency:
  group: ${{ github.repository }}-test-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true

jobs:
  test:
    # Option 1: GitHub-hosted GPU runner (Team/Enterprise plan).
    # Use the label you assigned when creating the runner in org settings.
    runs-on: my-gpu-runner

    # Option 2: Self-hosted runner with a GPU label.
    # Install the runner agent on a machine with an NVIDIA GPU:
    #   mkdir actions-runner && cd actions-runner
    #   curl -o actions-runner-linux-x64.tar.gz -L https://github.com/actions/runner/releases/download/v2.331.0/actions-runner-linux-x64-2.331.0.tar.gz
    #   tar xzf actions-runner-linux-x64.tar.gz
    #   ./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO --token YOUR_TOKEN --labels gpu
    #   ./run.sh  # or install as a systemd service with ./svc.sh install
    # runs-on: [self-hosted, gpu]

    # Option 3: RunsOn (runs-on.com) — on-demand AWS GPU runners, any GitHub plan.
    # Deploy their CloudFormation stack to your AWS account, then:
    # runs-on: "runs-on=${{ github.run_id }}/family=g4dn.xlarge/image=ubuntu22-gpu-x64"

    steps:
      - uses: actions/checkout@v4

      # Prevent OOM kills during model loading spikes.
      # GitHub's T4 runner has 28 GB RAM which is tight for large models.
      - name: Enable swap
        run: |
          sudo fallocate -l 16G /swapfile
          sudo chmod 600 /swapfile
          sudo mkswap /swapfile
          sudo swapon /swapfile

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      # Cache ~/.cache between runs: huggingface models, uv packages, etc.
      # For self-hosted runners, this directory persists naturally.
      - name: Cache
        uses: actions/cache@v4
        with:
          path: ~/.cache
          key: cache-${{ hashFiles('pyproject.toml', 'tests/conftest.py') }}
          restore-keys: cache-

      - name: Install dependencies
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          uv pip install --torch-backend=auto -e ".[ci]"

      - name: Run tests
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: pytest tests/ -x -vv
```

Key points:

- **Swap**: The 16 GB swap file acts as a safety net for RAM spikes during model loading. Without it, the kernel OOM-kills the process. This is a one-time cost at the start of the job.
- **Cache**: `actions/cache` persists all of `~/.cache` between runs — this covers Hugging Face models, `uv` package downloads, and other cached data. The key is derived from `pyproject.toml` (dependency changes invalidate the `uv` cache) and `conftest.py` (model registration changes invalidate the HF cache). For self-hosted runners, `~/.cache` persists naturally across jobs.
- **`HF_TOKEN`**: Required for gated models (FLUX, SD3, etc.). Add it as a repository secret under Settings > Secrets and variables > Actions.
- **`cancel-in-progress`**: Kills running jobs when a new commit is pushed, avoiding wasted GPU time.
- **`paths-ignore`**: Skips expensive GPU tests for documentation-only changes.

### Running Tests Locally

```shell
uv pip install --torch-backend=auto -e ".[ci]"
pytest tests/ -x -vv

# Run a specific workflow
pytest tests/ -k "my-workflow-a" -vv
```

---

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

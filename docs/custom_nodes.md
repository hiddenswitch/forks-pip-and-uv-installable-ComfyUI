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

### Self-Hosting the Package Index

The `serve-pip` command runs a local PEP 503 "simple" index backed by the Comfy registry. The facade builds wheels on demand from ComfyUI-Manager registry metadata, injects dependency metadata, and loads the actual nodes through the existing vanilla custom-node mitigation layer.

```bash
comfyui serve-pip --listen 0.0.0.0 --port 8190
```

Then install from it:

```bash
uv pip install --extra-index-url http://localhost:8190/simple/ comfyui-wanvideowrapper
```

The wheel cache can be a normal directory or any writable fsspec URI:

```bash
comfyui serve-pip --pip-facade-cache-prefix=/tmp/comfyui-pip-facade
comfyui serve-pip --pip-facade-cache-prefix=s3://my-bucket/comfyui/pip-facade
```

For SeaweedFS, prefer the S3 API over a SeaweedFS CSI-mounted directory:

```bash
AWS_ACCESS_KEY_ID=... \
AWS_SECRET_ACCESS_KEY=... \
comfyui serve-pip \
  --pip-facade-cache-prefix=s3://my-bucket/comfyui/pip-facade \
  --pip-facade-cache-s3-endpoint-url=http://seaweedfs-s3:8333
```

The S3 backend publishes each wheel as a complete object PUT. A CSI-mounted
object store can expose weaker filesystem semantics than a normal local disk,
which is not a good fit for generated wheel cache files served directly to pip
and uv clients.

Use `--pip-facade-only-known-nodes` to restrict to tested nodes. Use `snapshot-pip-registry` to pre-generate a sqlite snapshot for faster startup.

The installed wheel is only a transport wrapper. It exposes a `comfyui.custom_nodes` entry point, points ComfyUI at the vendored custom-node directory, and then the node is imported as a vanilla custom node with the same compatibility shims used for `custom_nodes/` clones.

### Local Facade Build for Long-Tail Packages

The bundled `pip_facade_registry_snapshot.sqlite.xz` carries the `class_type → canonical_name` mapping for registry packages; the public index builds and caches wheels on demand. When `comfyui run-workflow ... --all` resolves a workflow's custom-node requirements and a package isn't on the index, the CLI falls back to building a facade wheel **locally in-process** rather than installing from git directly:

1. `comfy.custom_node_facade.repo_lookup.resolve_package_repo_url(canonical_name)` looks up the source repository URL via comfy.org's node registry (`api.comfy.org/nodes/<name>`), then ComfyUI-Manager's `extension-node-map.json` as a second source.
2. `comfy.custom_node_facade.local_build.build_local_facade_wheel(canonical_name)` synthesizes a one-shot `FacadeRegistry`, points `FacadeWheelBuilder` at the GitHub or GitLab archive URL (`https://api.github.com/repos/<owner>/<repo>/zipball` or `https://gitlab.com/api/v4/projects/<encoded>/repository/archive.zip`), and produces the same kind of wheel the remote facade serves.
3. `uv pip install <wheel-path>` installs it.

This goes through the same `_appmana_facade_<name>` rename and stub-shim path as a remotely-built wheel, so ComfyUI's node loader picks it up without any special-casing. Examples that resolve via this path: `comfyui-umeairt-toolkit` (GitLab), `daxnodes` (GitHub but no wheel on the index), `comfyui-fbcnn`, `comfyui-image-saver`. Direct installers are not used because they would skip the rename and break ComfyUI's vanilla-custom-node import path.

Cache lives at the configured ComfyUI cache location; rebuild it through the
facade cache command rather than installing the repository directly.

---

## Programmatic Custom Node Management

This section documents how custom nodes are managed programmatically, covering the ComfyUI-Manager APIs, the node-to-package mapping system, dependency management, and the compatibility mitigations this fork applies.

### Architecture Overview

```
comfy-cli              comfyui_manager REST API       comfyui_manager Python API
(thin wrapper)    -->  POST /v2/manager/queue/task --> UnifiedManager (singleton)
                       GET  /v2/customnode/installed      |
                       GET  /v2/customnode/getmappings    +--> cnr_utils.install_node()  --> CNR API (api.comfy.org)
                                                          +--> git.Repo.clone_from()     --> GitHub
                                                          +--> manager_util.PIPFixer      --> pip/uv
```

`comfy-cli` (`comfy node install <name>`) does not implement installation logic. It shells out to `cm-cli.py` inside the cloned ComfyUI-Manager directory, which calls the same `UnifiedManager` methods documented below.

### Node Identification

Custom nodes are identified by three install types:

| Type | ID Format | Version | Example |
|------|-----------|---------|---------|
| **CNR** (ComfyUI Node Registry) | `node-name` | Semantic version (`1.2.3`) | `comfyui-impact-pack@1.0.0` |
| **Git clone** ("unknown") | `repo-name` | Git commit hash | `ComfyUI-KJNodes@unknown` |
| **Nightly** | `node-name` | `nightly` + commit hash | `comfyui-impact-pack@nightly` |

The `InstalledNodePackage` dataclass (`comfyui_manager.common.node_package`) stores this:

```python
@dataclass
class InstalledNodePackage:
    id: str           # node ID
    fullpath: str     # installation directory
    disabled: bool    # moved to .disabled/
    version: str      # semantic version, commit hash, "unknown", or "nightly"
```

### UnifiedManager Python API

The `UnifiedManager` singleton at `comfyui_manager.glob.manager_core.unified_manager` is the core programmatic interface. It requires ComfyUI's folder paths to be initialized.

#### Installation

```python
from comfyui_manager.glob.manager_core import unified_manager

# Install from ComfyUI Node Registry by name + version
result = unified_manager.cnr_install("comfyui-impact-pack", version_spec="1.0.0")
# result.result == True on success
# result.to_path == "/path/to/custom_nodes/comfyui-impact-pack"

# Install from git URL
result = unified_manager.repo_install(
    url="https://github.com/user/ComfyUI-SomeNodes",
    repo_path="/path/to/custom_nodes/ComfyUI-SomeNodes"
)

# Smart install (async) - resolves CNR vs git, handles enable/disable/version switching
result = await unified_manager.install_by_id(
    node_id="comfyui-impact-pack",
    version_spec="1.0.0",    # or "latest", "nightly", "unknown", None
    channel="default",
    mode="remote"
)
```

`cnr_install` calls `cnr_utils.install_node(node_id, version)` which hits the CNR API at `https://api.comfy.org/nodes/{node_id}/install?version={version}` to get a download URL, downloads the zip, extracts it into `custom_nodes/{node_id}/`, creates a `.tracking` file, then runs `execute_install_script` for pip dependencies and `install.py`.

`repo_install` runs `git.Repo.clone_from(url, repo_path, recursive=True)` then runs post-install scripts.

`install_by_id` is the highest-level method. When `version_spec` is `None`, it auto-resolves: checks if the node is already enabled (skip), disabled (enable it), or absent (install via CNR or git). It handles version switching between CNR/nightly transparently.

#### Version specification

The `resolve_node_spec` method parses version strings:

```python
# Accepts "name@version" format
spec = unified_manager.resolve_node_spec("comfyui-impact-pack@latest")
# Returns: ("comfyui-impact-pack", "1.2.3", True)  -- resolved semantic version

spec = unified_manager.resolve_node_spec("comfyui-impact-pack@nightly")
# Returns: ("comfyui-impact-pack", "nightly", True)

spec = unified_manager.resolve_node_spec("comfyui-impact-pack")
# Returns: ("comfyui-impact-pack", <auto-resolved>, False)
```

When `version_spec` is `"latest"`, it resolves to the concrete semantic version from the CNR map.

#### Uninstallation

```python
result = unified_manager.unified_uninstall("comfyui-impact-pack", is_unknown=False)
# Removes from active_nodes, nightly_inactive_nodes, and cnr_inactive_nodes
# Deletes the installation directory
```

This removes the node from all registries (active, inactive CNR, inactive nightly) and deletes the directory. It refuses to uninstall `comfyui-manager` itself.

#### Enable / Disable / Update

```python
unified_manager.unified_enable("node-id", version_spec="1.0.0")
unified_manager.unified_disable("node-id", is_unknown=False)
unified_manager.unified_update("node-id", version_spec="1.0.0")
unified_manager.cnr_switch_version("node-id", version_spec="2.0.0")
```

Disabling moves the node directory under `custom_nodes/.disabled/`. Enabling moves it back.

#### Return type

All methods return `ManagedResult`:

```python
class ManagedResult:
    action: str       # "install-cnr", "install-git", "uninstall", "enable", "skip", etc.
    result: bool      # True if successful
    msg: str          # error message on failure
    to_path: str      # installation path (for installs)
    target: str       # version spec
```

### REST API

When ComfyUI is running with `--enable-manager`, the manager exposes REST endpoints. All install/uninstall operations go through an async task queue.

#### Install a node

```
POST /v2/manager/queue/task
Content-Type: application/json

{
    "ui_id": "task-123",
    "client_id": "test",
    "kind": "install",
    "params": {
        "id": "comfyui-impact-pack",
        "version": "1.0.0",
        "selected_version": "1.0.0",
        "mode": "remote",
        "channel": "default"
    }
}
```

The `params` field is validated as `InstallPackParams` (Pydantic model in `comfyui_manager.data_models.generated_models`):

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Node ID or `publisher/node-name` |
| `version` | `str` | Semantic version or git commit hash |
| `selected_version` | `str` | `"latest"`, `"nightly"`, or specific version |
| `mode` | `"remote" \| "local" \| "cache"` | Database source |
| `channel` | `"default" \| "recent" \| "legacy" \| ...` | Channel |
| `skip_post_install` | `bool` | Skip post-install scripts |

#### Uninstall a node

```
POST /v2/manager/queue/task
Content-Type: application/json

{
    "ui_id": "task-124",
    "client_id": "test",
    "kind": "uninstall",
    "params": {
        "node_name": "comfyui-impact-pack",
        "is_unknown": false
    }
}
```

#### Other operations

| Kind | Params model | Description |
|------|-------------|-------------|
| `"enable"` | `EnablePackParams(cnr_id=...)` | Enable a disabled node |
| `"disable"` | `DisablePackParams(node_name=..., is_unknown=...)` | Disable a node |
| `"update"` | `UpdatePackParams(node_name=..., node_ver=...)` | Update to latest |
| `"fix"` | `FixPackParams(node_name=..., node_ver=...)` | Re-run dependency install |

#### Query endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v2/customnode/installed` | List all installed custom nodes |
| `GET /v2/manager/queue/status` | Task queue status |
| `GET /v2/manager/version` | Manager version |
| `GET /v2/customnode/getmappings?mode=remote` | Node class_type to package mapping |

### Mapping class_types to Custom Node Packages

ComfyUI-Manager maintains a bundled `extension-node-map.json` that maps git repository URLs to the list of `NODE_CLASS_MAPPINGS` keys (class_types) each package provides:

```json
{
    "https://github.com/user/ComfyUI-SomeNodes": [
        ["NodeClassA", "NodeClassB"],
        {
            "title_aux": "Some Nodes Pack",
            "nodename_pattern": "SomePrefix.*"
        }
    ]
}
```

The first element is the list of class_type strings. The second element contains metadata, including an optional `nodename_pattern` regex for nodes that follow a naming convention.

#### How the mapping works

The `GET /v2/customnode/getmappings` endpoint (`manager_server.py:1356`):

1. Loads `extension-node-map.json` via `get_data_by_mode(mode, 'extension-node-map.json')` (fetched from the channel URL remotely, or read from local cache/bundled file)
2. Calls `map_to_unified_keys()` which converts git-URL keys to CNR node IDs using `unified_manager.get_cnr_by_repo(url)`
3. Builds a set of all known class_types across all packages
4. Compares against `nodes.NODE_CLASS_MAPPINGS.keys()` (currently loaded class_types) to find unaccounted nodes
5. Applies `nodename_pattern` regex matching for missing nodes

#### Workflow dependency resolution

The function at `manager_core.py:2785` resolves which custom nodes a workflow requires:

1. Parses the workflow JSON, extracts all `class_type` values from nodes (skipping `Reroute`, `Note`, and `workflow/` prefixed names)
2. Loads `extension-node-map.json` and builds a **reverse map**: `class_type -> [repo_url]`
3. Applies a `preemption_map` for class_types claimed by core ComfyUI
4. Falls back to `nodename_pattern` regex for unresolved class_types
5. Returns `(used_exts, unknown_nodes)` -- the set of required extensions and any class_types that couldn't be mapped

To build a `class_type -> node_id` lookup for programmatic use:

```python
# After map_to_unified_keys(), the data is: cnr_node_id -> [class_type_list, metadata]
# Invert it:
reverse_map = {}
for node_id, (class_types, metadata) in mapping.items():
    for ct in class_types:
        reverse_map[ct] = node_id
```

### Vanilla Environment Compatibility Layer

This fork restructures ComfyUI as an installable Python package (`comfy.*`), but vanilla custom nodes expect top-level modules like `nodes`, `folder_paths`, `execution`, `server`, etc. The compatibility layer in `comfy_compatibility/vanilla.py` bridges this gap.

`prepare_vanilla_environment()` (called once during startup) injects shims into `sys.modules`:

| Expected by custom nodes | Actual location in this fork |
|---|---|
| `import nodes` | `comfy.nodes.base_nodes` (via `_NodeShim`) |
| `import folder_paths` | `comfy.cmd.folder_paths` |
| `import execution` | `comfy.cmd.execution` |
| `import server` | `comfy.cmd.server` |
| `import model_patcher` | `comfy.model_patcher` |
| `import cuda_malloc` | `comfy.cmd.cuda_malloc` |
| `import latent_preview` | `comfy.cmd.latent_preview` |
| `import comfyui_version` | Synthetic module with `__version__` |
| `comfy_extras.*` | Re-exported under shortened names |

The `_NodeClassMappingsShim` provides a lazy, reference-counted view of `NODE_CLASS_MAPPINGS` that returns all currently loaded nodes when activated (during import/execution) or just base nodes when deactivated.

The `_PromptServerStub` (`vanilla_node_importing.py:53`) provides a stub `PromptServer.instance` so nodes that call `server.PromptServer.instance.send_sync()` during import don't crash.

### Manager Integration

`comfy/manager_integration.py` provides the bridge between this fork's startup sequence and comfyui_manager:

| Function | Purpose |
|---|---|
| `init_manager(args)` | Import and initialize comfyui_manager if `--enable-manager` is set |
| `prestartup()` | Run manager's prestartup script |
| `start()` | Start manager UI endpoints |
| `get_middleware()` | Get manager's aiohttp middleware |
| `should_be_disabled(module_path)` | Check if manager policy blocks a specific node |

`should_be_disabled` is called during node loading (`vanilla_node_importing.py:94,262`) to respect manager's enable/disable state for each custom node directory.

### Import Process for Vanilla Custom Nodes

The full loading sequence is in `comfy/nodes/vanilla_node_importing.py`:

1. **`mitigated_import_of_vanilla_custom_nodes()`** -- Entry point. Calls `prepare_vanilla_environment()`, collects `custom_nodes` paths, then runs prestartup and import phases.

2. **Prestartup phase** (`_vanilla_load_importing_execute_prestartup_script`):
   - Iterates over every directory in `custom_nodes/`
   - Skips `.disabled` directories and nodes blocked by manager policy
   - For ComfyUI-Manager specifically: patches its `security_check` to fail gracefully, suppresses its logging handler, and sets `COMFYUI_PATH`/`COMFYUI_FOLDERS_BASE_PATH` env vars
   - Executes each node's `prestartup_script.py` if present

3. **Import phase** (`_vanilla_load_custom_nodes_2`):
   - Iterates over every directory/file in `custom_nodes/`
   - Skips disabled nodes, blacklisted nodes, and nodes blocked by manager policy
   - For each module, calls `_vanilla_load_custom_nodes_1` which:
     - Imports the module via `importlib`
     - Applies `_exec_mitigations` for known problematic nodes (patches `folder_paths.__file__`, optionally blocks pip installs)
     - Extracts `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`, and `WEB_DIRECTORY`
   - Records import times for diagnostics

### Workspace Compatibility

`comfy_compatibility/workspace.py` handles the case where ComfyUI is run from a cloned upstream workspace directory (where `nodes.py`, `comfy/`, `comfy_extras/` etc. exist as bare directories without `__init__.py`). It:

1. Detects if the workspace has a `nodes.py` (indicating an upstream-style layout)
2. Creates `__init__.py` files in all directories containing `.py` files under `comfy/`, `comfy_extras/`, `comfy_execution/`, `comfy_api/`, and `comfy_config/`
3. Adds these files to `.git/info/exclude` to avoid polluting git status
4. Restarts the process so the new packages are importable

### Import Order Control

`comfy_compatibility/imports.py` provides `ImportContext`, a context manager that temporarily overrides Python's import resolution order for specific modules. It inserts a custom `PathFinder` into `sys.meta_path` that controls whether modules are resolved from the main script directory, the current working directory, or site-packages first. This is used to ensure the correct version of ambiguous modules (like `comfy`) is imported when both a workspace copy and a pip-installed copy exist.

### Directory Layout

```
custom_nodes/
    some-node/                    # active CNR or git-clone node
        __init__.py
        .tracking                 # present for CNR nodes (lists extracted files)
        pyproject.toml            # CNR nodes have version info here
        requirements.txt
        install.py                # optional post-install script
    .disabled/
        another-node/             # disabled node (moved here by manager)
```

---

## Authoring Custom Nodes

These instructions will allow you to quickly author installable custom nodes.

### How the Package Index Works

The `serve-pip` command exposes vanilla custom nodes as PEP 503 Python packages. It:

- enumerates installable custom nodes from ComfyUI-Manager's registry data
- resolves versions from the Comfy registry (`api.comfy.org`)
- injects known missing dependencies from this fork's compatibility tables
- builds wheels on demand
- publishes `comfyui.custom_nodes` entry points that point back to vendored vanilla custom-node directories

Each generated wheel contains the upstream custom node repository contents, generated `METADATA` dependencies merged from registry metadata, and a `comfyui.custom_nodes` entry point. At runtime, the entry point advertises the vendored custom-node directory to `comfy.nodes.package`, and ComfyUI imports the vendored repo through the vanilla custom-node importer. Facade-installed nodes behave like ecosystem custom nodes, not like native packaged extensions.

For nodes without published versions on the Comfy Node Registry, `inject_version` in `comfy/component_model/node_registry.py` generates a synthetic version from the GitHub archive. URL dependencies (like `sam2 @ git+https://...`) are rewritten to plain package names using `packaging.requirements.Requirement`.

Use `snapshot-pip-registry` to pre-generate a sqlite snapshot of the full registry for faster startup. The snapshot can be served directly from disk (no decompression needed for `.db` files) and updated periodically by a separate process.

#### Wheel cache

Built wheels are cached at `{cache-prefix}/v{revision}/{project}/{filename}.whl`.
`--pip-facade-cache-revision` selects the cache revision.

#### Dependency rewriting

The facade rewrites certain dependencies in generated wheel metadata:

- **Stripped versions**: `numpy`, `jax`, `jaxlib`, `image-reward`, `timm` are emitted without version constraints so the resolver picks the latest compatible version.
- **opencv homogenization**: All opencv variants (`opencv-python`, `opencv-python-headless`, `opencv-contrib-python`, `opencv-contrib-python-headless`) are rewritten to bare `opencv-contrib-python-headless`.
- **onnxruntime platform expansion**: `onnxruntime` (and all variants) is expanded to platform-conditional dependencies: `onnxruntime` on macOS/ARM Linux, `onnxruntime-gpu` on x86 Linux and Windows.
- **URL dependency stripping**: URL dependencies like `sam2 @ git+https://...` are rewritten to plain package names.

#### Using `pyproject.toml` for projects with existing `requirements.txt`

Suppose your custom nodes called `my_comfyui_nodes` has a folder layout that looks like this:

```
__init__.py
some_python_file.py
requirements.txt
LICENSE.txt
some_directory/some_code.py
```

First, add an `__init__.py` to `some_directory`, so that it is a Python package:

```
__init__.py
some_python_file.py
requirements.txt
LICENSE.txt
some_directory/__init__.py
some_directory/some_code.py
```

Then, if your `NODE_CLASS_MAPPINGS` are declared in `__init__.py`, use the following as a `pyproject.toml`, substituting your actual project name:

**pyproject.toml**

```toml
[project]
name = "my_comfyui_nodes"
description = "My nodes description."
version = "1.0.0"
license = { file = "LICENSE.txt" }
dynamic = ["dependencies"]

[project.urls]
Repository = "https://github.com/your-github-username/my-comfyui-nodes"
#  Used by Comfy Registry https://comfyregistry.org

[tool.comfy]
PublisherId = "your-github-username"
DisplayName = "my_comfyui_nodes"
Icon = ""

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["my_comfyui_nodes", "my_comfyui_nodes.some_directory"]
package-dir = { "my_comfyui_nodes" = ".", "my_comfyui_nodes.some_directory" = "some_directory" }

[tool.setuptools.dynamic]
dependencies = { file = ["requirements.txt"] }

[project.entry-points."comfyui.custom_nodes"]
my_comfyui_nodes = "my_comfyui_nodes"
```

Observe that the directory should now be listed as a package in the `packages` and `package-dir` statement.

#### Using `setup.py`

Create a `requirements.txt`:

```
comfyui
```

Observe `comfyui` is now a requirement for using your custom nodes. This will ensure you will be able to access `comfyui` as a library. For example, your code will now be able to import the folder paths using `from comfy.cmd import folder_paths`. Because you will be using my fork, point your `requirements.txt` at the package index so `comfyui` resolves to it:

```
--extra-index-url https://nodes.appmana.com/simple/
comfyui
```

Additionally, create a `pyproject.toml`:

```
[build-system]
requires = ["setuptools", "wheel", "pip"]
build-backend = "setuptools.build_meta"
```

This ensures you will be compatible with later versions of Python.

Finally, move your nodes to a directory with an empty `__init__.py`, i.e., a package. You should have a file structure like this:

```
# the root of your git repository
/.git
/pyproject.toml
/requirements.txt
/mypackage_custom_nodes/__init__.py
/mypackage_custom_nodes/some_nodes.py
```

Finally, create a `setup.py` at the root of your custom nodes package / repository. Here is an example:

**setup.py**

```python
from setuptools import setup, find_packages
import os.path

setup(
    name="mypackage",
    version="0.0.1",
    packages=find_packages(),
    install_requires=open(os.path.join(os.path.dirname(__file__), "requirements.txt")).readlines(),
    author='',
    author_email='',
    description='',
    entry_points={
        'comfyui.custom_nodes': [
            'mypackage = mypackage_custom_nodes',
        ],
    },
)
```

All `.py` files located in the package specified by the entrypoint with your package's name will be scanned for node class mappings declared like this:

**some_nodes.py**:

```py
from comfy.nodes.package_typing import CustomNode


class Binary_Preprocessor(CustomNode):
    ...


NODE_CLASS_MAPPINGS = {
    "BinaryPreprocessor": Binary_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BinaryPreprocessor": "Binary Lines"
}
```

These packages will be scanned recursively.

Extending the `comfy.nodes.package_typing.CustomNode` provides type hints for authoring nodes.

## Adding Custom Configuration

Declare an entry point for configuration hooks in your **setup.py** that defines a function that takes and returns an
`configargparser.ArgParser` object:

**setup.py**

```python
setup(
    name="mypackage",
    ...
entry_points = {
    'comfyui.custom_nodes': [
        'mypackage = mypackage_custom_nodes',
    ],
    'comfyui.custom_config': [
        'mypackage = mypackage_custom_config:add_configuration',
    ]
},
)
```

**mypackage_custom_config.py**:

```python
import configargparse


def add_configuration(parser: configargparse.ArgParser) -> configargparse.ArgParser:
    parser.add_argument("--openai-api-key",
                        required=False,
                        type=str,
                        help="Configures the OpenAI API Key for the OpenAI nodes", env_var="OPENAI_API_KEY")
    return parser

```

Pass custom options to the server command:

```shell
uv run comfyui serve --openai-api-key=example-value
```

Custom configuration hooks use a separate parser; their options are not listed
in the Typer-generated `comfyui serve --help`. Configuration files describe the
built-in server options; pass custom-node options explicitly on the command line.

Your entry point for adding configuration options should **not** import your nodes. This gives you the opportunity to
use the configuration you added in your nodes; otherwise, if you imported your nodes in your configuration entry point,
the nodes will potentially be initialized without any configuration.

Access your configuration from `cli_args`:

```python
from comfy.cli_args import args
from comfy.cli_args_types import Configuration
from typing import Optional


# Add type hints when accessing args
class CustomConfiguration(Configuration):
    def __init__(self):
        super().__init__()
        self.openai_api_key: Optional[str] = None


args: CustomConfiguration


class OpenAINode(CustomNode):
    ...

    def execute(self):
        openai_api_key = args.openai_api_key
```

---

## Model Discovery & Registration

Models referenced by custom node workflows are registered in `comfy/model_downloader.py` so they can be downloaded on-demand during execution. The download interception layer (`comfy/nodes/download_interception.py`) routes `huggingface_hub.hf_hub_download()`, `snapshot_download()`, and `torch.hub.download_url_to_file()` calls through the centralized model downloader.

## GGUF Native Support

This fork includes native GGUF model loading built into the core model loading pipeline. The `ComfyUI-GGUF` custom node by city96 is **fully compatible but redundant** -- you can:

1. Place `.gguf` files directly in `models/diffusion_models/`, `models/text_encoders/`, `models/clip/`, or any folder where `.safetensors` works
2. Select them from the standard model dropdowns (UNETLoader, CLIPLoader, etc.)
3. No custom node installation required

The native implementation supports the same quantization formats (Q4_K_M, Q8_0, etc.) and uses the same ggml tensor loading path. If you have `ComfyUI-GGUF` installed, it will still work -- the native loader takes precedence for standard node types, while the custom node's specialized loaders remain available for advanced use cases.

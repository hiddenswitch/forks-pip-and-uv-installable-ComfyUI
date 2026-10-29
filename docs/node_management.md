# Manage custom nodes programmatically

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


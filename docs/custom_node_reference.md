# Custom node management reference

Use [node installation](custom_nodes.md) for everyday setup and
[authoring](custom_node_authoring.md) for native packages. This page describes
Manager integration and the compatibility layer for application developers.
Manager interfaces belong to the installed `comfyui_manager` version; check that
version's schema when building automation against them.

## Manager identification and state

Manager distinguishes CNR packages (`node-name`, semantic version), git clones
(`repo-name`, commit/unknown version), and nightly packages (`node-name`, nightly
plus commit). `InstalledNodePackage` records `id`, `fullpath`, `disabled`, and
`version`. Disabled directories live under `custom_nodes/.disabled/`; CNR
installations include a `.tracking` file.

After initializing ComfyUI's folder paths, the Manager singleton is available at
`comfyui_manager.glob.manager_core.unified_manager`:

| Method | Purpose |
|---|---|
| `cnr_install(node_id, version_spec=...)` | Install a registry version |
| `repo_install(url=..., repo_path=...)` | Clone a repository and run installation |
| `await install_by_id(node_id=..., version_spec=..., channel="default", mode="remote")` | Resolve installation, switching, or enablement |
| `resolve_node_spec("node-name@latest")` | Resolve a version specification |
| `unified_enable(node_id, version_spec=...)` | Enable a disabled node |
| `unified_disable(node_id, is_unknown=False)` | Move a node to disabled storage |
| `unified_update(node_id, version_spec=...)` | Update an installed node |
| `cnr_switch_version(node_id, version_spec=...)` | Switch CNR version |
| `unified_uninstall(node_id, is_unknown=False)` | Remove installation and registry state |

Installation can execute dependency installers and node post-install scripts.
Mutation results use `ManagedResult` fields including `action`, `result`, `msg`,
`to_path`, and `target`; check `result` and report `msg` on failure.

## Manager REST interface

Start with `comfyui serve --enable-manager`. Installation changes go through
`POST /v2/manager/queue/task`. An installation body has this shape:

```json
{
  "ui_id": "task-123",
  "client_id": "my-application",
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

Use an actual available version for your package. `InstallPackParams` also accepts
`skip_post_install`. Other queue kinds include `uninstall`, `enable`, `disable`,
`update`, and `fix`, each with its own params model; they do not all accept the
installation body's fields.

| Query | Purpose |
|---|---|
| `GET /v2/customnode/installed` | Installed nodes |
| `GET /v2/manager/queue/status` | Mutation queue status |
| `GET /v2/manager/version` | Manager version |
| `GET /v2/customnode/getmappings?mode=remote` | Node class-to-package mapping |

## Workflow dependency mapping

Manager's `extension-node-map.json` maps repositories to class IDs and optional
`nodename_pattern` expressions. Registry mapping normalizes repository URLs to
CNR IDs, then uses loaded class mappings and patterns to resolve remaining nodes.
The fork's workflow dependency resolver also uses its bundled facade registry
snapshot to map `class_type` values to canonical package names. Missing or
ambiguous metadata can require a manual package selection.

## Vanilla imports

`comfy_compatibility/vanilla.py` establishes aliases during node loading:

| Vanilla import | Packaged implementation |
|---|---|
| `nodes` | `comfy.nodes.base_nodes` through a mapping shim |
| `folder_paths` | `comfy.cmd.folder_paths` |
| `execution` | `comfy.cmd.execution` |
| `server` | `comfy.cmd.server` |
| `model_patcher` | `comfy.model_patcher` |
| `cuda_malloc` | `comfy.cmd.cuda_malloc` |
| `latent_preview` | `comfy.cmd.latent_preview` |

The node mapping shim exposes currently loaded nodes while active and base nodes
otherwise. A `PromptServer` stub supports import-time assumptions, but cannot
provide every live-server behavior in headless applications.

`comfy/nodes/vanilla_node_importing.py` performs prestartup and import phases,
skips disabled/blocked nodes, applies known mitigations, and extracts node/display
mappings and `WEB_DIRECTORY`. `comfy/manager_integration.py` connects startup,
middleware, and Manager policy to this loader. Facade wheels advertise their
vendored directory through entry points and follow this same path.

## Existing workspace imports

`comfy_compatibility/workspace.py` detects an upstream-style workspace via
`nodes.py`. It can create missing `__init__.py` files under ComfyUI's source
directories, add them to `.git/info/exclude`, and restart to make imports
unambiguous. `ImportContext` in `comfy_compatibility/imports.py` controls resolution
order when a workspace copy and an installed package coexist. Do not compensate
for conflicts by manually adding arbitrary source directories to `PYTHONPATH`.

# Installing custom nodes

Install ecosystem nodes into the same Python environment as `comfyui`. The
package facade wraps vanilla custom-node repositories as wheels and loads them
through the fork's compatibility layer. Compatibility fixes cover common import
and dependency assumptions; individual nodes can still require specific hardware,
external programs, or an interactive server.

## Install a package

With your ComfyUI environment active:

```console
uv pip install --extra-index-url https://nodes.appmana.com/simple/ comfyui-wanvideowrapper
comfyui serve
```

Restart the running application after installing or upgrading nodes. Use the
registry's package name, which can differ from a node's visible title or its
workflow `class_type`. Avoid installing the same node as both a wheel and a clone.

## Install a workflow's requirements

```console
comfyui run-workflow workflow.json --dry-run
comfyui run-workflow workflow.json --all
```

`--dry-run` reports discovered models and planned package/model installs without
running inference. `--all` installs missing dependencies and downloads resolvable
models before execution. Resolution depends on registry metadata and workflow
notes; unknown nodes or model filenames may still need manual setup.

For packages absent from the public index, the CLI can build a facade wheel
locally from the registry's source repository. This uses the same compatibility
loading path as remotely built wheels. It requires network access and build
dependencies; it does not make every repository installable.

## Existing custom_nodes directories

You can reuse a workspace's `custom_nodes/` directory:

```console
comfyui serve --cwd /path/to/ComfyUI
```

Install the node's documented requirements in the active environment. When a
node supplies a requirements file, for example:

```console
uv pip install -r /path/to/ComfyUI/custom_nodes/my-node/requirements.txt
```

The compatibility loader supports vanilla imports such as `nodes`, `folder_paths`,
and `server`. Nodes that require HTTP routes may need the server and cannot always
run through the embedded client.

## ComfyUI-Manager

Enable the Manager interface explicitly:

```console
comfyui serve --enable-manager
```

Manager maintains its own installation and enable/disable state. Use its UI for
nodes it manages, and the Python package installer for facade wheels. Workers
block runtime package installation by default; prepare their environments before
starting them.

## More information

- [Authoring installable nodes](custom_node_authoring.md)
- [Hosting a package index](package_index.md)
- [Programmatic custom node management](custom_node_reference.md)
- [Workflow and model discovery](workflows.md)

## Self-Hosting the Package Index

The hosting instructions are now in [Package index hosting](package_index.md).

## Programmatic Custom Node Management

The API and compatibility contracts are in the [management reference](custom_node_reference.md).

## Authoring Custom Nodes

See [Authoring installable nodes](custom_node_authoring.md).

## Adding Custom Configuration

See [Configuration hooks](custom_node_authoring.md#configuration-hooks).

## Model Discovery & Registration

See [Model discovery](workflows.md#models-and-existing-files) and
[registering downloads](embedded.md#adding-known-models-for-automatic-download).

## GGUF Native Support

Core loaders support GGUF for supported model architectures. A `.gguf` extension
alone does not establish compatibility with a loader, model family, or parallel
execution mode. Specialized custom-node loaders can still be needed for their
own supported architectures. See [quantization](../QUANTIZATION.md).

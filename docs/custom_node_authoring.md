# Authoring installable nodes

Publish a Python package with a `comfyui.custom_nodes` entry point to expose
your nodes to the packaged application. For unmodified ecosystem repositories,
the [package facade](package_index.md) provides a separate compatibility route.

## Package layout

```text
pyproject.toml
my_comfyui_nodes/
    __init__.py
    nodes.py
    configuration.py
```

Declare dependencies as package requirements, not installer command-line options:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "my-comfyui-nodes"
version = "0.1.0"
dependencies = ["comfyui"]

[project.entry-points."comfyui.custom_nodes"]
my_nodes = "my_comfyui_nodes"

[tool.setuptools.packages.find]
include = ["my_comfyui_nodes*"]
```

Install your project from its directory:

```console
uv pip install --extra-index-url https://nodes.appmana.com/simple/ -e .
```

If using `[tool.setuptools.dynamic]` to read a `requirements.txt`, that file must
contain dependency specifiers only. Keep `--extra-index-url` in the install command;
it is not valid wheel dependency metadata.

## Node mappings

Modules in the entry-point package are scanned recursively for node mappings.
For example, `nodes.py` can contain:

```python
class JoinText:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"left": ("STRING",), "right": ("STRING",)}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "join"
    CATEGORY = "examples"

    def join(self, left, right):
        return (left + right,)


NODE_CLASS_MAPPINGS = {"ExampleJoinText": JoinText}
NODE_DISPLAY_NAME_MAPPINGS = {"ExampleJoinText": "Join Text"}
```

`comfy.nodes.package_typing.CustomNode` supplies optional authoring type hints.
Use package imports such as `from comfy.cmd import folder_paths` in native
extensions. Keep node IDs stable because saved workflows reference them.

## Configuration hooks

Add a separate configuration entry point:

```toml
[project.entry-points."comfyui.custom_config"]
my_nodes = "my_comfyui_nodes.configuration:add_configuration"
```

In `configuration.py`:

```python
import configargparse


def add_configuration(parser: configargparse.ArgParser) -> configargparse.ArgParser:
    parser.add_argument("--my-node-endpoint", type=str, default=None)
    return parser
```

Pass the option explicitly:

```console
comfyui serve --my-node-endpoint=https://example.com
```

Custom configuration hooks use a separate parser. Their options do not appear in
Typer's `comfyui serve --help`; built-in configuration files are not a substitute
for passing custom-node options. Keep both the hook and the package's
`__init__.py` free of node imports so configuration is established before node
initialization. Node execution can then read `args.my_node_endpoint` from
`comfy.cli_args`.

Test loading in a fresh process and exercise the node in a workflow. See
[testing](testing.md) for embedded execution and fixtures, and the
[compatibility reference](custom_node_reference.md) when porting vanilla nodes.

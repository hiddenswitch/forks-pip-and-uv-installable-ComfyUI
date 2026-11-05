from __future__ import annotations

import ast
import json
import re
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from astroid import nodes
from pylint.checkers import BaseChecker

if TYPE_CHECKING:
    from pylint.lint import PyLinter


_ROOT = Path(str(files("tests"))).parent


def _rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _read(path: str) -> str:
    return (_ROOT / path).read_text(encoding="utf-8")


def _parse_python(path: str) -> ast.Module:
    return ast.parse(_read(path), filename=path)


class MergeHygieneChecker(BaseChecker):
    name = "merge-hygiene"
    msgs = {
        "W9101": (
            "Package version mismatch: pyproject.toml has %s but comfy/__init__.py has %s",
            "comfy-version-mismatch",
            "Keep pyproject.toml and comfy.__version__ synchronized before tagging a release.",
        ),
        "W9102": (
            "Module-level transformers import must go through comfy.transformers_compat or call patch_transformers_finegrained_fp8_import first: %s",
            "direct-transformers-import",
            "Avoid module-level direct transformers imports; transformers import chains can break older torch builds.",
        ),
        "W9103": (
            "Class %s accepts textmodel_json_config but does not forward it to SDClipModel initialization",
            "textmodel-json-config-not-forwarded",
            "Text encoder subclasses must pass textmodel_json_config through to the parent SDClipModel initializer.",
        ),
        "W9106": (
            "Blueprint package resource contract is broken: %s",
            "blueprint-package-resource-broken",
            "Blueprints must live under comfy.blueprints and be loaded with importlib.resources.",
        ),
        "W9107": (
            "Do not use dynamic importlib imports for import_all_nodes_in_workspace: %s",
            "dynamic-root-node-loader-import",
            "Import import_all_nodes_in_workspace directly and use a pylint disable comment if needed.",
        ),
        "W9108": (
            "PYTORCH_CUDA_ALLOC_CONF default handling is broken: %s",
            "cuda-alloc-conf-default-broken",
            "The default must set expandable_segments:True only when the environment value is unset or blank.",
        ),
        "W9109": (
            "Converted workflow cache still contains UI-format keys: %s",
            "workflow-cache-contains-ui-format",
            "Playwright workflow conversion caches must contain API-format prompts, not UI workflow/widget state.",
        ),
    }

    def __init__(self, linter: Optional["PyLinter"] = None) -> None:
        super().__init__(linter)
        self._source_cache: dict[str, str] = {}

    def visit_module(self, node: nodes.Module) -> None:
        if _rel(node.file) == "comfy/__init__.py":
            self._check_version_sync(node)
            self._check_blueprint_package_resources(node)
            self._check_cuda_alloc_conf_default(node)
            self._check_workflow_conversion_caches(node)

    def visit_import(self, node: nodes.Import) -> None:
        if not self._is_module_level(node):
            return
        for name, _alias in node.names:
            if name == "transformers" or name.startswith("transformers."):
                self._check_transformers_import(node, name)

    def visit_importfrom(self, node: nodes.ImportFrom) -> None:
        if not self._is_module_level(node):
            return
        if node.modname and (node.modname == "transformers" or node.modname.startswith("transformers.")):
            self._check_transformers_import(node, node.modname)

    def visit_classdef(self, node: nodes.ClassDef) -> None:
        if not self._inherits_sd_clip_model(node):
            return
        init_method = self._get_init_method(node)
        if init_method is None:
            return
        if "textmodel_json_config" not in self._argument_names(init_method):
            return
        if not self._forwards_textmodel_json_config(init_method):
            self.add_message("textmodel-json-config-not-forwarded", node=node, args=(node.name,))

    def visit_call(self, node: nodes.Call) -> None:
        import_target = self._dynamic_import_target(node)
        if import_target and (
            import_target == "comfy.nodes.package"
            or import_target == "comfy.nodes.package.import_all_nodes_in_workspace"
        ):
            self.add_message("dynamic-root-node-loader-import", node=node, args=(import_target,))

    def _source_for(self, node: nodes.NodeNG) -> str:
        path = _rel(node.root().file)
        if path not in self._source_cache:
            self._source_cache[path] = _read(path)
        return self._source_cache[path]

    def _is_module_level(self, node: nodes.NodeNG) -> bool:
        parent = node.parent
        while parent is not None and not isinstance(parent, nodes.Module):
            if isinstance(parent, (nodes.FunctionDef, nodes.AsyncFunctionDef, nodes.ClassDef, nodes.Lambda)):
                return False
            parent = parent.parent
        return parent is not None

    def _check_transformers_import(self, node: nodes.NodeNG, import_name: str) -> None:
        rel = _rel(node.root().file)
        if rel == "comfy/transformers_compat.py":
            return
        source_before = "\n".join(self._source_for(node).splitlines()[: max(node.fromlineno - 1, 0)])
        if "patch_transformers_finegrained_fp8_import()" in source_before:
            return
        self.add_message("direct-transformers-import", node=node, args=(import_name,))

    def _check_version_sync(self, node: nodes.Module) -> None:
        pyproject = _read("pyproject.toml")
        pyproject_match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject)
        init_match = re.search(r'(?m)^__version__\s*=\s*"([^"]+)"', _read("comfy/__init__.py"))
        pyproject_version = pyproject_match.group(1) if pyproject_match else "<missing>"
        init_version = init_match.group(1) if init_match else "<missing>"
        if pyproject_version != init_version:
            self.add_message("comfy-version-mismatch", node=node, args=(pyproject_version, init_version))

    def _check_blueprint_package_resources(self, node: nodes.Module) -> None:
        blueprints_dir = _ROOT / "comfy/blueprints"
        if not blueprints_dir.exists():
            return
        failures: list[str] = []
        if (_ROOT / "blueprints").exists():
            failures.append("root blueprints/ directory exists; move blueprints under comfy/blueprints")
        if not (blueprints_dir / "__init__.py").exists():
            failures.append("comfy/blueprints is missing __init__.py")
        subgraph_manager = _read("comfy/app/subgraph_manager.py")
        if 'BLUEPRINTS_PACKAGE = "comfy.blueprints"' not in subgraph_manager:
            failures.append("subgraph manager does not use the comfy.blueprints package constant")
        if "resources.files(BLUEPRINTS_PACKAGE)" not in subgraph_manager:
            failures.append("subgraph manager does not load blueprints with importlib.resources")
        if failures:
            self.add_message("blueprint-package-resource-broken", node=node, args=("; ".join(failures),))

    def _check_cuda_alloc_conf_default(self, node: nodes.Module) -> None:
        cuda_env = _read("comfy/component_model/cuda_env.py")
        failures: list[str] = []
        if '"expandable_segments:True"' not in cuda_env:
            failures.append("missing expandable_segments:True default")
        if "current is None or not current.strip()" not in cuda_env:
            failures.append("default is not limited to unset or blank environment values")
        if 'env["PYTORCH_CUDA_ALLOC_CONF"] = _DEFAULT_CUDA_ALLOC_CONF' not in cuda_env:
            failures.append("default is not written through _DEFAULT_CUDA_ALLOC_CONF")
        main_pre = _read("comfy/cmd/main_pre.py")
        if "ensure_pytorch_cuda_alloc_conf(skip_for_xpu=should_skip_cuda_alloc_conf_for_xpu())" not in main_pre:
            failures.append("main_pre does not apply the CUDA alloc default during startup")
        if failures:
            self.add_message("cuda-alloc-conf-default-broken", node=node, args=("; ".join(failures),))

    def _check_workflow_conversion_caches(self, node: nodes.Module) -> None:
        cache_dir = _ROOT / "tests/unit/playwright_cache"
        if not cache_dir.exists():
            return
        bad: list[str] = []
        for path in sorted(cache_dir.glob("*/*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict) or "__frontend_error__" in data:
                continue
            if "nodes" in data or "links" in data:
                bad.append(_rel(path))
                continue
            for prompt_node in data.values():
                if isinstance(prompt_node, dict) and "widgets_values" in prompt_node:
                    bad.append(_rel(path))
                    break
        if bad:
            sample = ", ".join(bad[:10])
            if len(bad) > 10:
                sample += f", ... (+{len(bad) - 10} more)"
            self.add_message("workflow-cache-contains-ui-format", node=node, args=(sample,))

    def _inherits_sd_clip_model(self, node: nodes.ClassDef) -> bool:
        for base in node.bases:
            if getattr(base, "name", "") == "SDClipModel":
                return True
            if getattr(base, "attrname", "") == "SDClipModel":
                return True
        return False

    def _get_init_method(self, node: nodes.ClassDef) -> nodes.FunctionDef | None:
        methods = node.locals.get("__init__", [])
        return methods[0] if methods and isinstance(methods[0], nodes.FunctionDef) else None

    def _argument_names(self, node: nodes.FunctionDef) -> set[str]:
        args = {arg.name for arg in node.args.args}
        args.update(arg.name for arg in node.args.kwonlyargs)
        return args

    def _forwards_textmodel_json_config(self, node: nodes.FunctionDef) -> bool:
        for call in node.nodes_of_class(nodes.Call):
            if not self._is_sd_clip_init_call(call):
                continue
            for keyword in call.keywords or []:
                if keyword.arg == "textmodel_json_config":
                    return True
        return False

    def _is_sd_clip_init_call(self, node: nodes.Call) -> bool:
        func = node.func
        if isinstance(func, nodes.Attribute) and func.attrname == "__init__":
            expr = func.expr
            if isinstance(expr, nodes.Call) and isinstance(expr.func, nodes.Name) and expr.func.name == "super":
                return True
            if isinstance(expr, nodes.Attribute) and expr.attrname == "SDClipModel":
                return True
            if isinstance(expr, nodes.Name) and expr.name == "SDClipModel":
                return True
        return False

    def _dynamic_import_target(self, node: nodes.Call) -> str | None:
        func = node.func
        is_importlib_import_module = (
            isinstance(func, nodes.Attribute)
            and func.attrname == "import_module"
            and isinstance(func.expr, nodes.Name)
            and func.expr.name == "importlib"
        )
        is_dunder_import = isinstance(func, nodes.Name) and func.name == "__import__"
        if not is_importlib_import_module and not is_dunder_import:
            return None
        if not node.args:
            return None
        first_arg = node.args[0]
        if isinstance(first_arg, nodes.Const) and isinstance(first_arg.value, str):
            return first_arg.value
        return None


def register(linter: "PyLinter") -> None:
    linter.register_checker(MergeHygieneChecker(linter))

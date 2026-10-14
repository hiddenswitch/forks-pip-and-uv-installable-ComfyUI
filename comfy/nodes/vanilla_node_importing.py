from __future__ import annotations

import fnmatch
import importlib
import importlib.util
import logging
import os
import sys
import time
import types
from contextlib import contextmanager, nullcontext
from functools import wraps
from os.path import join, basename, dirname, isdir, isfile, exists, abspath, split, splitext, realpath
from typing import Iterable, Any, Generator
from unittest.mock import patch

from comfy_compatibility.vanilla import prepare_vanilla_environment, patch_pip_install_subprocess_run, patch_pip_install_popen
from . import base_nodes
from ..app import governance
from .comfyui_v3_package_imports import _comfy_entrypoint_upstream_v3_imports
from .download_interception import (
    patch_hf_hub_download,
    patch_snapshot_download,
    patch_folder_paths_functions,
    patch_folder_names_dict,
    patch_torch_downloads,
)
from .package_typing import ExportedNodes
from .python_module_metadata import resolve_python_module_name
from ..cmd import folder_paths
from ..component_model.plugins import prompt_server_instance_routes
from ..distributed.server_stub import ServerStub
from ..execution_context import current_execution_context

logger = logging.getLogger(__name__)


# ast.Num, ast.Str, ast.Bytes, ast.NameConstant, and ast.Ellipsis were deprecated in
# Python 3.8 and removed in 3.14, along with the legacy `.n` and `.s` attributes on
# ast.Constant. Many custom nodes still use them (e.g. comfyui-custom-scripts'
# MathExpression). Restore them as aliases of ast.Constant so `isinstance(node, ast.Num)`
# and `node.n` continue to work on 3.14+.
if sys.version_info >= (3, 14):
    import ast as _ast
    for _legacy_name in ("Num", "Str", "Bytes", "NameConstant", "Ellipsis"):
        if not hasattr(_ast, _legacy_name):
            setattr(_ast, _legacy_name, _ast.Constant)
    if not hasattr(_ast.Constant, "n"):
        _ast.Constant.n = property(lambda self: self.value)
    if not hasattr(_ast.Constant, "s"):
        _ast.Constant.s = property(lambda self: self.value)
    del _ast, _legacy_name


def _stamp_relative_python_modules(node_class_mappings: dict[str, type]) -> None:
    for node_class in node_class_mappings.values():
        node_class.RELATIVE_PYTHON_MODULE = resolve_python_module_name(node_class)


class StreamToLogger:
    """
    File-like stream object that redirects writes to a logger instance.
    This is used to capture print() statements from modules during import.
    """

    def __init__(self, logger: logging.Logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.original_stdout = sys.stdout
        self._writing = False

    def write(self, buf):
        # A custom node may bind its own logging handler to this stream.
        if self._writing:
            self.original_stdout.write(buf)
            return
        self._writing = True
        try:
            for line in buf.rstrip().splitlines():
                self.logger.log(self.log_level, line.rstrip())
        finally:
            self._writing = False

    def flush(self):
        # The logger handles its own flushing, so this can be a no-op.
        pass

    @property
    def encoding(self):
        return "utf-8"


class _PromptQueueStub:
    """Stub for PromptServer.instance.prompt_queue used by VideoHelperSuite at import time."""

    def __init__(self):
        self.currently_running = {}

    def put(self, item):
        logger.warning("prompt_queue.put() called on stub — ignored")

    def get_current_queue(self):
        return [], []

    def get_current_queue_volatile(self):
        return [], []

    def get_history(self, **kwargs):
        return {}

    def size(self):
        return 0

    def get_tasks_remaining(self):
        return 0


class _PromptServerRouterStub:
    frozen = True


class _PromptServerAppStub:
    def __init__(self):
        self.router = _PromptServerRouterStub()

    def add_routes(self, routes):
        logger.debug("PromptServer.app.add_routes() called on stub with %s routes", len(routes or []))


class _PromptServerStub(ServerStub):
    def __init__(self):
        super().__init__()
        self.app = _PromptServerAppStub()
        self.routes = prompt_server_instance_routes
        self.on_prompt_handlers = []
        self.prompt_queue = _PromptQueueStub()
        self.number = 0
        self._warned_events: set[str] = set()

    def add_on_prompt_handler(self, handler):
        # todo: these need to be added to a real prompt server if the loading order is behaving in a complex way
        self.on_prompt_handlers.append(handler)

    def send_sync(self, *args, **kwargs):
        # Suppress repetitive monitor messages (e.g. crystools.monitor fires every second)
        event_type = args[0] if args else None
        if event_type and event_type in self._warned_events:
            return
        if event_type:
            self._warned_events.add(event_type)
        logger.warning(f"Node tried to send a message over the websocket while importing, args={args} kwargs={kwargs}")


def _vanilla_load_importing_execute_prestartup_script(node_paths: Iterable[str]) -> None:
    def execute_script(script_path):
        module_name = splitext(script_path)[0]
        try:
            with _stdout_intercept(module_name):
                spec = governance.pack_module_spec(module_name, script_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            return True
        except Exception as e:
            logger.error(f"Failed to execute startup-script: {script_path}", exc_info=e)
        return False

    node_prestartup_times = []
    for custom_node_path in node_paths:
        # patched
        if not isdir(custom_node_path):
            continue
        # end patch
        possible_modules = os.listdir(custom_node_path)

        for possible_module in possible_modules:
            module_path = join(custom_node_path, possible_module)
            if isfile(module_path) or module_path.endswith(".disabled") or module_path == "__pycache__":
                continue

            refusal = governance.pack_refusal(module_path)
            if refusal is not None:
                logger.warning(refusal)
                continue

            # Check if manager policy blocks this node
            from ..manager_integration import should_be_disabled
            if should_be_disabled(module_path):
                logger.info(f"Blocked by manager policy: {module_path}")
                continue

            script_path = join(module_path, "prestartup_script.py")
            if exists(script_path):
                if "comfyui-manager" in module_path.lower():
                    os.environ['COMFYUI_PATH'] = str(folder_paths.base_path)
                    os.environ['COMFYUI_FOLDERS_BASE_PATH'] = str(folder_paths.models_dir)
                    # Monkey-patch ComfyUI-Manager's security check to prevent it from crashing on startup
                    # and its logging handler to prevent it from taking over logging.
                    glob_path = join(module_path, "glob")
                    glob_path_added = False
                    original_add_handler = logging.Logger.addHandler

                    def no_op_add_handler(self, handler):
                        logger.info(f"Skipping addHandler for {type(handler).__name__} during ComfyUI-Manager prestartup.")

                    try:
                        sys.path.insert(0, glob_path)
                        glob_path_added = True
                        # Patch security_check
                        import security_check
                        original_check = security_check.security_check

                        def patched_security_check():
                            try:
                                return original_check()
                            except Exception as e:
                                logger.error(f"ComfyUI-Manager security_check failed but was caught gracefully: {e}", exc_info=e)

                        security_check.security_check = patched_security_check
                        logger.debug("Patched ComfyUI-Manager's security_check to fail gracefully.")

                        # Patch logging
                        logging.Logger.addHandler = no_op_add_handler
                        logger.debug("Patched logging.Logger.addHandler to prevent ComfyUI-Manager from adding a logging handler.")

                        time_before = time.perf_counter()
                        success = execute_script(script_path)
                        node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
                    except Exception as e:
                        logger.error(f"Failed to patch and execute ComfyUI-Manager's prestartup script: {e}", exc_info=e)
                    finally:
                        if glob_path_added and glob_path in sys.path:
                            sys.path.remove(glob_path)
                        logging.Logger.addHandler = original_add_handler
                else:
                    time_before = time.perf_counter()
                    success = execute_script(script_path)
                    node_prestartup_times.append((time.perf_counter() - time_before, module_path, success))
    if len(node_prestartup_times) > 0:
        logger.info("\nPrestartup times for custom nodes:")
        for n in sorted(node_prestartup_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (PRESTARTUP FAILED)"
            logger.info("{:6.1f} seconds{}: {}".format(n[0], import_message, n[1]))
        logger.info("")


_MITIGATED_MODULES = frozenset((
    "comfyui-manager",
    "comfyui_ryanonyheinside",
    "comfyui-easy-use",
    "comfyui_custom_nodes_alekpet",
))

# Map short model folder names to HuggingFace repo IDs.  Custom nodes often
# pass ``os.path.join(folder_paths.models_dir, name)`` to ``from_pretrained``
# which fails when the model isn't pre-downloaded.  We patch the call site to
# resolve to the HF repo ID so ``from_pretrained`` downloads from HuggingFace.
_MODEL_NAME_TO_HF_REPO: dict[str, str] = {
    "segformer_b2_clothes": "mattmdjaga/segformer_b2_clothes",
    "segformer_b3_clothes": "mattmdjaga/segformer_b3_clothes",
    "segformer_b3_fashion": "mattmdjaga/segformer_b3_fashion",
}



def _apply_post_import_patches(module_name: str) -> None:
    """Apply patches to custom-node submodules after they finish importing.

    Adapt dependency API moves and resolve custom-node model paths through
    the shared download infrastructure.
    """
    _patch_essentials_pixeloe(module_name)
    _patch_segformer_model_resolution(module_name)
    _install_deferred_controlnet_patches(module_name)
    if module_name.lower() == "comfyui-hunyuanimage3":
        from ..tensor_parallel.hunyuan_image3 import install

        install(sys.modules[module_name])
    if module_name.lower() == "kandinsky-6-sr":
        from ..model_downloader import get_filename_list_with_downloadable

        nodes = sys.modules[module_name + ".kandinsky6_vsr.nodes"]

        def local_model_files():
            # SR discovers component types from adjacent config files. Remote
            # catalog entries must not trigger downloads during this scan.
            names = set()
            for folder in (nodes.SR_MODEL_FOLDER, "diffusion_models"):
                try:
                    names.update(get_filename_list_with_downloadable(folder, known_files=[]))
                except (KeyError, OSError):
                    continue
            return sorted(names)

        nodes._model_files = local_model_files


def _patch_essentials_pixeloe(module_name: str) -> None:
    if module_name.lower() != "comfyui_essentials":
        return
    node = sys.modules[module_name].NODE_CLASS_MAPPINGS.get("PixelOEPixelize+")
    if node is None:
        return
    original_execute = node.execute

    @wraps(original_execute)
    def execute(self, *args, **kwargs):
        # Essentials still imports the original NumPy implementation from its
        # old path. PixelOE now ships that same API under legacy.pixelize.
        try:
            importlib.import_module("pixeloe.pixelize")
        except ModuleNotFoundError as exc:
            if exc.name != "pixeloe.pixelize":
                raise
            sys.modules["pixeloe.pixelize"] = importlib.import_module("pixeloe.legacy.pixelize")
        return original_execute(self, *args, **kwargs)

    node.execute = execute


def _patch_segformer_model_resolution(module_name: str) -> None:
    """Patch segformer_ultra.get_segmentation to resolve model names to HF repos.

    ComfyUI_LayerStyle's segformer_ultra builds a local path like
    ``models/segformer_b2_clothes`` and passes it to ``from_pretrained``.
    When the folder doesn't exist, we resolve the HuggingFace repo via
    ``model_downloader`` so the model is fetched to the HF cache and
    ``from_pretrained`` loads from the cached snapshot path.
    """
    if module_name.lower() != "comfyui_layerstyle":
        return

    for mod_name, mod in list(sys.modules.items()):
        if "segformer_ultra" not in mod_name or not hasattr(mod, "get_segmentation"):
            continue

        _orig_get_seg = mod.get_segmentation

        def _patched_get_seg(tensor_image, model_name='segformer_b2_clothes', _orig=_orig_get_seg):
            hf_repo = _MODEL_NAME_TO_HF_REPO.get(model_name)
            if hf_repo is not None:
                # Check if the model already exists at the local path the node expects
                try:
                    resolved = os.path.normpath(folder_paths.folder_names_and_paths[model_name][0][0])
                except Exception:
                    resolved = os.path.join(folder_paths.models_dir, model_name)
                if not os.path.isdir(resolved):
                    # Use the project's download infrastructure to get the repo
                    # into the HF cache; from_pretrained accepts the repo ID directly
                    from .. import model_downloader
                    logger.info("Model folder %s not found, resolving via model_downloader for %s", model_name, hf_repo)
                    model_downloader.get_or_download_huggingface_repo(hf_repo)
                    # Pass the HF repo ID so from_pretrained fetches from cache
                    return _orig(tensor_image, hf_repo)
            return _orig(tensor_image, model_name)

        mod.get_segmentation = _patched_get_seg
        logger.info("Patched segformer_ultra.get_segmentation to resolve model names to HuggingFace repos")
        break


def _install_deferred_controlnet_patches(module_name: str) -> None:
    """Install patches for lazily-loaded controlnet-aux submodules.

    Controlnet-aux loads preprocessor implementations on demand. We install a
    single ``builtins.__import__`` hook that applies all pending patches as
    their target modules are imported.
    """
    util_mod = sys.modules.get("custom_controlnet_aux.util")
    if util_mod is None:
        return

    # --- Patch definitions keyed by module name ---
    # Each value is a callable(module) that applies the patch.
    pending: dict[str, list] = {}

    # DSINE: replace local custom_hf_download with the shared one from util
    if hasattr(util_mod, "custom_hf_download"):
        shared_hf = util_mod.custom_hf_download

        def _patch_dsine(mod):
            if hasattr(mod, "custom_hf_download") and mod.custom_hf_download is not shared_hf:
                mod.custom_hf_download = shared_hf
                logger.info("Patched DSINE custom_hf_download to use shared util version")

        pending.setdefault("custom_controlnet_aux.dsine", []).append(_patch_dsine)

    # Pipeline .to() patches: fix device placement for HuggingFace pipeline wrappers.
    # These detectors use transformers.pipeline() which auto-selects CUDA, but
    # their to() only moves self.pipe.model without updating self.pipe.device.
    _PIPELINE_CLASSES = {
        "custom_controlnet_aux.depth_anything": ("DepthAnythingDetector",),
        "custom_controlnet_aux.zoe": ("ZoeDetector", "ZoeDepthAnythingDetector"),
    }

    def _make_pipeline_patcher(class_names):
        def _patch(mod):
            for cls_name in class_names:
                cls = getattr(mod, cls_name, None)
                if cls is None or not hasattr(cls, "to"):
                    continue
                original_to = cls.to

                def _patched_to(self, device, _orig=original_to):
                    _orig(self, device)
                    if hasattr(self, "pipe"):
                        import torch
                        self.pipe.device = torch.device(device) if isinstance(device, str) else device
                        self.pipe.model = self.pipe.model.to(device)
                    return self

                cls.to = _patched_to
                logger.info("Patched %s.%s.to() for proper pipeline device placement", mod.__name__, cls_name)
        return _patch

    for mod_name, cls_names in _PIPELINE_CLASSES.items():
        pending.setdefault(mod_name, []).append(_make_pipeline_patcher(cls_names))

    # custom_torch_download: default ckpts_dir bakes in the site-packages path.
    # Replace it to download to torch hub cache instead.
    def _patch_torch_download(mod):
        if not hasattr(mod, "custom_torch_download"):
            return

        def _patched_torch_download(filename, ckpts_dir=None):
            import torch
            model_url = "https://download.pytorch.org/models/" + filename
            cache_dir = torch.hub.get_dir()
            local_dir = os.path.join(cache_dir, "checkpoints")
            os.makedirs(local_dir, exist_ok=True)
            model_path = os.path.join(local_dir, filename)
            if not os.path.exists(model_path):
                torch.hub.download_url_to_file(model_url, model_path, progress=True)
            return model_path

        mod.custom_torch_download = _patched_torch_download
        logger.info("Patched custom_torch_download to use torch hub cache")

    pending.setdefault("custom_controlnet_aux.util", []).append(_patch_torch_download)

    # DepthAnythingV2: image2tensor hardcodes CUDA device selection.
    # Patch it to use the model's device instead.
    def _patch_depth_anything_v2(mod):
        cls = getattr(mod, "DepthAnythingV2Detector", None)
        if cls is None:
            return
        original_call = cls.__call__

        def _patched_call(self, *args, **kwargs):
            # Temporarily patch the DPT model's image2tensor to use self.device
            model = self.model
            if hasattr(model, "image2tensor"):
                original_i2t = model.image2tensor

                def _patched_i2t(raw_image, input_size=518, _orig=original_i2t, _device=self.device):
                    image, hw = _orig(raw_image, input_size)
                    return image.to(_device), hw

                model.image2tensor = _patched_i2t
                try:
                    return original_call(self, *args, **kwargs)
                finally:
                    model.image2tensor = original_i2t
            return original_call(self, *args, **kwargs)

        cls.__call__ = _patched_call
        logger.info("Patched DepthAnythingV2Detector.__call__ for proper device placement")

    pending.setdefault("custom_controlnet_aux.depth_anything_v2", []).append(_patch_depth_anything_v2)

    # Apply patches for modules already loaded
    for mod_name in list(pending):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            for fn in pending.pop(mod_name):
                fn(mod)

    if not pending:
        return

    # Install a single __import__ hook for all remaining deferred patches
    import builtins

    # Walk the chain of any previously-installed _import_hook closures to find
    # the real built-in __import__, so re-entrant calls never loop back to us.
    original_import = builtins.__import__
    while getattr(original_import, '_is_deferred_patch_hook', False):
        original_import = original_import._original_import
    _in_hook = False

    def _import_hook(name, *args, **kwargs):
        nonlocal _in_hook
        if _in_hook:
            return original_import(name, *args, **kwargs)
        _in_hook = True
        try:
            result = original_import(name, *args, **kwargs)
            patches = pending.pop(name, None)
            if patches:
                mod = sys.modules.get(name)
                if mod is not None:
                    for fn in patches:
                        fn(mod)
                if not pending:
                    builtins.__import__ = original_import
            return result
        finally:
            _in_hook = False

    _import_hook._is_deferred_patch_hook = True
    _import_hook._original_import = original_import
    builtins.__import__ = _import_hook


@contextmanager
def _protect_sys_path():
    """Snapshot and restore sys.path after custom node import.

    Custom nodes often call ``sys.path.insert(0, ...)`` during import to make
    sibling modules importable.  We allow mutations during import (some nodes
    need them for intra-package imports) but restore the original path list
    afterward so that one node's path additions don't leak to later nodes.

    Before restoring, any directories the node added are scanned for Python
    packages not yet in ``sys.modules``.  Those packages are pre-registered
    with ``spec_from_file_location`` so that lazy imports at execution time
    still resolve (e.g. ``custom_mmpkg`` from comfyui_controlnet_aux).
    """
    snapshot = set(sys.path)
    snapshot_list = list(sys.path)
    try:
        yield
    finally:
        added = [p for p in sys.path if p not in snapshot]
        for directory in added:
            _register_packages_from_directory(directory)
        sys.path[:] = snapshot_list


def _register_packages_from_directory(directory: str) -> None:
    """Register importable packages found in *directory* into ``sys.modules``.

    Only creates the module spec and placeholder; actual execution is deferred
    to first attribute access via the spec loader.
    """
    try:
        entries = os.listdir(directory)
    except OSError:
        return
    for name in entries:
        if name.startswith((".", "_")):
            continue
        if name in sys.modules:
            continue
        candidate = join(directory, name)
        init_py = join(candidate, "__init__.py")
        if isdir(candidate) and isfile(init_py):
            spec = importlib.util.spec_from_file_location(
                name, init_py,
                submodule_search_locations=[candidate],
            )
            mod = importlib.util.module_from_spec(spec)
            mod.__path__ = [candidate]
            sys.modules[name] = mod


@contextmanager
def _prepare_pixeloe_import(module: types.ModuleType, module_path: str, block_installation: bool):
    # PixelOE vendors its library under src. Its obsolete pkg_resources
    # installer is unnecessary when that library is already bundled, including
    # normal CLI startup where runtime installation is allowed.
    sys.path.insert(0, join(module_path, "src"))
    if not block_installation and not isdir(join(module_path, "src")):
        yield
        return
    name = f"{module.__name__}.nodes.installer"
    installer = types.ModuleType(name)
    installer.install_pixeloe = lambda: None
    sys.modules[name] = installer
    try:
        yield
    finally:
        sys.modules.pop(name, None)


@contextmanager
def _exec_mitigations(module: types.ModuleType, module_path: str) -> Generator[ExportedNodes, Any, None]:
    config = current_execution_context()
    block_installation = config and config.configuration and config.configuration.block_runtime_package_installation

    needs_file_mitigation = module.__name__.lower() in _MITIGATED_MODULES

    with (
        # download interception — always active during custom node import
        patch_hf_hub_download(),
        patch_snapshot_download(),
        patch_folder_paths_functions(),
        patch_folder_names_dict(),
        patch_torch_downloads(),
        # pip blocking
        patch_pip_install_subprocess_run() if block_installation else nullcontext(),
        patch_pip_install_popen() if block_installation else nullcontext(),
        # sys.path protection — prevent custom nodes from polluting the path
        _protect_sys_path(),
        _prepare_pixeloe_import(module, module_path, block_installation) if basename(module_path).lower() == "pixeloe" else nullcontext(),
    ):
        if needs_file_mitigation:
            from ..cmd import folder_paths
            old_file = folder_paths.__file__
            try:
                new_path = join(abspath(join(dirname(old_file), "..", "..")), basename(old_file))
                with patch.object(folder_paths, "__file__", new_path):
                    yield ExportedNodes()
            finally:
                logger.info(f"Exec mitigations were applied for {module.__name__}, due to using the folder_paths.__file__ symbol and manipulating EXTENSION_WEB_DIRS")
        else:
            yield ExportedNodes()


@contextmanager
def _stdout_intercept(name: str):
    original_stdout = sys.stdout

    try:
        module_logger = logging.getLogger(name)
        sys.stdout = StreamToLogger(module_logger, logging.INFO)
        yield
    finally:
        sys.stdout = original_stdout


def _vanilla_load_custom_nodes_1(module_path, ignore: set = None) -> ExportedNodes:
    if ignore is None:
        ignore = set()
    exported_nodes = ExportedNodes()
    module_name = basename(module_path)
    if isfile(module_path):
        sp = splitext(module_path)
        module_name = sp[0]
    try:
        if isfile(module_path):
            module_spec = governance.pack_module_spec(module_name, module_path)
            module_dir = split(module_path)[0]
        else:
            module_spec = governance.pack_module_spec(module_name, join(module_path, "__init__.py"))
            module_dir = module_path

        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module

        if not isfile(module_path):
            _stale_prefix = module_name + "."
            for _key in [k for k in sys.modules if k.startswith(_stale_prefix)]:
                del sys.modules[_key]
            _mod_name = module_name

            def _submodule_getattr(name, _prefix=_mod_name):
                fullname = f'{_prefix}.{name}'
                sub = sys.modules.get(fullname)
                if sub is not None:
                    return sub
                raise AttributeError(f"module {_prefix!r} has no attribute {name!r}")

            module.__getattr__ = _submodule_getattr

        with _exec_mitigations(module, module_path) as mitigated_exported_nodes, _stdout_intercept(module_name):
            module_spec.loader.exec_module(module)
            exported_nodes.update(mitigated_exported_nodes)

        _apply_post_import_patches(module_name)

        if hasattr(module, "WEB_DIRECTORY") and getattr(module, "WEB_DIRECTORY") is not None:
            web_dir = abspath(join(module_dir, getattr(module, "WEB_DIRECTORY")))
            if isdir(web_dir):
                exported_nodes.EXTENSION_WEB_DIRS[module_name] = web_dir

        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            _stamp_relative_python_modules(module.NODE_CLASS_MAPPINGS)
            for name in module.NODE_CLASS_MAPPINGS:
                if name not in ignore:
                    exported_nodes.NODE_CLASS_MAPPINGS[name] = module.NODE_CLASS_MAPPINGS[name]
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module,
                                                                         "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                for name, display_name in module.NODE_DISPLAY_NAME_MAPPINGS.items():
                    if name not in ignore:
                        exported_nodes.NODE_DISPLAY_NAME_MAPPINGS[name] = display_name
        elif not callable(getattr(module, "comfy_entrypoint", None)):
            logger.error(f"Skip {module_path} module for custom nodes: no NODE_CLASS_MAPPINGS or callable comfy_entrypoint.")

        exported_nodes.update(_comfy_entrypoint_upstream_v3_imports(module, ignore=ignore))
    except Exception as e:
        logger.error(f"Cannot import {module_path} module for custom nodes:", exc_info=e)
    return exported_nodes


def _vanilla_load_custom_nodes_2(node_paths: Iterable[str]) -> ExportedNodes:
    from ..cli_args import args
    base_node_names = set(base_nodes.NODE_CLASS_MAPPINGS.keys())
    node_import_times = []
    exported_nodes = ExportedNodes()
    for custom_node_path in node_paths:
        if not exists(custom_node_path) or not isdir(custom_node_path):
            continue
        possible_modules = os.listdir(realpath(custom_node_path))
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")

        for possible_module in possible_modules:
            module_path = join(custom_node_path, possible_module)
            if isfile(module_path) and splitext(module_path)[1] != ".py": continue
            if module_path.endswith(".disabled"): continue
            if args.disable_all_custom_nodes and possible_module not in args.whitelist_custom_nodes:
                logger.info(f"Skipping {possible_module} due to disable_all_custom_nodes and whitelist_custom_nodes")
                continue
            if any(fnmatch.fnmatch(possible_module, pattern) for pattern in args.blacklist_custom_nodes):
                logger.info(f"Skipping {possible_module} due to blacklist_custom_nodes")
                continue
            # Check if manager policy blocks this node
            from ..manager_integration import should_be_disabled
            if should_be_disabled(module_path):
                logger.info(f"Blocked by manager policy: {module_path}")
                continue
            refusal = governance.pack_refusal(module_path)
            if refusal is not None:
                logger.warning(refusal)
                continue
            time_before = time.perf_counter()
            possible_exported_nodes = _vanilla_load_custom_nodes_1(module_path, ignore=base_node_names)
            # comfyui-manager mitigation
            import_succeeded = len(possible_exported_nodes.NODE_CLASS_MAPPINGS) > 0 or "ComfyUI-Manager" in module_path
            node_import_times.append(
                (time.perf_counter() - time_before, module_path, import_succeeded))
            exported_nodes.update(possible_exported_nodes)

    if len(node_import_times) > 0:
        for n in sorted(node_import_times):
            if n[2]:
                import_message = ""
            else:
                import_message = " (IMPORT FAILED)"
            logger.debug(f"{n[0]:6.1f} seconds{import_message}: {n[1]}")
    return exported_nodes


def mitigated_import_of_vanilla_custom_nodes(extra_node_paths: Iterable[str] = ()) -> ExportedNodes:
    # only vanilla custom nodes will ever go into the custom_nodes directory
    # this mitigation puts files that custom nodes expects are at the root of the repository back where they should be
    # found. we're in the middle of executing the import of execution and server, in all likelihood, so like all things,
    # the way community custom nodes is pretty radioactive
    # there's a lot of subtle details here, and unfortunately, once this is called, there are some things that have
    # to be activated later, in different places, to make all the hacks necessary for custom nodes to work
    prepare_vanilla_environment()

    from ..cmd import folder_paths
    node_paths = list(folder_paths.get_folder_paths("custom_nodes"))
    node_paths.extend(str(path) for path in extra_node_paths)

    potential_git_dir_parent = join(dirname(__file__), "..", "..")
    is_git_repository = exists(join(potential_git_dir_parent, ".git"))
    if is_git_repository:
        node_paths += [abspath(join(potential_git_dir_parent, "custom_nodes"))]

    node_paths = frozenset(abspath(custom_node_path) for custom_node_path in node_paths)
    _vanilla_load_importing_execute_prestartup_script(node_paths)
    vanilla_custom_nodes = _vanilla_load_custom_nodes_2(node_paths)

    try:
        from comfy_compatibility.vhs_fsspec import apply_vhs_fsspec_patch
        apply_vhs_fsspec_patch()
    except Exception as exc:  # noqa: BLE001
        logger.debug("VHS fsspec patch skipped: %s", exc)

    return vanilla_custom_nodes

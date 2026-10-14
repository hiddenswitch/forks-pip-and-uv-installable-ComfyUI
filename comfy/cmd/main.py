from .main_pre import tracer

import asyncio
import gc
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from ..component_model.abstract_prompt_queue import AbstractPromptQueue
from . import hook_breaker_ac10a0
from .extra_model_paths import load_extra_path_config
from .. import model_management
from ..analytics.analytics import initialize_event_tracking
from ..cli_args_types import Configuration
from . import cuda_malloc
from . import folder_paths
from . import server as server_module
from ..component_model.entrypoints_common import configure_application_paths
from ..component_model.file_counter import cleanup_temp as fc_cleanup_temp
from ..distributed.distributed_prompt_queue import DistributedPromptQueue
from ..distributed.server_stub import ServerStub
from ..execution_context import current_execution_context
from ..nodes.package import import_all_nodes_in_workspace
from ..nodes_context import get_nodes
from ..app.assets.manager import AssetManager, default_asset_manager
from ..app.assets.event_log import error_kind
from ..app.database.db import get_database_url, get_db_path, init_db

logger = logging.getLogger(__name__)
_shutdown_event = threading.Event()


def cuda_malloc_warning():
    device = model_management.get_torch_device()
    device_name = model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logger.warning(
                "\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")


def vram_performance_warnings():
    import psutil
    import torch

    args = current_execution_context().configuration
    total_vram_mb = model_management.total_vram

    if total_vram_mb <= 16384 and not args.novram:
        logger.warning(
            "You have %.0f MB of VRAM. For best performance with 16GB or less, start ComfyUI with --novram. "
            "This aggressively offloads models without meaningfully impacting inference speed.",
            total_vram_mb
        )

    if total_vram_mb <= 16384 and not args.disable_pinned_memory:
        swap = psutil.swap_memory()
        physical_ram = psutil.virtual_memory().total
        if swap.total > 0 and physical_ram < 32 * 1024 * 1024 * 1024:
            logger.warning(
                "Swap is enabled with less than 32GB of physical RAM and 16GB or less of VRAM. "
                "Consider starting ComfyUI with --disable-pinned-memory to avoid swap thrashing."
            )

    if torch.cuda.is_available() and model_management.is_nvidia():
        from ..cli_args import PerformanceFeature
        device = model_management.get_torch_device()
        if hasattr(device, 'type') and device.type == 'cuda':
            props = torch.cuda.get_device_properties(device)
            if props.major >= 8 and PerformanceFeature.CublasOps not in args.fast:
                logger.warning(
                    "You have an Ampere or newer GPU (%s). Consider starting ComfyUI with --fast cublas_ops for improved performance.",
                    props.name
                )


def handle_comfyui_manager_unavailable(args: Configuration):
    if not args.windows_standalone_build:
        logger.warning(f"\n\nYou appear to be running comfyui-manager from source, this is not recommended. Please install comfyui-manager using the following command:\ncommand:\n\t{sys.executable} -m pip install --pre comfyui_manager\n")
    args.enable_manager = False


def _collect_output_absolute_paths(history_result: dict) -> list[str]:
    """Extract absolute file paths for output items from a history result."""
    paths: list[str] = []
    seen: set[str] = set()
    for node_output in history_result.get("outputs", {}).values():
        for items in node_output.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type not in ("output", "temp"):
                    continue
                base_dir = folder_paths.get_directory_by_type(item_type)
                if base_dir is None:
                    continue
                base_dir = os.path.abspath(base_dir)
                filename = item.get("filename")
                if not filename:
                    continue
                abs_path = os.path.abspath(
                    os.path.join(base_dir, item.get("subfolder", ""), filename)
                )
                if not abs_path.startswith(base_dir + os.sep) and abs_path != base_dir:
                    continue
                if abs_path not in seen:
                    seen.add(abs_path)
                    paths.append(abs_path)
    return paths


async def _prompt_worker(q: AbstractPromptQueue, server_instance: server_module.PromptServer, asset_manager: AssetManager | None = None):
    from . import execution
    from ..component_model import queue_types
    from .. import model_management

    args = current_execution_context().configuration
    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_ram != 0:
        cache_type = execution.CacheType.RAM_PRESSURE
    elif args.cache_none:
        cache_type = execution.CacheType.NONE

    cache_ram = args.cache_ram
    if cache_ram < 0:
        cache_ram = min(32.0, max(4.0, model_management.total_ram * 0.25 / 1024.0))

    asset_manager = asset_manager if asset_manager is not None else default_asset_manager()
    e = execution.PromptExecutor(server_instance, cache_type=cache_type, cache_args={"lru": args.cache_lru, "ram": cache_ram}, asset_manager=asset_manager)
    last_gc_collect = 0
    need_gc = False
    gc_collect_interval = 10.0
    current_time = 0.0
    background_scan_paused = False
    try:
        while not _shutdown_event.is_set():
            timeout = 1.0
            if need_gc:
                timeout = max(min(gc_collect_interval - (current_time - last_gc_collect), 1.0), 0.0)

            queue_item = q.get(timeout=timeout)
            if queue_item is not None:
                item, item_id = queue_item
                execution_start_time = time.perf_counter()
                prompt_id = item[1]
                server_instance.last_prompt_id = prompt_id
                server_instance.workflow_metadata = item[3].get("workflow_metadata", {})

                sensitive = item[5]
                extra_data = item[3].copy()
                for k in sensitive:
                    extra_data[k] = sensitive[k]

                # todo: ??? what jank
                remove_sensitive = lambda prompt: prompt[:5] + prompt[6:]

                asset_manager.pause_background_scan()
                background_scan_paused = True
                await e.execute_async(item[2], prompt_id, item[3], item[4])
                need_gc = True

                error_details = None
                if not e.success:
                    for event, data in e.status_messages:
                        if event == "execution_error":
                            error_details = data
                            break

                messages = [f"{event}: {data.get('exception_message', str(data))}" if isinstance(data, dict) and 'exception_message' in data else f"{event}" for event, data in e.status_messages]

                q.task_done(item_id,
                            e.history_result,
                            status=queue_types.ExecutionStatus(
                                status_str='success' if e.success else 'error',
                                completed=e.success,
                                messages=messages),
                            error_details=error_details,
                            process_item=remove_sensitive,
                            )
                if q.get_tasks_remaining() == 0:
                    # Nothing else is queued, so let the background scan run now rather than on the next GC tick.
                    asset_manager.resume_background_scan()
                    background_scan_paused = False

                if server_instance.client_id is not None:
                    server_instance.send_sync("executing", {"node": None, "prompt_id": prompt_id}, server_instance.client_id)
                server_instance.workflow_metadata = {}

                current_time = time.perf_counter()
                execution_time = current_time - execution_start_time

                # Log Time in a more readable way after 10 minutes
                if execution_time > 600:
                    execution_time = time.strftime("%H:%M:%S", time.gmtime(execution_time))
                    logger.info(f"Prompt executed in {execution_time}")
                else:
                    logger.info("Prompt executed in {:.2f} seconds".format(execution_time))


            flags = q.get_flags()
            free_memory = flags.get("free_memory", False)

            if flags.get("unload_models", free_memory):
                model_management.unload_all_models()
                need_gc = True
                last_gc_collect = 0

            if free_memory:
                e.reset()
                need_gc = True
                last_gc_collect = 0

            if need_gc:
                current_time = time.perf_counter()
                if (current_time - last_gc_collect) > gc_collect_interval:
                    gc.collect()
                    model_management.soft_empty_cache()
                    last_gc_collect = current_time
                    need_gc = False
                    hook_breaker_ac10a0.restore_functions()

                    asset_manager.queue_output_scan()
                    asset_manager.resume_background_scan()
                    background_scan_paused = False
    finally:
        if background_scan_paused:
            try:
                asset_manager.resume_background_scan()
            except Exception:
                logger.exception("Failed to resume background asset scanning after prompt worker exit")


def prompt_worker(q: AbstractPromptQueue, server_instance: server_module.PromptServer, asset_manager: AssetManager | None = None):
    asyncio.run(_prompt_worker(q, server_instance, asset_manager))


async def run(server_instance, address='', port=8188, call_on_start=None):
    addresses = []
    for addr in address.split(","):
        addresses.append((addr, port))
    await asyncio.gather(server_instance.start_multi_address(addresses, call_on_start), server_instance.publish_loop())


def cleanup_temp():
    try:
        folder_paths.get_temp_directory()
        temp_dir = folder_paths.get_temp_directory()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except NameError:
        # __file__ was not defined
        pass


def start_comfyui(asyncio_loop: asyncio.AbstractEventLoop = None):
    asyncio_loop = asyncio_loop or asyncio.get_event_loop()
    asyncio_loop.run_until_complete(_start_comfyui())


def setup_database(asset_manager: AssetManager | None = None):
    asset_manager = asset_manager if asset_manager is not None else default_asset_manager()
    if not asset_manager.enabled:
        asset_manager.startup()
        return
    try:
        init_db()
        asset_manager.startup()
    except Exception as error:
        failure, message = database_failure_message(error, get_database_url())
        if failure != "unsupported_url":
            logger.debug("Asset database startup failed", exc_info=True)
        stop_startup(failure, message)


WITHOUT_ASSETS = "Or start ComfyUI without the assets system: --disable-assets"


def stop_startup(kind, message):
    """Exit after a line a launcher can match, then the message for the user."""
    logging.error(f"ASSETS_STARTUP_FAILED: {kind}\n{message}")
    sys.exit(1)


def database_failure_message(error, db_url):
    """The kind of failure that stopped the asset database from opening, and how to fix it."""
    if not (db_url.startswith("sqlite:///") or db_url == "sqlite://"):
        return "unsupported_url", ("--database-url must start with sqlite:///, like sqlite:///path/to/comfyui.db, "
                                   f"or be left out to use the default database.\n{WITHOUT_ASSETS}")
    location = get_db_path() if db_url.startswith("sqlite:///") else db_url
    kind = error_kind(error)
    detail = getattr(error, "orig", None) or error
    if "Could not acquire lock on database" in str(error):
        failure = "in_use"
        what = f"Another ComfyUI is already using this database: '{location}'."
        fix = "Close the other ComfyUI and start this one again."
    elif kind in ("database_locked", "file_locked"):
        failure = "locked"
        what = f"The asset database '{location}' is locked by another program ({detail})."
        fix = "Close any program that has it open, such as another ComfyUI or a database viewer, and start again."
    elif "Can't locate revision" in str(error):
        failure = "newer_revision"
        what = f"The asset database '{location}' was last used by a newer version of ComfyUI ({detail})."
        fix = "Update ComfyUI, or move that file aside and start again to create a new database."
    elif isinstance(error, (FileExistsError, NotADirectoryError)):
        failure = "path_blocked"
        what = f"A file is in the way of the folder for the asset database '{location}' ({detail})."
        fix = "Move that file, or choose another folder."
    elif kind in ("read_only", "unable_to_open") or isinstance(error, OSError):
        failure = "not_writable"
        what = f"ComfyUI can't create, open or write the asset database '{location}' ({detail})."
        fix = ("Make sure its folder is a writable directory, the database path is a writable file (or doesn't exist yet), "
               "and no other program has it open.")
    elif kind == "database_corrupt":
        failure = "corrupt"
        what = f"The asset database '{location}' is corrupt ({detail})."
        fix = ("Move that file aside, or delete it, and start again: ComfyUI creates a new database "
               "and rebuilds the asset catalog by rescanning your files.")
    else:
        failure = "other"
        what = f"Could not open or upgrade the asset database '{location}': {detail}"
        fix = ("If the database is damaged, move that file aside and start again: ComfyUI creates a new "
               "database and rebuilds the asset catalog by rescanning your files. Run with --verbose DEBUG for the full error.")
    lines = [what, fix]
    if failure in ("in_use", "locked") and current_execution_context().configuration.database_url is None:
        lines.append("Or give this ComfyUI its own database: --database-url sqlite:///path/to/another.db")
    return failure, "\n".join(lines + [WITHOUT_ASSETS])



async def _start_comfyui(from_script_dir: Optional[Path] = None, configuration: Optional[Configuration] = None):
    from ..execution_context import context_configuration
    from ..cli_args import cli_args_configuration
    configuration = configuration or cli_args_configuration()
    with (
        context_configuration(configuration),
        fc_cleanup_temp()
    ):
        await __start_comfyui(from_script_dir=from_script_dir)


@tracer.start_as_current_span("Start ComfyUI")
async def __start_comfyui(from_script_dir: Optional[Path] = None):
    """
    Runs ComfyUI's frontend and backend like upstream.
    :param from_script_dir: when set to a path, assumes that you are running ComfyUI's legacy main.py entrypoint at the root of the git repository located at the path
    """
    args = current_execution_context().configuration
    if args.enable_asset_hashing and (args.disable_assets or not (args.enable_assets or args.windows_standalone_build)):
        logger.warning("--enable-asset-hashing is ignored because the assets system is off. Start ComfyUI with --enable-assets, and without --disable-assets, to use it.")
    if not from_script_dir:
        os_getcwd = os.getcwd()
    else:
        os_getcwd = str(from_script_dir)

    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logger.debug(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)

    if args.user_directory:
        user_dir = os.path.abspath(args.user_directory)
        logger.info(f"Setting user directory to: {user_dir}")
        folder_paths.set_user_directory(user_dir)

    # A governed build applies its signed policy before Manager is imported: Manager's prestartup runs scheduled installs
    # before any pack is checked, so the policy may turn it off, and a policy that cannot be applied stops startup first.
    from ..app import governance
    governance.initialize(args)

    # Initialize comfyui_manager if available and enabled
    from ..manager_integration import init_manager, prestartup as manager_prestartup, start as manager_start
    init_manager(args)
    manager_prestartup()

    # Initialize manager model cache for model fallback/validation
    from ..manager_model_cache import init_manager_model_cache
    init_manager_model_cache(refresh_from_github=args.refresh_manager_models)

    # Initialize Civitai live-search fallback for model lookup
    from .. import civitai_model_cache
    civitai_model_cache.init_civitai_model_cache()

    # configure extra model paths earlier
    try:
        extra_model_paths_config_path = os.path.join(os_getcwd, "extra_model_paths.yaml")
        if os.path.isfile(extra_model_paths_config_path):
            load_extra_path_config(extra_model_paths_config_path)
    except NameError:
        pass

    if args.extra_model_paths_config:
        for config_path in args.extra_model_paths_config:
            load_extra_path_config(config_path)

    if args.create_directories:
        # then, import and exit
        import_all_nodes_in_workspace(raise_on_failure=False)
        folder_paths.create_directories()
        exit(0)

    if args.windows_standalone_build:
        try:
            from . import new_updater
            new_updater.update_windows_updater()
        except:
            pass

    loop = asyncio.get_event_loop()
    server = server_module.PromptServer(loop)

    if args.enable_manager and not args.disable_manager_ui:
        folder_paths.create_directories()
        manager_start()

    if args.external_address is not None:
        server.external_address = args.external_address

    # at this stage, it's safe to import nodes
    hook_breaker_ac10a0.save_functions()
    if args.quick_test_for_ci:
        import_all_nodes_in_workspace(raise_on_failure=True)
    nodes_to_import = get_nodes()
    logger.debug(f"Imported {len(nodes_to_import)} nodes")
    server.nodes = nodes_to_import
    hook_breaker_ac10a0.restore_functions()
    # as a side effect, this also populates the nodes for execution

    if args.distributed_queue_connection_uri is not None:
        distributed = True
        q = DistributedPromptQueue(
            caller_server=server if args.distributed_queue_frontend else None,
            connection_uri=args.distributed_queue_connection_uri,
            is_caller=args.distributed_queue_frontend,
            is_callee=args.distributed_queue_worker,
            loop=loop,
            queue_name=args.distributed_queue_name
        )
        await q.init()
    else:
        distributed = False
        from .execution import PromptQueue
        q = PromptQueue(server)
    server.prompt_queue = q

    server.add_routes()
    cuda_malloc_warning()
    vram_performance_warnings()
    setup_database(server.asset_manager)
    if args.quick_test_for_ci:
        return

    # in a distributed setting, the default prompt worker will not be able to send execution events via the websocket
    _prompt_worker_executor = None
    worker_thread_server = server if not distributed else ServerStub()
    if not distributed or args.distributed_queue_worker:
        if distributed:
            logger.warning(
                f"Distributed workers started in the default thread loop cannot notify clients of progress updates. Instead of comfyui or main.py, use comfyui-worker.")
        from ..distributed.executors import ContextVarExecutor
        _prompt_worker_executor = ContextVarExecutor(max_workers=1, thread_name_prefix="PromptWorker")
        _prompt_worker_executor.submit(prompt_worker, q, worker_thread_server, server.asset_manager)

    # server has been imported and things should be looking good
    initialize_event_tracking(loop)

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        logger.debug(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    # These are the default folders that checkpoints, clip and vae models will be saved to when using CheckpointSave, etc.. nodes
    folder_paths.add_model_folder_path("checkpoints", os.path.join(folder_paths.get_output_directory(), "checkpoints"))
    folder_paths.add_model_folder_path("clip", os.path.join(folder_paths.get_output_directory(), "clip"))
    folder_paths.add_model_folder_path("vae", os.path.join(folder_paths.get_output_directory(), "vae"))
    folder_paths.add_model_folder_path("diffusion_models",
                                       os.path.join(folder_paths.get_output_directory(), "diffusion_models"))
    folder_paths.add_model_folder_path("loras", os.path.join(folder_paths.get_output_directory(), "loras"))

    if args.input_directory:
        input_dir = os.path.abspath(args.input_directory)
        logger.debug(f"Setting input directory to: {input_dir}")
        folder_paths.set_input_directory(input_dir)

    folder_paths.create_directories()

    if len(args.workflows) > 0:
        configure_application_paths(args)
        from ..entrypoints.workflow import run_workflows
        await run_workflows(args.workflows)
        return

    # replaced my folder_paths.create_directories
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme="http", address="localhost", port=8188):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0' or address == '':
                address = '127.0.0.1'
            if ':' in address:
                address = "[{}]".format(address)
            webbrowser.open(f"{scheme}://{address}:{port}")

        call_on_start = startup_server

    first_listen_addr = args.listen.split(',')[0] if ',' in args.listen else args.listen
    server.address = first_listen_addr
    server.port = args.port

    try:
        await server.setup()
        await run(server, address=first_listen_addr, port=args.port, call_on_start=call_on_start)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.debug("Stopped server")
    finally:
        _shutdown_event.set()
        if _prompt_worker_executor is not None:
            _prompt_worker_executor.shutdown(wait=True, cancel_futures=True)
        server.asset_manager.shutdown()
        if distributed:
            await q.close()

    if server.restart_requested:
        # Reconstruct a command that works after os.execv.
        # When invoked via ``python -m comfy.cmd.main``, sys.argv[0] is
        # the file path, which breaks relative imports if executed
        # directly.  Detect this and use ``-m`` instead.
        argv0 = sys.argv[0]
        if argv0.endswith(("main.py", "main.pyc")) and os.path.isfile(argv0):
            restart_args = [sys.executable, "-m", "comfy.cmd.main"] + sys.argv[1:]
        else:
            restart_args = [sys.executable] + sys.argv
        logger.info("Restarting server via os.execv(%s, %s)", sys.executable, restart_args)
        os.execv(sys.executable, restart_args)


def entrypoint():
    """Delegates to the Typer CLI."""
    from .cli import entrypoint as cli_entrypoint
    cli_entrypoint()


def main():
    entrypoint()


if __name__ == "__main__":
    entrypoint()

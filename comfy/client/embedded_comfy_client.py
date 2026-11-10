from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import copy
import multiprocessing
import gc
import json
import logging
import threading
import uuid
from asyncio import get_event_loop
from multiprocessing import RLock
from typing import Optional, Literal, NamedTuple

from opentelemetry import context, propagate, trace
from opentelemetry.context import Context, attach, detach
from opentelemetry.trace import Status, StatusCode

from .async_progress_iterable import QueuePromptWithProgress
from .client_types import V1QueuePromptResponse
from ..api.components.schema.prompt import PromptDict
from ..cli_args_types import Configuration
from ..cli_args import default_configuration
from ..cmd.folder_paths import init_default_paths
from ..component_model.executor_types import ExecutorToClientProgress, HistoryResultDict
from ..component_model.make_mutable import make_mutable
from ..component_model.queue_types import QueueItem, ExecutionStatus, TaskInvocation, QueueTuple, ExtraData
from ..component_model.workflow_convert import is_ui_workflow, convert_ui_to_api
from ..distributed.executors import ContextVarExecutor
from ..distributed.history import History
from ..distributed.process_pool_executor import ProcessPoolExecutor
from ..distributed.server_stub import ServerStub
from ..component_model.configuration import MODEL_MANAGEMENT_ARGS, requires_process_pool_executor, model_management_fingerprint

_prompt_executor = threading.local()

logger = logging.getLogger(__name__)


class _PromptExecutionResult(NamedTuple):
    history: HistoryResultDict
    status: ExecutionStatus
    exception: Exception | None = None


def _add_node_site_to_path(configuration: Configuration | None) -> None:
    """Ensure custom-node dependencies installed to ``node_site/`` are importable."""
    if configuration is None or configuration.base_directory is None:
        return
    from ..component_model.site_packages import add_node_site
    add_node_site(configuration.base_directory)


_worker_process_setup_done = False


def _setup_worker_process(configuration: Configuration) -> None:
    """Run the CLI's post-torch setup once inside a spawned worker process.

    ``comfyui serve`` reaches DynamicVRAM (comfy-aimdo) through
    ``setup_post_torch``; a process-pool worker otherwise imports torch and
    model_management without it and silently runs the legacy ModelPatcher.
    The in-process thread executor is left alone: swapping the CUDA allocator
    of the host process is the caller's decision, not the client's.
    """
    global _worker_process_setup_done
    if _worker_process_setup_done or multiprocessing.parent_process() is None:
        return
    _worker_process_setup_done = True
    from ..component_model.setup import setup_post_torch
    from ..execution_context import context_configuration
    with context_configuration(configuration):
        setup_post_torch(configuration)


def _execute_prompt(
        prompt: dict,
        prompt_id: str,
        client_id: str,
        span_context: dict,
        progress_handler: ExecutorToClientProgress | None,
        configuration: Configuration | None,
        partial_execution_targets: Optional[list[str]] = None) -> _PromptExecutionResult:
    configuration = copy.deepcopy(configuration) if configuration is not None else None

    # In a ProcessPoolExecutor subprocess, main_pre.py runs at import time
    # before the context is available, so CUDA_VISIBLE_DEVICES won't be set
    # from the configuration. Apply device env vars here, before torch is
    # imported by model_management.
    if configuration is not None:
        from ..component_model.setup import setup_cuda_devices
        setup_cuda_devices(configuration)
        _setup_worker_process(configuration)

    # Ensure custom-node deps installed to node_site/ are importable in this
    # worker (thread or spawned process).
    _add_node_site_to_path(configuration)

    from ..execution_context import current_execution_context
    execution_context = current_execution_context()
    if len(execution_context.folder_names_and_paths) == 0 or configuration is not None:
        init_default_paths(execution_context.folder_names_and_paths, configuration, replace_existing=True)
    span_context: Context = propagate.extract(span_context)
    token = attach(span_context)
    try:
        # there is never an event loop running on a thread or process pool thread here
        # this also guarantees nodes will be able to successfully call await
        return asyncio.run(__execute_prompt(prompt, prompt_id, client_id, span_context, progress_handler, configuration, partial_execution_targets))
    finally:
        detach(token)


async def __execute_prompt(
        prompt: dict,
        prompt_id: str,
        client_id: str,
        span_context: Context,
        progress_handler: ExecutorToClientProgress | None,
        configuration: Configuration | None,
        partial_execution_targets: list[str] | None) -> _PromptExecutionResult:
    from ..execution_context import context_configuration
    with context_configuration(configuration):
        return await ___execute_prompt(prompt, prompt_id, client_id, span_context, progress_handler, partial_execution_targets)


async def ___execute_prompt(
        prompt: dict,
        prompt_id: str,
        client_id: str,
        span_context: Context,
        progress_handler: ExecutorToClientProgress | None,
        partial_execution_targets: list[str] | None) -> _PromptExecutionResult:
    from ..cmd.execution import PromptExecutor

    progress_handler = progress_handler or ServerStub()
    prompt_executor: PromptExecutor = None
    try:
        prompt_executor: PromptExecutor = _prompt_executor.executor
    except (LookupError, AttributeError):
        with trace.get_tracer(__name__).start_as_current_span("Initialize Prompt Executor", context=span_context):
            # todo: deal with new caching features
            prompt_executor = PromptExecutor(progress_handler)
            prompt_executor.raise_exceptions = True
            _prompt_executor.executor = prompt_executor

    prompt_executor.status_messages = []
    prompt_executor.history_result = None
    prompt_executor.success = False
    from ..nodes.download_interception import patch_folder_paths_functions
    with trace.get_tracer(__name__).start_as_current_span("Execute Prompt", context=span_context) as span, patch_folder_paths_functions():
        try:
            prompt_mut = make_mutable(prompt)
            from ..cmd.execution import validate_prompt
            validation_tuple = await validate_prompt(prompt_id, prompt_mut, partial_execution_targets)
            if not validation_tuple.valid:
                if validation_tuple.node_errors is not None and len(validation_tuple.node_errors) > 0:
                    validation_error_dict = validation_tuple.node_errors
                elif validation_tuple.error is not None:
                    validation_error_dict = validation_tuple.error
                else:
                    validation_error_dict = {"message": "Unknown", "details": ""}
                raise ValueError(json.dumps(validation_error_dict))

            if client_id is None:
                prompt_executor.server = ServerStub()
            else:
                prompt_executor.server = progress_handler

            await prompt_executor.execute_async(prompt_mut, prompt_id, {"client_id": client_id},
                                                execute_outputs=validation_tuple.good_output_node_ids)
            return _PromptExecutionResult(prompt_executor.history_result, prompt_executor.execution_status)
        except Exception as exc_info:
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(exc_info)
            status = prompt_executor.execution_status
            if not any(event in ('execution_error', 'execution_interrupted') for event, _ in status.messages):
                status = ExecutionStatus('error', False, status.messages + ExecutionStatus.from_exception(prompt_id, exc_info).messages)
            return _PromptExecutionResult(prompt_executor.history_result or {"outputs": {}, "meta": {}}, status, exc_info)


def _cleanup(invalidate_nodes=True, unload_models=True):
    from ..cmd.execution import PromptExecutor
    from ..nodes_context import invalidate
    try:
        prompt_executor: PromptExecutor = _prompt_executor.executor
        # this should clear all references to output tensors and make it easier to collect back the memory
        prompt_executor.reset()
        prompt_executor.history_result = None
    except (LookupError, AttributeError):
        pass
    from .. import model_management
    if unload_models:
        model_management.unload_all_models()
    gc.collect()
    try:
        model_management.soft_empty_cache()
    except:
        pass
    if invalidate_nodes:
        try:
            invalidate()
        except:
            pass


class Comfy:
    """
    A client for running ComfyUI workflows within a Python application.

    This client allows you to execute ComfyUI workflows (in API JSON format) programmatically.
    It manages the execution environment, including model loading and resource cleanup.

    ### Configuration and Executors

    ComfyUI relies on global state for model management (e.g., loaded models in VRAM). To handle this safely, `Comfy`
    executes workflows using one of two strategies based on your `configuration`:

    1.  **ContextVarExecutor (Default)**: Runs in a thread pool within the current process.
        -   **Pros**: Efficient, low overhead.
        -   **Cons**: Modifies global state in the current process.
        -   **Use Case**: Standard workflows where you are happy with the default ComfyUI settings or sharing state.

    2.  **ProcessPoolExecutor**: Runs in a separate process.
        -   **Pros**: Complete isolation. Configuration changes (like `lowvram`) do not affect the main process.
        -   **Cons**: Higher overhead (process startup).
        -   **Use Case**: Required when `configuration` overrides arguments that affect global model management state.
            These arguments include: `lowvram`, `highvram`, `cpu`, `gpu_only`, `deterministic`, `directml`,
            various `fp8`/`fp16`/`bf16` settings, and attention optimizations (e.g., `use_flash_attention`).

    The client automatically selects `ProcessPoolExecutor` if you provide a `configuration` that modifies any of these
    global settings, unless you explicitly pass an `executor`.

    ### Parameters

    -   **configuration** (`Optional[Configuration]`): A dictionary of arguments to override defaults.
        See `comfy.cli_args_types.Configuration`.
        Example: `{"lowvram": True}` or `{"gpu_only": True}`.
    -   **progress_handler** (`Optional[ExecutorToClientProgress]`): callback handler for progress updates and previews.
    -   **max_workers** (`int`): Maximum number of concurrent workflows (default: 1).
    -   **executor** (`Optional[Union[Executor, str]]`): Explicitly define the executor to use.
        -   Pass an instance of `ProcessPoolExecutor` or `ContextVarExecutor`.
        -   Pass the string `"ProcessPoolExecutor"` or `"ContextVarExecutor"` to force initialization of that type.
        -   If `None` (default), the best executor is chosen based on `configuration`.

    ### Examples

    #### 1. Running a Workflow (Basic)

    This example executes a simple workflow and prints the path of the saved image.

    ```python
    import asyncio
    from comfy.client.embedded_comfy_client import Comfy

    # A simple API format workflow (simplified for brevity)
    prompt_dict = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 8566257, "steps": 20, "cfg": 8, "sampler_name": "euler",
                "scheduler": "normal", "denoise": 1,
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["5", 0]
            }
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "masterpiece best quality girl", "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "bad hands", "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI_API", "images": ["8", 0]}}
    }

    async def main():
        # Using default configuration (runs in-process)
        async with Comfy() as client:
            # Queue the prompt and await the result
            outputs = await client.queue_prompt(prompt_dict)

            # Retrieve the output path from the SaveImage node (Node ID "9")
            image_path = outputs["9"]["images"][0]["abs_path"]
            print(f"Image saved to: {image_path}")

    # asyncio.run(main())
    ```

    #### 2. Using Custom Configuration (Isolated Process)

    To run with specific settings like `lowvram`, pass the configuration. This implies `ProcessPoolExecutor`.

    ```python
    async def run_lowvram():
        # This will spawn a new process with lowvram enabled
        async with Comfy(configuration={"lowvram": True}) as client:
            outputs = await client.queue_prompt(prompt_dict)
            print("Finished lowvram generation")
    ```

    #### 3. Programmatically Building Workflows

    You can use `GraphBuilder` constructing workflows with a more pythonic API.

    ```python
    from comfy_execution.graph_utils import GraphBuilder

    def build_graph():
        builder = GraphBuilder()
        checkpoint = builder.node("CheckpointLoaderSimple", ckpt_name="v1-5-pruned-emaonly.safetensors")
        latent = builder.node("EmptyLatentImage", width=512, height=512, batch_size=1)
        pos = builder.node("CLIPTextEncode", text="masterpiece", clip=checkpoint.out(1))
        neg = builder.node("CLIPTextEncode", text="bad quality", clip=checkpoint.out(1))
        
        sampler = builder.node("KSampler", 
            seed=42, steps=20, cfg=8, sampler_name="euler", scheduler="normal", denoise=1,
            model=checkpoint.out(0), positive=pos.out(0), negative=neg.out(0), latent_image=latent.out(0)
        )
        vae = builder.node("VAEDecode", samples=sampler.out(0), vae=checkpoint.out(2))
        builder.node("SaveImage", filename_prefix="Generated", images=vae.out(0))
        return builder.finalize()

    async def run_builder():
        prompt = build_graph()
        async with Comfy() as client:
            await client.queue_prompt(prompt)
    ```

    #### 4. Streaming Progress and Previews

    To receive real-time progress updates and preview images (e.g., step-by-step decoding).

    ```python
    from comfy.component_model.queue_types import BinaryEventTypes

    async def run_streaming():
        async with Comfy() as client:
            # Get a task that supports progress iteration
            task = client.queue_with_progress(prompt_dict)
            
            async for notification in task.progress():
                if notification.event == BinaryEventTypes.PREVIEW_IMAGE_WITH_METADATA:
                    # 'data' contains the PIL Image and metadata
                    image, metadata = notification.data
                    print(f"Received preview: {image.size}")
                elif notification.event == "progress":
                    print(f"Step: {notification.data['value']}/{notification.data['max']}")

            # Await final result
            result = await task.get()
    ```
    """

    def __init__(self, configuration: Optional[Configuration] = None, progress_handler: Optional[ExecutorToClientProgress] = None, max_workers: int = 1, executor: ProcessPoolExecutor | ContextVarExecutor | Literal["ProcessPoolExecutor", "ContextVarExecutor"] = None):
        self._progress_handler = progress_handler or ServerStub()
        self._default_configuration = default_configuration()
        self._configuration = configuration

        need_process_pool = requires_process_pool_executor(configuration)

        if executor is None:
            if need_process_pool:
                self._executor = ProcessPoolExecutor(max_workers=max_workers)
                self._owns_executor = True
            else:
                self._executor = ContextVarExecutor(max_workers=max_workers)
                self._owns_executor = True
        elif isinstance(executor, str):
            self._owns_executor = True
            if executor == "ProcessPoolExecutor":
                self._executor = ProcessPoolExecutor(max_workers=max_workers)
            elif executor == "ContextVarExecutor":
                if need_process_pool:
                    raise ValueError(f"Configuration requires ProcessPoolExecutor but ContextVarExecutor was requested. Configuration keys causing this: {[k for k in MODEL_MANAGEMENT_ARGS if configuration.get(k) != self._default_configuration.get(k)]}")
                self._executor = ContextVarExecutor(max_workers=max_workers)
            else:
                raise ValueError(f"Unknown executor type string: {executor}")
        else:
            # Executor instance passed
            self._owns_executor = False
            self._executor = executor
            if need_process_pool and not isinstance(executor, ProcessPoolExecutor):
                raise ValueError(f"Configuration requires ProcessPoolExecutor but {type(executor).__name__} was passed. Configuration keys causing this: {[k for k in MODEL_MANAGEMENT_ARGS if configuration.get(k) != self._default_configuration.get(k)]}")

        self._is_running = False
        self._task_count_lock = RLock()
        self._task_count = 0
        self._history = History()
        self._exit_stack = None
        self._async_exit_stack = None
        self._in_process_execution_lock = asyncio.Lock()
        # Remember the user's requested concurrency so reconfigure() can
        # rebuild the executor with the same max_workers.
        self._max_workers = max_workers
        # Cache the fingerprint of the MODEL_MANAGEMENT_ARGS subset so
        # reconfigure() only cycles the subprocess when the new config
        # actually affects model management state.
        self._fingerprint = model_management_fingerprint(configuration)

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def task_count(self) -> int:
        return self._task_count

    def __enter__(self):
        self._exit_stack = contextlib.ExitStack()
        self._is_running = True
        from ..execution_context import context_configuration
        cm = context_configuration(self._configuration)
        self._exit_stack.enter_context(cm)
        from ..component_model.setup import setup_tracing
        setup_tracing(self._configuration or self._default_configuration)
        if self._owns_executor:
            self._exit_stack.enter_context(self._executor)
        return self

    @property
    def history(self) -> History:
        return self._history

    async def clear_cache(self, *, unload_models=True):
        """Clear request/node caches and, by default, unload models.

        With unload_models=False, caller-owned models remain resident. Models
        owned only by cleared node caches can still be garbage-collected.
        """
        await get_event_loop().run_in_executor(self._executor, _cleanup, False, unload_models)

    @property
    def fingerprint(self) -> tuple:
        """Fingerprint of the current configuration's MODEL_MANAGEMENT_ARGS.

        Exposed for tests and for the server-side ``/api/v1/prompts``
        worker loop that needs to decide whether the next job's
        ``__metadata_v1__.configuration`` requires a subprocess swap.
        """
        return self._fingerprint

    async def reconfigure(self, new_configuration: Optional[Configuration]) -> bool:
        """Swap in a new Configuration, cycling the subprocess only when needed.

        Returns True when the executor was actually rebuilt (i.e., the
        MODEL_MANAGEMENT_ARGS fingerprint changed and we own the
        executor), False when the existing worker is compatible and the
        configuration simply replaces the stored reference.

        The old executor is cleanly shut down first — its queued tasks
        run to completion before the new one takes over. Because
        ``max_workers=1`` is the default, this amounts to "finish the
        current job, then start the new subprocess for job N+1".

        Callers that passed an executor instance in ``__init__`` can't
        have the executor swapped out from under them; in that case this
        method only updates the stored configuration and returns False.
        """
        new_fp = model_management_fingerprint(new_configuration)
        if new_fp == self._fingerprint:
            # Fast path: same fingerprint → same worker.
            self._configuration = new_configuration
            return False

        if not self._owns_executor:
            # We can't cycle an executor we don't own. Update the stored
            # config and let queue_prompt's per-call context_configuration
            # propagate the change as best it can.
            self._configuration = new_configuration
            self._fingerprint = new_fp
            return False

        # Shut down the old executor (max_workers=1 ⇒ one in-flight job
        # finishes, then the process terminates).
        old_executor = self._executor
        old_executor.shutdown(wait=True)

        # Rebuild with the same max_workers and the same decision about
        # ProcessPool vs ContextVar. New subprocesses inherit the new
        # Configuration via context_configuration() in
        # _execute_prompt → setup_cuda_devices.
        need_process_pool = requires_process_pool_executor(new_configuration)
        if need_process_pool:
            self._executor = ProcessPoolExecutor(max_workers=self._max_workers)
        else:
            self._executor = ContextVarExecutor(max_workers=self._max_workers)

        # Re-register the executor with any open exit stack so __exit__ /
        # __aexit__ still cleans it up when the client tears down.
        if self._async_exit_stack is not None:
            self._async_exit_stack.enter_context(self._executor)
        elif self._exit_stack is not None:
            self._exit_stack.enter_context(self._executor)

        self._configuration = new_configuration
        self._fingerprint = new_fp
        return True

    def __exit__(self, *args):
        try:
            get_event_loop().run_in_executor(self._executor, _cleanup)
            self._is_running = False
            self._exit_stack.__exit__(*args)
        finally:
            from ..component_model.setup import flush_tracing
            flush_tracing()

    async def __aenter__(self):
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._is_running = True
        from ..execution_context import context_configuration
        cm = context_configuration(self._configuration)
        self._async_exit_stack.enter_context(cm)
        from ..component_model.setup import setup_tracing
        setup_tracing(self._configuration or self._default_configuration)
        if self._owns_executor:
            self._async_exit_stack.enter_context(self._executor)

        from ..manager_model_cache import init_manager_model_cache
        init_manager_model_cache(
            disabled=getattr(self._configuration, "disable_manager_model_fallback", False),
        )

        from .. import civitai_model_cache
        civitai_model_cache.init_civitai_model_cache(
            disabled=getattr(self._configuration, "disable_civitai_model_fallback", False),
        )

        return self

    async def __aexit__(self, *args):
        exc_type = args[0] if args else None
        if exc_type in (asyncio.CancelledError, KeyboardInterrupt):
            from .. import interruption
            interruption.interrupt_current_processing()

        while self.task_count > 0:
            await asyncio.sleep(0.1)

        try:
            await get_event_loop().run_in_executor(self._executor, _cleanup)
            self._is_running = False
            await self._async_exit_stack.__aexit__(*args)
        finally:
            from ..component_model.setup import flush_tracing
            flush_tracing()

    async def queue_prompt_api(self,
                               prompt: PromptDict | str | dict,
                               progress_handler: Optional[ExecutorToClientProgress] = None,
                               prompt_id: Optional[str] = None) -> V1QueuePromptResponse:
        """
        Queues a prompt for execution, returning the output when it is complete.
        :param prompt: a PromptDict, string or dictionary containing a so-called Workflow API prompt
        :return: a response of URLs for Save-related nodes and the node outputs
        """
        prompt_id = prompt_id or str(uuid.uuid4())
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        if isinstance(prompt, dict):
            from ..api.components.schema.prompt import Prompt
            prompt = Prompt.validate(prompt)
        outputs = await self.queue_prompt(prompt, progress_handler=progress_handler, prompt_id=prompt_id)
        return V1QueuePromptResponse(urls=[], outputs=outputs, prompt_id=prompt_id)

    def queue_with_progress(self, prompt: PromptDict | str | dict) -> QueuePromptWithProgress:
        """
        Queues a prompt with progress notifications.

        >>> from comfy.client.embedded_comfy_client import Comfy
        >>> from comfy.client.client_types import ProgressNotification
        >>> async with Comfy() as comfy:
        >>>     task = comfy.queue_with_progress({ ... })
        >>>     # Raises an exception while iterating
        >>>     notification: ProgressNotification
        >>>     async for notification in task.progress():
        >>>         print(notification.data)
        >>>     # If you get this far, no errors occurred.
        >>>     result = await task.get()
        :param prompt:
        :return:
        """
        handler = QueuePromptWithProgress()
        task = asyncio.create_task(self.queue_prompt_api(prompt, progress_handler=handler.progress_handler))
        task.add_done_callback(handler.complete)
        return handler

    async def queue_prompt(self,
                           prompt: PromptDict | dict,
                           prompt_id: Optional[str] = None,
                           client_id: Optional[str] = None,
                           partial_execution_targets: Optional[list[str]] = None,
                           progress_handler: Optional[ExecutorToClientProgress] = None,
                           configuration: Optional[Configuration] = None) -> dict:
        with trace.get_tracer(__name__).start_as_current_span("Queue Prompt"):
            return await self._queue_prompt(
                prompt,
                prompt_id=prompt_id,
                client_id=client_id,
                partial_execution_targets=partial_execution_targets,
                progress_handler=progress_handler,
                configuration=configuration,
            )

    async def _queue_prompt(self,
                           prompt: PromptDict | dict,
                           prompt_id: Optional[str] = None,
                           client_id: Optional[str] = None,
                           partial_execution_targets: Optional[list[str]] = None,
                           progress_handler: Optional[ExecutorToClientProgress] = None,
                           configuration: Optional[Configuration] = None) -> dict:
        if isinstance(self._executor, ProcessPoolExecutor) and progress_handler is not None:
            logger.debug(f"a progress_handler={progress_handler} was passed, it must be pickleable to support ProcessPoolExecutor")
        progress_handler = progress_handler or self._progress_handler

        if isinstance(prompt, dict) and is_ui_workflow(prompt):
            prompt = convert_ui_to_api(prompt)

        # Strip ``__metadata_v1__`` off the prompt before the worker sees it
        # so validate_prompt stays upstream-vanilla. Any per-job
        # Configuration overrides live in metadata["configuration"] — apply
        # them here (via reconfigure) so the worker that runs this job has
        # the right VRAM/precision/attention state.
        call_configuration: Optional[Configuration] = configuration
        if isinstance(prompt, dict):
            from ..component_model.prompt_envelope import extract_metadata
            prompt, metadata = extract_metadata(prompt)
            if call_configuration is None and isinstance(metadata, dict):
                conf = metadata.get("configuration")
                if isinstance(conf, dict) and conf:
                    call_configuration = Configuration(**conf)

        if call_configuration is not None and call_configuration != self._configuration:
            await self.reconfigure(call_configuration)

        with self._task_count_lock:
            self._task_count += 1
        prompt_id = prompt_id or str(uuid.uuid4())
        assert prompt_id is not None
        client_id = client_id or self._progress_handler.client_id or None
        span_context = context.get_current()
        carrier = {}
        propagate.inject(carrier, span_context)
        prompt = make_mutable(prompt)

        queue_item = QueueItem(QueueTuple(float(self._task_count), prompt_id, prompt, ExtraData(), [], {}), None)

        async def run_prompt_in_executor() -> _PromptExecutionResult:
            return await get_event_loop().run_in_executor(
                self._executor,
                _execute_prompt,
                prompt,
                prompt_id,
                client_id,
                carrier,
                # todo: a proxy object or something more sophisticated will have to be done here to restore progress notifications for ProcessPoolExecutors
                None if isinstance(self._executor, ProcessPoolExecutor) else progress_handler,
                self._configuration,
                partial_execution_targets,
            )

        execution_result = None
        try:
            if isinstance(self._executor, ContextVarExecutor):
                async with self._in_process_execution_lock:
                    execution_result = await run_prompt_in_executor()
            else:
                execution_result = await run_prompt_in_executor()

            if execution_result.exception is not None:
                raise execution_result.exception
            history_result = execution_result.history

            outputs = history_result["outputs"]
            fut = concurrent.futures.Future()
            try:
                outputs_copy = copy.deepcopy(outputs)
            except Exception:
                outputs_copy = outputs
            fut.set_result(TaskInvocation(prompt_id, outputs_copy, execution_result.status))
            queue_item.completed = fut
            self._history.put(queue_item, outputs, execution_result.status, meta=history_result["meta"])
            return outputs
        except Exception as exc_info:
            fut = concurrent.futures.Future()
            fut.set_exception(exc_info)
            status = execution_result.status if execution_result is not None else ExecutionStatus.from_exception(prompt_id, exc_info)
            queue_item.completed = fut
            self._history.put(queue_item, {}, status)
            raise exc_info
        finally:
            with self._task_count_lock:
                self._task_count -= 1

    def __str__(self):
        diff = {k: v for k, v in (self._configuration or {}).items() if v != self._default_configuration.get(k)}
        return f"<Comfy task_count={self.task_count} configuration={diff} executor={self._executor}>"


EmbeddedComfyClient = Comfy

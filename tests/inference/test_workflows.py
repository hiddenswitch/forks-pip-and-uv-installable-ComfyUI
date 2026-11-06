import atexit
import asyncio
import contextlib
import importlib.resources
import json
import logging
import threading
import time
from importlib.abc import Traversable
from pathlib import Path
from typing import Any, AsyncGenerator

import psutil
import torch

try:
    import pynvml
except ImportError:
    pynvml = None

import pytest

from comfy.api.components.schema.prompt import Prompt
from comfy.client.embedded_comfy_client import Comfy
from comfy.distributed.process_pool_executor import ProcessPoolExecutor
from comfy.model_downloader import add_known_models
from comfy.model_downloader_types import HuggingFile
from . import workflows
from tests.inference_benchmark import install_benchmark, benchmark_rows, prepare_benchmark_prompt
from tests.inference.mini_weights import miniature_models
from comfy.cli_args import default_configuration
from comfy.cli_args_types import PerformanceFeature

logger = logging.getLogger(__name__)
_LOCAL_TEST_IMAGE = "pkg://tests.custom_nodes.test_data/president_official_portrait_hires2-1-1024x1024.jpg"
_mini_loaded = None


class ResourceMonitor:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_cpu_ram = 0
        self.peak_gpu_vram = 0
        self._stop_event = threading.Event()
        self._thread = None
        self._pynvml_available = False
        self._gpu_handles = []

    def _monitor(self):
        current_process = psutil.Process()
        while not self._stop_event.is_set():
            # Monitor CPU RAM (RSS) for process tree
            try:
                children = current_process.children(recursive=True)
                processes = [current_process] + children
                pids = {p.pid for p in processes}

                total_rss = 0
                for p in processes:
                    try:
                        total_rss += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak_cpu_ram = max(self.peak_cpu_ram, total_rss)

                # Monitor GPU VRAM if available
                if self._pynvml_available and self._gpu_handles:
                    total_vram = 0
                    try:
                        # Iterate over all detected GPUs
                        for handle in self._gpu_handles:
                            # Get all processes running on the GPU
                            try:
                                compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                                graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)

                                # Filter for our process tree
                                for p in compute_procs + graphics_procs:
                                    if p.pid in pids:
                                        total_vram += p.usedGpuMemory
                            except Exception:
                                pass  # Skip errors for specific GPU queries

                        self.peak_gpu_vram = max(self.peak_gpu_vram, total_vram)
                    except Exception:
                        pass
            except Exception:
                pass

            time.sleep(self.interval)

    def __enter__(self):
        if pynvml:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)]
                self._pynvml_available = True
            except Exception as e:
                logger.warning(f"Failed to initialize pynvml for VRAM monitoring: {e}")

        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        if self._pynvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def _generate_config_params():
    yield {
        "use_pytorch_cross_attention": True,
        "fast": {PerformanceFeature.CublasOps, PerformanceFeature.Fp16Accumulation},
    }


@pytest.fixture(scope="module", params=_generate_config_params(), ids=lambda p: ",".join(f"{k}={sorted(v, key=str) if isinstance(v, set) else v}" for k, v in sorted(p.items())))
def workflow_process(tmp_path_factory, request):
    # Keep imports warm, with a separate worker for each configuration. Each client
    # still runs native cleanup to discard prompt caches and unload its models.
    benchmark_path = tmp_path_factory.mktemp("benchmark") / "steps.jsonl"
    initializer = _initialize_mini_worker if request.config.getoption("--weights") == "mini" else _initialize_worker
    with ProcessPoolExecutor(max_workers=1, initializer=initializer, initargs=(str(benchmark_path),)) as executor:
        yield executor, benchmark_path, request.param


@pytest.fixture
async def client(tmp_path_factory, request, workflow_process) -> AsyncGenerator[Any, Any]:
    executor, benchmark_path, configuration = workflow_process
    first_row = len(benchmark_rows(benchmark_path))
    config = default_configuration()
    config.disable_all_custom_nodes = True
    config.update(configuration)
    if request.config.getoption("--weights") == "mini":
        config.disable_dynamic_vram = True
        config.force_fp32 = True
        config.fp32_unet = True
        config.fp32_vae = True
        config.cwd = str(tmp_path_factory.mktemp("mini_cwd"))
        config.input_directory = str(Path.cwd() / "input")
        config.output_directory = str(tmp_path_factory.mktemp("mini_output"))
        await asyncio.wrap_future(executor.submit(_reset_mini_materialization))
    async with Comfy(configuration=config, executor=executor) as client:
        yield client
    for row in benchmark_rows(benchmark_path)[first_row:]:
        request.node.user_properties.append(("inference_benchmark", row))


def _register_workflow_models():
    add_known_models("loras", HuggingFile("artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5", "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors"))
    add_known_models("checkpoints", HuggingFile("autismanon/modeldump", "cardosAnime_v20.safetensors"))


def _initialize_worker(benchmark_path: str):
    # Workers are spawned, so models registered in the test process are registered again here.
    _register_workflow_models()
    install_benchmark(benchmark_path)


def _initialize_mini_worker(benchmark_path: str):
    global _mini_loaded
    from comfy import model_management
    from tests.mini_downloads import miniature_downloads
    device = model_management.get_torch_device()
    context = contextlib.ExitStack()
    context.enter_context(miniature_downloads(Path(benchmark_path).parent.parent))
    _mini_loaded = context.enter_context(miniature_models(initialization_device=device if device.type == "cuda" else "cpu"))
    atexit.register(context.__exit__, None, None, None)
    _initialize_worker(benchmark_path)


def _reset_mini_materialization():
    _mini_loaded.clear()


def _mini_materialized_modules():
    return _mini_loaded


def _prepare_for_workflows():
    _register_workflow_models()
    files = sorted((f for f in importlib.resources.files(workflows).iterdir() if f.is_file() and f.name.endswith(".json")), key=lambda f: f.name)
    return [pytest.param(f.name, f, id=f.name) for f in files]


def _replace_remote_media_inputs(workflow: dict) -> None:
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        value = inputs.get("value")
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            inputs["value"] = _LOCAL_TEST_IMAGE


@pytest.mark.inference
@pytest.mark.requires_weights
@pytest.mark.requires_accelerator
@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_name, workflow_file", _prepare_for_workflows())
async def test_workflow(workflow_name: str, workflow_file: Traversable, has_gpu: bool, client: Comfy, request):
    if not has_gpu:
        pytest.skip("requires gpu")

    workflow = json.loads(workflow_file.read_text(encoding="utf8"))

    _replace_remote_media_inputs(workflow)
    workflow = prepare_benchmark_prompt(workflow)
    if request.config.getoption("--weights") == "mini":
        for node in workflow.values():
            inputs = node.get('inputs', {})
            if isinstance(inputs.get('max_new_tokens'), int):
                inputs['max_new_tokens'] = min(8, inputs['max_new_tokens'])
            if 'Latent' in node.get('class_type', ''):
                for key in ('width', 'height'):
                    value = node.get('inputs', {}).get(key)
                    if isinstance(value, int):
                        # Retain odd latent/patch boundaries exercised by regression workflows.
                        node['inputs'][key] = value if value <= 64 else 64 + value % 64
    prompt = Prompt.validate(workflow)
    # todo: add all the models we want to test a bit m2ore elegantly
    outputs = {}

    start_time = time.time()
    monitor = ResourceMonitor()
    try:
        with monitor:
            outputs = await client.queue_prompt(prompt)
            if request.config.getoption("--weights") == "mini":
                loaded = await asyncio.wrap_future(client._executor.submit(_mini_materialized_modules))
                assert loaded, "Workflow bypassed miniature checkpoint loading"
                request.node.user_properties.append(("miniature_modules", loaded))
    except torch.OutOfMemoryError:
        if request.config.getoption("--weights") == "mini":
            raise
        pytest.skip(f"insufficient VRAM for workflow {workflow_name}")
    finally:
        end_time = time.time()
        duration = end_time - start_time
        ram_gb = monitor.peak_cpu_ram / (1024 ** 3)
        vram_gb = monitor.peak_gpu_vram / (1024 ** 3)
        logger.info(f"Test {workflow_name} with client {client} took {duration:.4f}s | Peak RAM: {ram_gb:.2f} GB | Peak VRAM: {vram_gb:.2f} GB")

    if any(v.class_type == "SaveImage" for v in prompt.values()):
        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
        image_path = outputs[save_image_node_id]["images"][0]["abs_path"]
        assert image_path is not None and Path(image_path).is_file()
    elif any(v.class_type == "SaveAudio" for v in prompt.values()):
        save_audio_node_id = next(key for key in prompt if prompt[key].class_type == "SaveAudio")
        assert outputs[save_audio_node_id]["audio"][0]["filename"] is not None
    elif any(v.class_type == "SaveAnimatedWEBP" for v in prompt.values()):
        save_video_node_id = next(key for key in prompt if prompt[key].class_type == "SaveAnimatedWEBP")
        assert outputs[save_video_node_id]["images"][0]["filename"] is not None
    elif any(v.class_type == "PreviewString" for v in prompt.values()):
        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "PreviewString")
        output_str = outputs[save_image_node_id]["string"][0]
        assert output_str is not None
        if request.config.getoption("--weights") == "mini" and any(v.class_type == "TransformersLoader" for v in prompt.values()):
            # Random models may emit only special tokens, which decode to an empty string.
            assert isinstance(output_str, str)
            assert any(name.endswith(':finite logits') for name in loaded)
        else:
            assert len(output_str) > 0
    else:
        assert len(outputs) > 0
        logger.warning(f"test {workflow_name} did not have a node that could be checked for output")

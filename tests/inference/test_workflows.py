import importlib.resources
import json
import logging
import threading
import time
from importlib.abc import Traversable
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
from comfy.cli_args import default_configuration
from comfy.cli_args_types import PerformanceFeature

logger = logging.getLogger(__name__)
_LOCAL_TEST_IMAGE = "pkg://tests.custom_nodes.test_data/president_official_portrait_hires2-1-1024x1024.jpg"


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


@pytest.fixture(scope="function", autouse=False, params=_generate_config_params(), ids=lambda p: ",".join(f"{k}={v}" for k, v in p.items()))
async def client(tmp_path_factory, request) -> AsyncGenerator[Any, Any]:
    config = default_configuration()
    # this should help things go a little faster
    config.disable_all_custom_nodes = True
    config.update(request.param)
    # use ProcessPoolExecutor to respect various config settings
    benchmark_path = tmp_path_factory.mktemp("benchmark") / "steps.jsonl"
    with ProcessPoolExecutor(max_workers=1, initializer=install_benchmark, initargs=(str(benchmark_path),)) as executor:
        async with Comfy(configuration=config, executor=executor) as client:
            yield client
    for row in benchmark_rows(benchmark_path):
        request.node.user_properties.append(("inference_benchmark", row))


def _prepare_for_workflows() -> dict[str, Traversable]:
    add_known_models("loras", HuggingFile("artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5", "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors"))
    add_known_models("checkpoints", HuggingFile("autismanon/modeldump", "cardosAnime_v20.safetensors"))

    return {f.name: f for f in importlib.resources.files(workflows).iterdir() if f.is_file() and f.name.endswith(".json")}


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
@pytest.mark.asyncio
@pytest.mark.parametrize("workflow_name, workflow_file", _prepare_for_workflows().items())
async def test_workflow(workflow_name: str, workflow_file: Traversable, has_gpu: bool, client: Comfy):
    if not has_gpu:
        pytest.skip("requires gpu")

    workflow = json.loads(workflow_file.read_text(encoding="utf8"))

    _replace_remote_media_inputs(workflow)
    workflow = prepare_benchmark_prompt(workflow)
    prompt = Prompt.validate(workflow)
    # todo: add all the models we want to test a bit m2ore elegantly
    outputs = {}

    start_time = time.time()
    monitor = ResourceMonitor()
    try:
        with monitor:
            outputs = await client.queue_prompt(prompt)
    except torch.OutOfMemoryError:
        pytest.skip(f"insufficient VRAM for workflow {workflow_name}")
    finally:
        end_time = time.time()
        duration = end_time - start_time
        ram_gb = monitor.peak_cpu_ram / (1024 ** 3)
        vram_gb = monitor.peak_gpu_vram / (1024 ** 3)
        logger.info(f"Test {workflow_name} with client {client} took {duration:.4f}s | Peak RAM: {ram_gb:.2f} GB | Peak VRAM: {vram_gb:.2f} GB")

    if any(v.class_type == "SaveImage" for v in prompt.values()):
        save_image_node_id = next(key for key in prompt if prompt[key].class_type == "SaveImage")
        assert outputs[save_image_node_id]["images"][0]["abs_path"] is not None
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
        assert len(output_str) > 0
    else:
        assert len(outputs) > 0
        logger.warning(f"test {workflow_name} did not have a node that could be checked for output")

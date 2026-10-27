"""Recent model workflows across memory, attention and multi-device configurations."""
import json
import os
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from comfy.api.components.schema.prompt import Prompt
from comfy.cli_args import default_configuration
from comfy.client.embedded_comfy_client import Comfy
from comfy.distributed.process_pool_executor import ProcessPoolExecutor
from tests.inference_benchmark import benchmark_rows, install_benchmark
from .test_workflows import ResourceMonitor, _prepare_for_workflows, _replace_remote_media_inputs


BASELINES = (
    "moss-image", "fibo2", "qwen-image21-int8-convrot", "flux3", "flux2",
    "flux2-klein", "ideogram4", "krea2", "ltx-2", "minimax-h3-fl2va",
)
CASES = [(name, "default", {}) for name in BASELINES]
CASES += [(name, mode, changes) for name in ("moss-image", "fibo2", "qwen-image21-int8-convrot")
          for mode, changes in (("static-vram", {"disable_dynamic_vram": True}),
                                ("low-vram", {"disable_dynamic_vram": True, "lowvram": True}))]
CASES += [(name, mode, {flag: True, "use_pytorch_cross_attention": False})
          for name in ("moss-image", "flux2")
          for mode, flag in (("sage", "use_sage_attention"), ("kitchen", "use_ck_attention"))]
CASES += [("moss-image", "int8", {})]
CASES += [(name, "tp2", {"tensor_parallel_size": 2})
          for name in ("flux2", "ideogram4", "krea2", "minimax-h3-fl2va")]
CASES += [(name, "pp2", {"pipeline_parallel_size": 2})
          for name in ("flux2", "qwen-image", "minimax-h3-fl2va")]
CASES += [("qwen-image", mode, {flag: 2})
          for mode, flag in (("ulysses2", "ulysses_degree"), ("ring2", "ring_degree"))]
CASES += [("qwen-image-multigpu-cfg", "cfg2", {}), ("flux-compile", "compile", {})]


def install_runtime_probe(benchmark_path, runtime_path, torch_device=None):
    if torch_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = torch_device.split(":", 1)[1]
    from comfy import samplers

    stack = ExitStack()
    stack.enter_context(install_benchmark(benchmark_path))
    original = samplers.calc_cond_batch

    @wraps(original)
    def measured(model, conds, x_in, timestep, model_options):
        outputs = original(model, conds, x_in, timestep, model_options)
        assert all(value is None or torch.isfinite(value).all().item() for value in outputs)
        attachments = model.current_patcher.attachments
        record = {"model": type(model).__name__, "finite": True,
                  "parallel": [name for name in ("tensor_parallel_executor", "pipeline_parallel_executor", "xdit_sequence_parallel_executor")
                               if name in attachments],
                  "cfg_devices": len(model_options.get("multigpu_clones", {}))}
        with Path(runtime_path).open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        return outputs

    stack.enter_context(patch.object(samplers, "calc_cond_batch", measured))
    return stack


def runtime_workflow(name, mode):
    workflow = json.loads(_prepare_for_workflows()[name + "-0.json"].read_text())
    _replace_remote_media_inputs(workflow)
    for node in workflow.values():
        values = node["inputs"]
        if node["class_type"] == "ResolutionSelector":
            values.setdefault("multiple", 8)
            values["megapixels"] = 0.0625
        for field in ("steps", "num_inference_steps", "inference_steps"):
            if field in values:
                values[field] = 2
        for field in ("width", "height"):
            if isinstance(values.get(field), int):
                values[field] = 256
    if name == "flux2-klein":
        for key in ("75:68", "75:69"):
            workflow[key]["inputs"]["value"] = 256
    if name == "ltx-2":
        workflow["92:62"]["inputs"]["value"] = 17
    if name == "moss-image" and mode == "int8":
        workflow["1"]["inputs"]["unet_name"] = "moss_image_sft_int8_convrot_attn.safetensors"
        workflow["2"]["inputs"]["clip_name"] = "qwen3.5_9b_int8_convrot.safetensors"
    return workflow


@pytest.mark.inference
@pytest.mark.asyncio
@pytest.mark.parametrize("name,mode,changes", CASES, ids=[f"{name}-{mode}" for name, mode, _ in CASES])
async def test_runtime_configuration(name, mode, changes, tmp_path, request):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    multiple = mode in {"tp2", "pp2", "ulysses2", "ring2", "cfg2"}
    if multiple and torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    config = default_configuration()
    config.disable_all_custom_nodes = True
    config.tensor_parallel_size = 1
    config.pipeline_parallel_size = 1
    config.use_pytorch_cross_attention = True
    config.torch_device = None if multiple else f"cuda:{torch.cuda.device_count() - 1}"
    config.update(changes)
    benchmark_path = tmp_path / "steps.jsonl"
    runtime_path = tmp_path / "runtime.jsonl"
    workflow = runtime_workflow(name, mode)
    monitor = ResourceMonitor()
    with monitor:
        with ProcessPoolExecutor(max_workers=1, initializer=install_runtime_probe,
                                 initargs=(str(benchmark_path), str(runtime_path), config.torch_device)) as executor:
            async with Comfy(configuration=config, executor=executor) as client:
                outputs = await client.queue_prompt(Prompt.validate(workflow))
    assert outputs
    for result in outputs.values():
        for image in result.get("images", []):
            path = image.get("abs_path")
            if path and Path(path).suffix.lower() in {".png", ".jpg", ".webp"}:
                with Image.open(path) as decoded:
                    decoded.load()
                    assert decoded.width > 0 and decoded.height > 0
    records = [json.loads(line) for line in runtime_path.read_text().splitlines()]
    assert records and all(row["finite"] for row in records)
    expected = {"tp2": "tensor_parallel_executor", "pp2": "pipeline_parallel_executor",
                "ulysses2": "xdit_sequence_parallel_executor", "ring2": "xdit_sequence_parallel_executor"}.get(mode)
    if expected:
        assert any(expected in row["parallel"] for row in records), records
    if mode == "cfg2":
        assert any(row["cfg_devices"] == 2 for row in records), records
    rows = benchmark_rows(benchmark_path)
    assert rows, "no completed second sampling step was observed"
    expected_attention = {"sage": "SageAttention", "kitchen": "Comfy Kitchen"}.get(mode)
    if expected_attention:
        assert any(expected_attention in row["attention"] for row in rows), rows
    for row in rows:
        row.update(peak_cpu_bytes=monitor.peak_cpu_ram, peak_gpu_bytes=monitor.peak_gpu_vram,
                   runtime=records, configuration=mode)
        request.node.user_properties.append(("inference_benchmark", row))

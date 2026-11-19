"""Benchmark INT8 ConvRot models with --guess-settings and OpenTelemetry trace separation."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None


SUITE: list[dict[str, Any]] = [
    {
        "id": "flux3",
        "label": "Flux.3 Image Klein 4B",
        "workflow": "flux3-0.json",
        "generation_args": "1024×1024, 2 steps, CFG 1.0 (er_sde / simple)",
        "overrides": {
            "6": {"width": 1024, "height": 1024},
            "8": {"steps": 2},
        },
    },
    {
        "id": "z_image_turbo",
        "label": "Z-Image Turbo 6B",
        "workflow": "z_image-0.json",
        "generation_args": "1024×1024, 2 steps, CFG 1.0 (euler / simple)",
        "overrides": {
            "16": {"unet_name": "z_image_turbo_int8_convrot.safetensors"},
            "13": {"width": 1024, "height": 1024},
            "3": {"steps": 2},
        },
    },
    {
        "id": "moss_image",
        "label": "MOSS-Image SFT",
        "workflow": "moss-image-0.json",
        "generation_args": "1024×1024, 2 steps, CFG 3.0 (euler / sgm_uniform)",
        "overrides": {
            "1": {"unet_name": "moss_image_sft_int8_convrot.safetensors"},
            "2": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors"},
            "6": {"width": 1024, "height": 1024},
            "8": {"steps": 2},
        },
    },
    {
        "id": "fibo2",
        "label": "Fibo 2.0 Dev",
        "workflow": "fibo2-0.json",
        "generation_args": "1024×1024, 2 steps, CFG 5.0 (euler / Fibo2Scheduler)",
        "overrides": {
            "1": {"unet_name": "fibo2_dev_int8_convrot.safetensors"},
            "6": {"width": 1024, "height": 1024},
            "8": {"steps": 2, "width": 1024, "height": 1024},
        },
    },
    {
        "id": "mage_flow",
        "label": "Mage-Flow",
        "workflow": "mage-flow-0.json",
        "generation_args": "1024×1024, 2 steps, CFG 5.0 (euler / simple)",
        "overrides": {
            "27": {"megapixels": 1.0},
            "12:6": {"steps": 2},
        },
    },
    {
        "id": "ming_image",
        "label": "Ming-Image 0.1 Design",
        "workflow": "ming-image-0.json",
        "generation_args": "1024×1024, 2 steps, CFG 4.0 (euler / simple)",
        "overrides": {
            "80:21": {"steps": 2},
            "80:41": {"width": 1024, "height": 1024},
            "80:42": {"width": 1024, "height": 1024},
        },
    },
    {
        "id": "krea2",
        "label": "Krea 2 Turbo",
        "workflow": "krea2-0.json",
        "generation_args": "1024×1024, 2 steps, CFG 1.0 (res_multistep / normal)",
        "overrides": {
            "37": {"unet_name": "krea2_turbo_int8_convrot.safetensors"},
            "58": {"width": 1024, "height": 1024},
            "3": {"steps": 2},
        },
    },
    {
        "id": "ideogram4",
        "label": "Ideogram 4",
        "workflow": "ideogram4-0.json",
        "generation_args": "1024×1024, 2 steps, CFG 3.0 (res_multistep / Ideogram4Scheduler)",
        "overrides": {
            "37": {"aspect_ratio": "1:1 (Square)", "megapixels": 1.0},
            "98:23": {"unet_name": "ideogram4_int8_convrot.safetensors"},
            "98:154": {"unet_name": "ideogram4_unconditional_int8_convrot.safetensors"},
            "98:14": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors"},
            "98:17": {"steps": 2, "width": 1024, "height": 1024},
        },
    },
    {
        "id": "qwen_image21",
        "label": "Qwen-Image 2.1",
        "workflow": "qwen-image21-int8-convrot-0.json",
        "generation_args": "1024×1024, 2 steps, CFG 4.0 (euler / simple)",
        "overrides": {
            "4": {"resolution": 1024},
            "5": {"steps": 2},
        },
    },
    {
        "id": "joyimage",
        "label": "JoyAI Image Edit",
        "workflow": "joyimage-0.json",
        "generation_args": "512×512, 2 steps, CFG 4.0 (euler / simple)",
        "overrides": {
            "8": {"width": 512, "height": 512},
            "9": {"steps": 2},
        },
    },
    {
        "id": "boogu",
        "label": "Boogu Image Edit",
        "workflow": "boogu-0.json",
        "generation_args": "1024×1024, 2 steps, CFG 1.0 (res_multistep / normal)",
        "overrides": {
            "37": {"unet_name": "boogu_image_edit_int8_convrot.safetensors"},
            "38": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors"},
            "58": {"width": 1024, "height": 1024},
            "3": {"steps": 2},
        },
    },
    {
        "id": "qwen_image_layered",
        "label": "Qwen-Image Layered",
        "workflow": "qwen-image-layered-pipeline-0.json",
        "generation_args": "640×640, 2 layers, 2 steps, CFG 2.5 (euler / simple)",
        "overrides": {
            "83:3": {"steps": 2},
        },
    },
    {
        "id": "marigold_v2_depth",
        "label": "Marigold V2 Depth (Qwen-Image Edit 2509)",
        "workflow": "marigold-v2-depth-0.json",
        "generation_args": "512×512, 2 steps (euler, manual sigmas)",
        "overrides": {
            "2": {"width": 512, "height": 512},
        },
    },
    {
        "id": "minimax_h3_image",
        "label": "MiniMax H3 Image",
        "workflow": "minimax-h3-image-0.json",
        "generation_args": "1024×1024, 2 steps, CFG 1.0 (euler / simple)",
        "overrides": {
            "4": {"width": 1024, "height": 1024},
            "6": {"steps": 2},
        },
    },
    {
        "id": "flux2_dev",
        "label": "Flux.2 Dev 32B",
        "workflow": "flux2-0.json",
        "generation_args": "1024×1024, 2 steps, CFG 1.0 (euler / Flux2Scheduler)",
        "overrides": {
            "12": {"unet_name": "flux2-dev-int8-convrot-simple.safetensors"},
            "47": {"width": 1024, "height": 1024},
            "48": {"steps": 2, "width": 1024, "height": 1024},
        },
    },
    {
        "id": "minimax_h3_fl2va",
        "label": "MiniMax H3 FL2VA (Video+Audio)",
        "workflow": "minimax-h3-fl2va-0.json",
        "generation_args": "1280×704, 124 frames at 24 fps + audio, 2 steps, CFG 1.0 (res_multistep / simple)",
        "overrides": {
            "105:104": {"width": 1280, "height": 704, "length": 124},
            "105:13": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"},
            "105:9": {"steps": 2},
        },
    },
    {
        "id": "pan2_t2v",
        "label": "PAN 2 T2V 29B",
        "workflow": "pan2-t2v-0.json",
        "generation_args": "1248×704, 241 frames at 24 fps, 2 steps, CFG 5.0 (euler / simple)",
        "overrides": {
            "2": {"clip_name": "qwen3.5_9b_int8_convrot.safetensors"},
            "6": {"width": 1248, "height": 704, "length": 241},
            "7": {"steps": 2},
        },
    },
    {
        "id": "wan21_scail2",
        "label": "Wan 2.1 SCAIL-2 14B",
        "workflow": "wan21-scail2-0.json",
        "generation_args": "832×480, 81 frames at 8 fps, 2 steps, CFG 6.0 (euler / normal)",
        "overrides": {
            "1": {"unet_name": "wan2.1_14B_SCAIL_2_int8_convrot.safetensors"},
            "6": {"width": 832, "height": 480, "length": 81},
            "11": {"steps": 2},
        },
    },
    {
        "id": "seedvr2_3b",
        "label": "SeedVR2 3B",
        "workflow": "seedvr2-0.json",
        "generation_args": "512×512 to 2048×2048, 2 steps, CFG 1.0 (euler / simple)",
        "additional_nodes": {
            "upscale": {"class_type": "ImageScaleBy", "inputs": {
                "image": ["2", 0], "upscale_method": "bilinear", "scale_by": 4.0,
            }},
        },
        "overrides": {
            "3": {"resized_images": ["upscale", 0]},
            "10": {"original_resized_images": ["upscale", 0]},
            "6": {"unet_name": "seedvr2_3b_int8_convrot.safetensors"},
            "2": {"width": 512, "height": 512},
            "8": {"steps": 2},
        },
    },
    {
        "id": "yue2",
        "label": "YuE2 3B Music",
        "workflow": "yue2-int8-convrot-0.json",
        "generation_args": "Up to 30 s stereo audio, 2 steps, CFG 1.0 (euler / simple)",
        "overrides": {
            "2": {"max_duration": 30.0},
            "5": {"steps": 2},
        },
    },
    {
        "id": "minimax_music3",
        "label": "MiniMax Music 3",
        "workflow": "minimax-music3-0.json",
        "generation_args": "Up to 30 s stereo audio, 2 steps, basic guider (res_multistep / simple)",
        "overrides": {
            "3": {"max_duration": 30.0},
            "8": {"steps": 2},
        },
    },
    {
        "id": "magi2_t2va",
        "label": "MAGI-2 Preview 114B MoE (Video+Audio)",
        "workflow": "magi2-t2va-0.json",
        "generation_args": "896×512, 125 frames at 12.5 fps + stereo audio, 2 steps (uni_pc_flow / flow_unipc)",
        "overrides": {
            "7": {"width": 896, "height": 512, "seconds": 10.0},
            "12": {"steps": 2},
        },
    },
    {
        "id": "magi2_refiner",
        "label": "MAGI-2 Preview + Refiner (1088p)",
        "workflow": "magi2-i2va-refiner-0.json",
        "generation_args": "896×512, 125 frames at 12.5 fps -> 1920×1088, 249 frames at 25 fps + audio, 2+2 steps (uni_pc_flow / flow_unipc)",
        "overrides": {
            "7": {"width": 896, "height": 512, "seconds": 10.0},
            "12": {"steps": 2},
            "42": {"steps": 2},
        },
    },
]

_LOADER_NODE_TYPES = frozenset({
    "UNETLoader",
    "CLIPLoader",
    "DualCLIPLoader",
    "TripleCLIPLoader",
    "QuadrupleCLIPLoader",
    "VAELoader",
    "CheckpointLoaderSimple",
    "LoraLoader",
    "LoraLoaderModelOnly",
    "ModelPatchLoader",
    "CLIPVisionLoader",
    "ConditioningLoader",
    "CLIPTextEncode",
    "CLIPTextEncodeFlux",
    "CLIPTextEncodeSD3",
    "Magi2ImageToVideo",
    "MiniMaxH3ImageConditioning",
    "YuE2GenerateMusic",
})

_SAMPLER_NODE_TYPES = frozenset({
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
})

_DECODE_NODE_TYPES = frozenset({
    "VAEDecode",
    "VAEDecodeTiled",
    "VAEDecodeAudio",
    "LTXVAudioVAEDecode",
    "PixelLatentToImage",
})


def _mem_available_bytes() -> int:
    try:
        if psutil is not None:
            return int(psutil.virtual_memory().available)
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def _rocm_used_vram_bytes() -> int:
    try:
        for card in Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"):
            return int(card.read_text().strip())
    except Exception:
        pass
    return 0


class PeakMemoryTracker:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_rss_bytes = 0
        self.peak_vram_bytes = 0
        self.start_mem_available_bytes = _mem_available_bytes()
        self.min_mem_available_bytes = self.start_mem_available_bytes
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml_handles: list[Any] = []

    def _poll(self) -> None:
        try:
            import pynvml
        except ImportError:
            pynvml = None
        proc = psutil.Process() if psutil is not None else None
        while not self._stop.is_set():
            try:
                avail = _mem_available_bytes()
                if avail > 0:
                    self.min_mem_available_bytes = min(self.min_mem_available_bytes, avail)
                if proc is not None and psutil is not None:
                    family = [proc] + proc.children(recursive=True)
                    rss = 0
                    for p in family:
                        try:
                            rss += p.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    self.peak_rss_bytes = max(self.peak_rss_bytes, rss)

                if pynvml is not None and self._nvml_handles:
                    total_vram = 0
                    for handle in self._nvml_handles:
                        try:
                            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                            total_vram += int(info.used)
                        except Exception:
                            pass
                    self.peak_vram_bytes = max(self.peak_vram_bytes, total_vram)
                else:
                    rocm_vram = _rocm_used_vram_bytes()
                    if rocm_vram > 0:
                        self.peak_vram_bytes = max(self.peak_vram_bytes, rocm_vram)
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self) -> PeakMemoryTracker:
        try:
            import pynvml

            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
        except Exception:
            self._nvml_handles = []
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def parse_trace_phases(trace_path: Path) -> dict[str, Any]:
    if not trace_path.exists():
        return {}
    load_ms = 0.0
    sampling_node_ms = 0.0
    sampler_invoke_ms = 0.0
    decode_ms = 0.0
    execute_prompt_ms = 0.0
    sampler_spans: list[dict[str, Any]] = []
    node_spans: list[dict[str, Any]] = []
    decode_span_ids: set[str] = set()
    all_spans: list[dict[str, Any]] = []

    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            span = json.loads(line)
        except json.JSONDecodeError:
            continue
        all_spans.append(span)
        name = span.get("name")
        dur = float(span.get("duration_ms") or 0.0)
        attrs = span.get("attributes") or {}
        if name == "Execute Prompt":
            execute_prompt_ms = max(execute_prompt_ms, dur)
        elif name == "Sampler Invoke":
            sampler_invoke_ms += dur
            sampler_spans.append({
                "duration_ms": dur,
                "steps": attrs.get("sampling.steps"),
                "elapsed_seconds": attrs.get("sampling.elapsed_seconds"),
                "seconds_per_step": attrs.get("sampling.seconds_per_step"),
                "sampler": attrs.get("sampler.class"),
            })
        elif name == "Execute Node":
            ct = attrs.get("class_type", "")
            node_spans.append({"class_type": ct, "duration_ms": dur})
            if ct in _LOADER_NODE_TYPES or "Loader" in ct or "Encode" in ct:
                load_ms += dur
            elif ct in _SAMPLER_NODE_TYPES or "Sampler" in ct:
                sampling_node_ms += dur
            elif ct in _DECODE_NODE_TYPES or "Decode" in ct:
                decode_ms += dur
                if span.get("span_id"):
                    decode_span_ids.add(span["span_id"])

    load_in_decode_ms = sum(
        float(s.get("duration_ms") or 0.0)
        for s in all_spans
        if s.get("name") == "Load Models GPU" and s.get("parent_span_id") in decode_span_ids
    )
    decode_ms = max(0.0, decode_ms - load_in_decode_ms)
    # In ComfyUI, lazy model loading + GPU placement often happens inside prepare_sampling
    # at the start of the KSampler / SamplerCustomAdvanced node before Sampler Invoke.
    lazy_load_in_sampler_ms = max(0.0, sampling_node_ms - sampler_invoke_ms)
    total_load_seconds = (load_ms + lazy_load_in_sampler_ms + load_in_decode_ms) / 1000.0
    return {
        "execute_prompt_seconds": execute_prompt_ms / 1000.0,
        "load_and_encode_seconds": total_load_seconds,
        "explicit_loader_nodes_seconds": load_ms / 1000.0,
        "sampler_prepare_load_seconds": lazy_load_in_sampler_ms / 1000.0,
        "sampler_invoke_seconds": sampler_invoke_ms / 1000.0,
        "vae_decode_seconds": decode_ms / 1000.0,
        "sampler_spans": sampler_spans,
        "node_spans": node_spans,
    }


def prepare_workflow(spec: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    wf_path = repo_root / "tests" / "inference" / "workflows" / spec["workflow"]
    workflow = json.loads(wf_path.read_text(encoding="utf-8"))
    workflow = copy.deepcopy(workflow)
    src_img = repo_root / "tests" / "custom_nodes" / "test_data" / "president_official_portrait_hires2-1-1024x1024.jpg"
    input_dir = repo_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    dst_img = input_dir / src_img.name
    if src_img.exists() and not dst_img.exists():
        shutil.copyfile(src_img, dst_img)
    local_img = str(src_img)
    for node in workflow.values():
        if isinstance(node, dict) and isinstance(node.get("inputs"), dict):
            val = node["inputs"].get("value")
            if isinstance(val, str) and val.startswith(("http://", "https://", "pkg://tests.custom_nodes.test_data/")):
                node["inputs"]["value"] = local_img
            img_val = node["inputs"].get("image")
            if isinstance(img_val, str) and img_val.startswith("pkg://tests.custom_nodes.test_data/"):
                node["inputs"]["image"] = src_img.name

    for node_id, node in spec.get("additional_nodes", {}).items():
        if node_id in workflow:
            raise ValueError(f"Duplicate benchmark node {node_id}")
        workflow[node_id] = copy.deepcopy(node)

    for node_id, patch_inputs in spec.get("overrides", {}).items():
        inputs = workflow[node_id]["inputs"]
        if unknown := patch_inputs.keys() - inputs.keys():
            raise ValueError(f"{spec['workflow']} node {node_id} has no inputs {sorted(unknown)}")
        inputs.update(patch_inputs)

    checkpoints: list[str] = []
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        for val in (node.get("inputs") or {}).values():
            if isinstance(val, str) and val.endswith((".safetensors", ".safetensors.index.json", ".gguf", ".ckpt", ".pth")):
                if val not in checkpoints:
                    checkpoints.append(val)
    return workflow, checkpoints


def run_single_model(spec: dict[str, Any], repo_root: Path, out_dir: Path, extra_args: list[str]) -> dict[str, Any]:
    model_dir = out_dir / spec["id"]
    model_dir.mkdir(parents=True, exist_ok=True)
    result_file = model_dir / "result.json"
    workflow, checkpoints = prepare_workflow(spec, repo_root)
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    is_worker = is_distributed and int(os.environ["RANK"]) != 0
    if not is_distributed and result_file.exists():
        try:
            existing = json.loads(result_file.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "success"
                and existing.get("step2_seconds") is not None
                and existing.get("generation_args") == spec["generation_args"]
                and existing.get("extra_args") == extra_args
                and json.loads((model_dir / "workflow.json").read_text(encoding="utf-8")) == workflow
            ):
                return existing
        except (OSError, json.JSONDecodeError):
            pass
    wf_file = model_dir / "workflow.json"
    wf_file.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    trace_file = model_dir / "traces.jsonl"
    steps_file = model_dir / "steps.jsonl"
    log_file = model_dir / "run.log"
    for f in (trace_file, steps_file):
        if f.exists():
            f.unlink()

    runner_code = "\n".join([
        "import os, sys",
        f"sys.path.insert(0, {str(repo_root)!r})",
        "from comfy.component_model import setup as _setup",
        "_orig_post = _setup.setup_post_torch",
        "def _post(cfg):",
        "    _orig_post(cfg)",
        "    from tests.inference_benchmark import install_benchmark",
        "    _setup._bench_patches = install_benchmark(os.environ['COMFY_BENCHMARK_PATH'])",
        "_setup.setup_post_torch = _post",
        "from comfy.cmd.cli import entrypoint",
        "entrypoint()",
    ])
    cmd = [
        sys.executable,
        "-c",
        runner_code,
        "workflows",
        "run",
        str(wf_file),
        "--output-directory",
        str(model_dir),
        "--guess-settings",
        "--disable-all-custom-nodes",
        "--otel-exporter-otlp-endpoint",
        trace_file.resolve().as_uri(),
        "--logging-level",
        "INFO",
        *extra_args,
    ]
    env = os.environ.copy()
    env["COMFY_BENCHMARK_PATH"] = str(steps_file.resolve())
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
    env["HF_HUB_ETAG_TIMEOUT"] = "60"

    started = time.monotonic()
    with PeakMemoryTracker() as tracker, log_file.open("w", encoding="utf-8") as log_stream:
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=log_stream, stderr=subprocess.STDOUT)
        returncode = proc.wait()
    wall_seconds = time.monotonic() - started

    step_rows = [
        json.loads(line)
        for line in steps_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if steps_file.exists() else []
    trace_summary = parse_trace_phases(trace_file)
    step2_seconds = step_rows[-1]["second_step_seconds"] if step_rows else None
    sampler_ids = {node_id for node_id, node in workflow.items() if node["class_type"] in _SAMPLER_NODE_TYPES}
    measured_ids = {str(row["node_id"]) for row in step_rows}
    missing_sampler_ids = sorted(sampler_ids - measured_ids) if not is_worker else []

    unified_delta_gib = max(0.0, (tracker.start_mem_available_bytes - tracker.min_mem_available_bytes) / (1024 ** 3))
    peak_vram_gib = tracker.peak_vram_bytes / (1024 ** 3) if tracker.peak_vram_bytes > 0 else unified_delta_gib

    result = {
        "id": spec["id"],
        "label": spec["label"],
        "workflow": spec["workflow"],
        "generation_args": spec["generation_args"],
        "checkpoints": checkpoints,
        "extra_args": extra_args,
        "status": "success" if returncode == 0 and not missing_sampler_ids else "failed",
        "returncode": returncode,
        "missing_sampler_ids": missing_sampler_ids,
        "wall_seconds": round(wall_seconds, 3),
        "step2_seconds": round(step2_seconds, 4) if step2_seconds is not None else None,
        "iterations_per_second": round(1.0 / step2_seconds, 4) if step2_seconds and step2_seconds > 0 else None,
        "load_and_encode_seconds": round(trace_summary.get("load_and_encode_seconds", 0.0), 3),
        "sampler_invoke_seconds": round(trace_summary.get("sampler_invoke_seconds", 0.0), 3),
        "vae_decode_seconds": round(trace_summary.get("vae_decode_seconds", 0.0), 3),
        "peak_vram_gib": round(peak_vram_gib, 2),
        "peak_rss_gib": round(tracker.peak_rss_bytes / (1024 ** 3), 2),
        "min_mem_available_gib": round(tracker.min_mem_available_bytes / (1024 ** 3), 2),
        "step_measurements": step_rows,
        "trace_summary": trace_summary,
    }
    (model_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="all", help="Comma-separated model IDs or 'all'")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for results and traces")
    parser.add_argument("--repo-root", type=Path, default=None, help="ComfyUI checkout root containing tests/inference/workflows")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra CLI arguments passed to comfyui workflows run")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve() if args.repo_root is not None else Path(__file__).resolve().parents[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = None if args.model == "all" else [m.strip() for m in args.model.split(",") if m.strip()]
    by_id = {s["id"]: s for s in SUITE}
    if selected_ids is not None and (unknown := set(selected_ids) - by_id.keys()):
        parser.error(f"unknown benchmark models: {', '.join(sorted(unknown))}")
    specs = list(SUITE) if selected_ids is None else [by_id[m] for m in selected_ids]

    summary: list[dict[str, Any]] = []
    for spec in specs:
        sys.stdout.write(f"[BENCHMARK] Starting {spec['id']} ({spec['label']}) -> {spec['generation_args']}\n")
        sys.stdout.flush()
        res = run_single_model(spec, repo_root, args.out_dir, args.extra_arg)
        summary.append(res)
        (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        sys.stdout.write(
            f"[BENCHMARK] Finished {spec['id']}: status={res['status']} "
            f"step2={res['step2_seconds']}s load={res['load_and_encode_seconds']}s "
            f"decode={res['vae_decode_seconds']}s peak_vram={res['peak_vram_gib']}GiB\n"
        )
        if res["status"] != "success":
            if res["missing_sampler_ids"]:
                sys.stdout.write(f"[BENCHMARK] Missing step 2 timing for sampler nodes {res['missing_sampler_ids']}\n")
            err_log = (args.out_dir / spec["id"] / "run.log")
            if err_log.exists():
                sys.stdout.write(err_log.read_text(encoding="utf-8", errors="replace")[-1500:] + "\n")
        sys.stdout.flush()
    if any(result["status"] != "success" for result in summary):
        sys.exit(1)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

from scripts.benchmark_custom_node_tp import add_trace_timings, decode_timings
from tests.convrot_benchmark import SUITE, _SAMPLER_NODE_TYPES, parse_trace_phases, prepare_workflow


def test_decode_timings_ignore_progress_and_count_completed_sequences():
    assert decode_timings("rewriting: 8 tokens, 4s, 0.5s/token") == {
        "autoregressive_generated_tokens": None, "autoregressive_decode_seconds_rounded": None,
    }
    assert decode_timings("rewriting: 8 tokens, 4s\nrewriting: 273 tokens in 127s (0.47s/token)\n"
                          "rewriting: 10 tokens in 3s (0.30s/token)") == {
        "autoregressive_generated_tokens": 283, "autoregressive_decode_seconds_rounded": 130,
    }


def test_trace_timings_separate_text_diffusion_and_prefix_preparation(tmp_path):
    directory = tmp_path / "tp2"
    directory.mkdir()
    spans = [
        {"name": "Execute Node", "start_time_unix_nano": 1, "duration_ms": 9000,
         "attributes": {"class_type": "HunyuanImage3TextEncode"}},
        {"name": "Execute Node", "start_time_unix_nano": 11, "duration_ms": 1000,
         "attributes": {"class_type": "HunyuanImage3ModelLoader"}},
        {"name": "Execute Node", "start_time_unix_nano": 12, "duration_ms": 4000,
         "attributes": {"class_type": "HunyuanImage3TextEncode"}},
        {"name": "Execute Node", "start_time_unix_nano": 14, "duration_ms": 3000,
         "attributes": {"class_type": "KSampler"}},
        {"name": "Sampler Invoke", "start_time_unix_nano": 15, "duration_ms": 2500},
    ]
    (directory / "trace.jsonl").write_text("\n".join(json.dumps(span) for span in spans))
    rows = [{"tp": 2, "start_time_unix_nano": 10, "end_time_unix_nano": 20}]
    add_trace_timings(rows, tmp_path)
    assert rows[0]["text_encode_node_seconds"] == 4
    assert rows[0]["model_loader_seconds"] == 1
    assert rows[0]["generator_node_seconds"] == 3
    assert rows[0]["sampler_seconds"] == 2.5



def test_convrot_prepares_two_steps_per_sampler_and_representative_video_inputs(tmp_path):
    (tmp_path / "tests").symlink_to(Path(__file__).resolve().parents[1], target_is_directory=True)
    workflows = {}
    for spec in SUITE:
        workflow, _ = prepare_workflow(spec, tmp_path)
        workflows[spec["id"]] = workflow
        for node in workflow.values():
            if node["class_type"] not in _SAMPLER_NODE_TYPES:
                continue
            inputs = node["inputs"]
            if "sigmas" in inputs:
                inputs = workflow[inputs["sigmas"][0]]["inputs"]
            steps = inputs["steps"] if "steps" in inputs else len(inputs["sigmas"].split(",")) - 1
            assert steps == 2, spec["id"]

    for model, node_id, expected in (
        ("pan2_t2v", "6", (1248, 704, 241)),
        ("minimax_h3_fl2va", "105:104", (1280, 704, 124)),
        ("wan21_scail2", "6", (832, 480, 81)),
    ):
        inputs = workflows[model][node_id]["inputs"]
        assert (inputs["width"], inputs["height"], inputs["length"]) == expected
    for model in ("magi2_t2va", "magi2_refiner"):
        assert workflows[model]["7"]["inputs"]["seconds"] == 10.0
    for model, resize_id in (("joyimage", "8"), ("marigold_v2_depth", "2")):
        inputs = workflows[model][resize_id]["inputs"]
        assert (inputs["width"], inputs["height"]) == (1024, 1024)
    seedvr = workflows["seedvr2_3b"]
    assert seedvr["10"]["inputs"]["original_resized_images"] == seedvr["3"]["inputs"]["resized_images"]
    resize = seedvr[seedvr["3"]["inputs"]["resized_images"][0]]["inputs"]
    source = seedvr[resize["image"][0]]["inputs"]
    assert (source["width"], source["height"], resize["scale_by"]) == (512, 512, 4.0)
    selector = workflows["ideogram4"]["37"]["inputs"]
    assert (selector["aspect_ratio"], selector["megapixels"]) == ("1:1 (Square)", 1.0)


def test_convrot_charges_lazy_vae_load_to_loading(tmp_path):
    spans = [
        {"name": "Load Models GPU", "parent_span_id": "vae", "duration_ms": 3000},
        {"name": "Execute Node", "span_id": "vae", "duration_ms": 5000,
         "attributes": {"class_type": "VAEDecode"}},
    ]
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text("\n".join(json.dumps(span) for span in spans))
    phases = parse_trace_phases(trace_path)
    assert phases["load_and_encode_seconds"] == 3.0
    assert phases["vae_decode_seconds"] == 2.0


def test_convrot_missing_decode_span_has_no_duration(tmp_path):
    trace = tmp_path / "traces.jsonl"
    trace.write_text(json.dumps({"name": "Execute Node", "duration_ms": 1000,
                                 "attributes": {"class_type": "SaveImage"}}))
    assert parse_trace_phases(trace)["vae_decode_seconds"] is None


def test_convrot_resume_requires_output_from_the_recorded_run(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from PIL import Image
    from tests import convrot_benchmark as benchmark

    monkeypatch.setenv("WORLD_SIZE", "1")
    (tmp_path / "tests").symlink_to(Path(__file__).resolve().parents[1], target_is_directory=True)
    preset = next(spec for spec in SUITE if spec["id"] == "flux3")
    output = tmp_path / "output"
    directory = output / preset["id"]
    directory.mkdir(parents=True)
    stale = directory / "previous.png"
    Image.new("RGB", (4, 4)).save(stale)
    calls = []

    def launch(*args, env, **kwargs):
        calls.append(None)
        Path(env["COMFY_BENCHMARK_PATH"]).write_text(json.dumps({"node_id": "8", "second_step_seconds": 1}))
        (directory / "traces.jsonl").write_text(json.dumps({
            "name": "Sampler Invoke", "attributes": {"sampling.steps": 2},
        }))
        if len(calls) == 2:
            media = directory / "images"
            media.mkdir()
            image = Image.new("RGB", preset["expected_image_size"], "white")
            image.putpixel((0, 0), (0, 0, 0))
            image.save(media / "current.png")
        return SimpleNamespace(wait=lambda: 0)

    monkeypatch.setattr(benchmark.subprocess, "Popen", launch)
    monkeypatch.setattr(benchmark.PeakMemoryTracker, "__enter__", lambda self: self)
    monkeypatch.setattr(benchmark.PeakMemoryTracker, "__exit__", lambda *args: None)
    def run():
        return benchmark.run_single_model(preset, tmp_path, output, [])

    assert run()["status"] == "failed"
    good = run()
    assert good["status"] == "success"
    assert good["output_files"] == ["images/current.png"]
    assert run() == good
    assert len(calls) == 2
    legacy = dict(good)
    del legacy["output_files"]
    (directory / "result.json").write_text(json.dumps(legacy))
    assert run() == legacy
    assert len(calls) == 2
    legacy["vae_decode_seconds"] = 0.0
    (directory / "result.json").write_text(json.dumps(legacy))
    assert run() == good
    assert json.loads((directory / "result.json").read_text()) == good
    assert len(calls) == 2
    (directory / "result.json").write_text(json.dumps(dict(good, returncode=1)))
    assert run()["status"] == "failed"
    assert len(calls) == 3
    (directory / "result.json").write_text(json.dumps(good))
    (directory / "images/current.png").unlink()
    assert run()["status"] == "failed"
    assert len(calls) == 4


def test_seedvr_restoration_rejects_flat_and_wrong_size_images(tmp_path):
    from PIL import Image
    from tests.convrot_benchmark import _seedvr_restoration_error

    source = tmp_path / "source.png"
    output = tmp_path / "restored.png"
    image = Image.new("RGB", (64, 64), (30, 40, 50))
    image.paste((225, 210, 180), (32, 0, 64, 64))
    image.save(source)
    workflow = {
        "1": {"inputs": {"value": str(source)}},
        "2": {"inputs": {"width": 64, "height": 64}},
        "upscale": {"inputs": {"scale_by": 4.0}},
    }
    image.resize((256, 256)).save(output)
    assert _seedvr_restoration_error(workflow, output) is None
    assert "Expected a 256×256" in _seedvr_restoration_error(workflow, source)
    partial = image.resize((256, 256))
    partial.paste((0, 0, 0), (0, 0, 256, 64))
    partial.save(output)
    assert "band" in _seedvr_restoration_error(workflow, output)
    Image.new("RGB", (256, 256), (110, 98, 100)).save(output)
    assert "solid fill" in _seedvr_restoration_error(workflow, output)


def test_two_step_measurements_require_both_refiner_stages():
    from tests.convrot_benchmark import _sampling_error

    nodes = {"10", "40"}
    rows = [{"node_id": "10"}, {"node_id": "40"}]
    trace = {"sampler_spans": [{"steps": 2}, {"steps": 2}]}
    assert _sampling_error(nodes, rows, trace) is None
    assert _sampling_error(nodes, rows[:1], trace)
    assert _sampling_error(nodes, rows * 2, trace)
    assert _sampling_error(nodes, rows, {"sampler_spans": [{"steps": 2}, {"steps": 4}]})


def test_animation_validation_accepts_coalesced_frames_and_rejects_timing_drift(tmp_path):
    from PIL import Image
    from tests.convrot_benchmark import _animation_output_error

    first = Image.new("RGB", (8, 8), "white")
    second = Image.new("RGB", (8, 8), "black")
    inputs = {"width": 8, "height": 8, "length": 3}
    output = tmp_path / "sample.webp"
    first.save(output, save_all=True, append_images=[second], duration=[83, 42], lossless=True)
    assert _animation_output_error(output, inputs, 24) is None
    first.save(output, save_all=True, append_images=[second], duration=[82, 41], lossless=True)
    assert _animation_output_error(output, inputs, 24)


def test_still_benchmarks_reject_wrong_resolution_and_missing_layers(tmp_path):
    from PIL import Image
    from tests.convrot_benchmark import _output_error

    specs = {spec["id"]: spec for spec in SUITE}
    files = ["layer-0.png", "layer-1.png", "layer-2.png"]
    for index, name in enumerate(files):
        color = (32, 64, 96, 255) if index == 0 else (0, 0, 0, 0)
        image = Image.new("RGBA", (640, 640), color)
        if index == 0:
            image.putpixel((10, 10), (255, 255, 255, 255))
        image.save(tmp_path / name)
    assert _output_error(specs["flux3"], {}, tmp_path, files[:1])
    layered = specs["qwen_image_layered"]
    assert _output_error(layered, {}, tmp_path, files) is None
    assert _output_error(layered, {}, tmp_path, files[:2])
    Image.new("RGB", (640, 640)).save(tmp_path / files[-1])
    assert _output_error(layered, {}, tmp_path, files)


def test_convrot_matches_seed_and_cfg_to_their_sampler_stage(tmp_path):
    spans = [
        {"name": "Sampler Invoke", "parent_span_id": "preview", "attributes": {"sampling.seed": 0}},
        {"name": "Sampler Invoke", "parent_span_id": "refiner"},
        {"name": "Sampling", "span_id": "refiner", "attributes": {"sampling.cfg": 2, "sampling.seed": 41}},
        {"name": "Sampling", "span_id": "preview", "attributes": {"sampling.cfg": 7, "sampling.seed": 0}},
    ]
    path = tmp_path / "traces.jsonl"
    path.write_text("\n".join(json.dumps(span) for span in spans))
    stages = parse_trace_phases(path)["sampler_spans"]
    assert [(stage["seed"], stage["cfg"]) for stage in stages] == [(0, 7), (41, 2)]


def test_benchmark_rejects_uniform_images(tmp_path):
    from PIL import Image
    from tests.convrot_benchmark import _output_error

    for model, color in (("ideogram4", "black"), ("minimax_h3_image", (123, 116, 103))):
        preset = next(spec for spec in SUITE if spec["id"] == model)
        Image.new("RGB", preset["expected_image_size"], color).save(tmp_path / "sample.png")
        assert _output_error(preset, {}, tmp_path, ["sample.png"]) == "All benchmark images are spatially uniform"


def test_peak_memory_counts_only_worker_visible_nvidia_devices(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tests import convrot_benchmark as benchmark

    # NVML can see the busy desktop GPU even when the worker is restricted to another card.
    used = {"GPU-desktop": 6000, "GPU-worker": 150}
    fake_nvml = SimpleNamespace(
        nvmlDeviceGetHandleByUUID=lambda uuid: uuid,
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=used[handle]),
    )
    monkeypatch.setattr(benchmark, "psutil", None)
    path = tmp_path / "cuda-devices.json"
    for visible in ([], ["GPU-worker"], ["GPU-worker", "GPU-desktop"]):
        path.write_text(json.dumps(visible))
        tracker = benchmark.PeakMemoryTracker(path)
        tracker._nvml = fake_nvml
        monkeypatch.setattr(benchmark.time, "sleep", lambda _: tracker._stop.set())
        tracker._poll()
        assert tracker.peak_vram_bytes == sum(used[uuid] for uuid in visible)


def test_allocator_override_reruns_cached_benchmark(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from PIL import Image
    from unittest.mock import Mock
    from tests import convrot_benchmark as bench

    spec = {"id": "example", "workflow": "example.json", "label": "Example", "generation_args": "two steps"}
    workflow = {"1": {"class_type": "KSampler", "inputs": {}}}
    monkeypatch.setattr(bench, "prepare_workflow", lambda *args: (workflow, []))
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising=False)
    directory = tmp_path / spec["id"]
    directory.mkdir()
    (directory / "workflow.json").write_text(json.dumps(workflow))
    Image.new("RGB", (2, 2)).save(directory / "sample.png")
    cached = {
        "output_files": ["sample.png"], "status": "success", "returncode": 0, "step2_seconds": 1.0,
        "step_measurements": [{"node_id": "1"}], "trace_summary": {"sampler_spans": [{"steps": 2}]},
        "extra_args": [], "generation_args": spec["generation_args"],
    }
    (directory / "result.json").write_text(json.dumps(cached))
    popen = Mock()
    popen.return_value.wait.return_value = 1
    monkeypatch.setattr(bench.subprocess, "Popen", popen)
    tracker = Mock(start_mem_available_bytes=0, min_mem_available_bytes=0, peak_vram_bytes=0, peak_rss_bytes=0)
    monkeypatch.setattr(bench, "PeakMemoryTracker", lambda path: nullcontext(tracker))

    assert bench.run_single_model(spec, tmp_path, tmp_path, []) == cached
    popen.assert_not_called()
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    result = bench.run_single_model(spec, tmp_path, tmp_path, [])
    assert popen.call_args.kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:False"
    assert result["allocator_env"] == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False"}

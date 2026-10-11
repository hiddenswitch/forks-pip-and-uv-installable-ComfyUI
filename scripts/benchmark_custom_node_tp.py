"""Measure an API-format sample workflow in fresh TP1 and TP2 servers.

Prepare workflows with ``comfyui workflows convert`` and disable optional
approximation caches and prompt rewriting for the primary comparison.
"""
from __future__ import annotations

import argparse
import copy
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time
import urllib.request

import psutil
import pynvml


def request(base, path, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def measure_memory(process, devices):
    host = 0
    try:
        processes = [psutil.Process(process.pid), *psutil.Process(process.pid).children(recursive=True)]
        for child in processes:
            try:
                host += child.memory_full_info().pss
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    return host, [pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device)).used for device in devices]


def wait_for_idle_devices(devices, timeout):
    deadline = time.monotonic() + timeout
    while True:
        busy = []
        for device in devices:
            for process in pynvml.nvmlDeviceGetComputeRunningProcesses(pynvml.nvmlDeviceGetHandleByIndex(device)):
                try:
                    name = psutil.Process(process.pid).name()
                except psutil.NoSuchProcess:
                    continue
                if name != "gnome-remote-desktop-daemon":
                    busy.append((device, process.pid, name))
        if not busy:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Other compute jobs still occupy benchmark GPUs: {busy}")
        print(f"Waiting for other GPU jobs: {busy}", flush=True)  # noqa: T201
        time.sleep(30)


def seeded_workflow(workflow, seed):
    result = copy.deepcopy(workflow)
    changed = 0
    for node in result.values():
        for name in ("seed", "noise_seed"):
            if name in node["inputs"] and isinstance(node["inputs"][name], int):
                node["inputs"][name] = seed
                changed += 1
    if not changed:
        raise ValueError("Workflow has no scalar seed input; resolve its seed before benchmarking")
    return result


def add_trace_timings(results, output):
    for size in {row["tp"] for row in results}:
        trace = output / f"tp{size}" / "trace.jsonl"
        spans = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        for result in results:
            if result["tp"] != size:
                continue
            related = [span for span in spans if result["start_time_unix_nano"] <= span["start_time_unix_nano"] <= result["end_time_unix_nano"]]
            samplers = [span for span in related if span["name"] == "Sampler Invoke"]
            result["sampler_seconds"] = sum(span["duration_ms"] for span in samplers) / 1000 if samplers else None
            for key, classes in (("generator_node_seconds", {"KSampler", "Kandinsky6Sampler"}), ("sr_node_seconds", {"Kandinsky6VSRUpscale"})):
                nodes = [span for span in related if span["name"] == "Execute Node" and span.get("attributes", {}).get("class_type") in classes]
                result[key] = sum(span["duration_ms"] for span in nodes) / 1000 if nodes else None


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    workflow = json.loads(args.workflow.read_text())
    if "nodes" in workflow:
        raise ValueError("Convert the UI sample with comfyui workflows convert first")
    seeds = (42,) if args.cold_only else (42, 43, 44, 45)
    for node in workflow.values():
        if node["class_type"] == "Kandinsky6MagCache":
            raise ValueError("Bypass Kandinsky6MagCache for the primary TP comparison")
        if node["inputs"].get("enabled") and (node["class_type"] == "HunyuanImage3Spectrum" or (node["class_type"] == "HunyuanImage3PromptRewriting" and not args.allow_prompt_rewriting)):
            raise ValueError("Disable Spectrum and prompt rewriting for the primary TP comparison")
    pynvml.nvmlInit()
    environment = {
        "packages": {name: importlib.metadata.version(name) for name in ("comfyui", "torch", "comfy-kitchen", "kandinsky6", "kandinsky6-sr", "comfyui-hunyuanimage3")},
        "topology": subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True),
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "model_revisions": json.loads(args.revisions.read_text()),
        "sampling_interval_seconds": 0.2,
        "server_args": args.server_arg,
        "baseline_gpu": args.baseline_gpu,
        "tp_gpus": args.tp_gpus,
    }
    results_path = output / "results.json"
    results = [row for row in json.loads(results_path.read_text()) if row["tp"] not in args.tp_sizes] if results_path.exists() else []
    if results and json.loads((output / "environment.json").read_text()) != environment:
        raise ValueError("Existing benchmark environment differs; use a new output directory")
    for size in {row["tp"] for row in results}:
        if sorted(row["seed"] for row in results if row["tp"] == size) != list(seeds):
            raise ValueError(f"Existing TP{size} results are incomplete; rerun that phase")
        for seed in seeds:
            previous = json.loads((output / f"tp{size}" / f"seed{seed}.prompt.json").read_text())
            if previous != seeded_workflow(workflow, seed):
                raise ValueError("Existing benchmark workflow differs; use a new output directory")
    (output / "environment.json").write_text(json.dumps(environment, indent=2))
    base = f"http://127.0.0.1:{args.port}"
    for size, devices in ((1, [args.baseline_gpu]), (2, args.tp_gpus)):
        if size not in args.tp_sizes:
            continue
        wait_for_idle_devices(devices, args.timeout)
        destination = output / f"tp{size}"
        destination.mkdir(exist_ok=True)
        trace = destination / "trace.jsonl"
        command = [args.comfyui, "serve", "--listen", "127.0.0.1", "--port", str(args.port),
                   "--no-guess-settings", "--cuda-device", ",".join(map(str, devices)),
                   "--tensor-parallel-size", str(size), "--distributed-executor-backend", "mp",
                   "--output-directory", str(destination), "--otel-exporter-otlp-endpoint", trace.as_uri(),
                   *args.server_arg]
        env = dict(os.environ, NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH,P2P,ENV",
                   NCCL_DEBUG_FILE=str(destination / "nccl.%p.log"), HF_HUB_OFFLINE="1")
        (destination / "command.json").write_text(json.dumps({"command": command, "environment": {k: v for k, v in env.items() if k in {"NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE", "HF_HUB_OFFLINE"}}}, indent=2))
        with (destination / "server.log").open("w") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 300
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Server exited: {destination / 'server.log'}")
                    try:
                        request(base, "/system_stats")
                        break
                    except (OSError, ValueError):
                        if time.monotonic() > deadline:
                            raise TimeoutError("Server startup")
                        time.sleep(1)
                for index, seed in enumerate(seeds):
                    prompt = seeded_workflow(workflow, seed)
                    (destination / f"seed{seed}.prompt.json").write_text(json.dumps(prompt, indent=2))
                    start_ns = time.time_ns()
                    started = time.monotonic()
                    response = request(base, "/prompt", {"prompt": prompt})
                    if response.get("node_errors"):
                        raise RuntimeError(response)
                    prompt_id = response["prompt_id"]
                    host_peak, gpu_peak = measure_memory(process, devices)
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError("Server exited during generation")
                        host, gpu = measure_memory(process, devices)
                        host_peak = max(host_peak, host)
                        gpu_peak = [max(a, b) for a, b in zip(gpu_peak, gpu)]
                        history = request(base, "/history/" + prompt_id)
                        if prompt_id in history:
                            break
                        if time.monotonic() - started > args.timeout:
                            raise TimeoutError(prompt_id)
                        time.sleep(0.2)
                    elapsed = time.monotonic() - started
                    entry = history[prompt_id]
                    (destination / f"seed{seed}.history.json").write_text(json.dumps(entry, indent=2))
                    if entry["status"]["status_str"] != "success":
                        raise RuntimeError(entry["status"])
                    result = dict(tp=size, seed=seed, cold=index == 0, job_wall_seconds=elapsed,
                                  peak_host_pss_bytes=host_peak, peak_device_used_bytes=gpu_peak,
                                  start_time_unix_nano=start_ns, end_time_unix_nano=time.time_ns())
                    results.append(result)
                    (output / "results.json").write_text(json.dumps(results, indent=2))
                    print(json.dumps(result), flush=True)  # noqa: T201
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=45)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        if size > 1 and "with tensor-parallel ranks" not in (destination / "server.log").read_text():
            raise RuntimeError("TP2 did not report loading distributed ranks; refusing a fallback benchmark")
    add_trace_timings(results, output)
    (output / "results.json").write_text(json.dumps(results, indent=2))
    with (output / "results.csv").open("w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    medians = {size: statistics.median(r["job_wall_seconds"] for r in results if r["tp"] == size and not r["cold"]) for size in {r["tp"] for r in results if not r["cold"]}}
    speedup = medians[1] / medians[2] if 1 in medians and 2 in medians else None
    cold = {r["tp"]: r["job_wall_seconds"] for r in results if r["cold"]}
    (output / "summary.json").write_text(json.dumps({"cold_seconds": cold, "cold_tp2_speedup": cold[1] / cold[2] if 1 in cold and 2 in cold else None, "warm_median_seconds": medians, "tp2_speedup": speedup}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revisions", type=Path, required=True, help="JSON mapping model artifacts to immutable revisions")
    parser.add_argument("--comfyui", default="comfyui")
    parser.add_argument("--server-arg", action="append", default=[], help="Repeat using --server-arg=VALUE for model paths and common server flags")
    parser.add_argument("--baseline-gpu", type=int, default=1)
    parser.add_argument("--tp-gpus", type=int, nargs=2, default=[0, 1])
    parser.add_argument("--tp-sizes", type=int, nargs="+", choices=(1, 2), default=[1, 2], help="Run selected phases, preserving complete results for the other phase")
    parser.add_argument("--cold-only", action="store_true", help="Run one complete workflow per fresh server, with no warm repeats")
    parser.add_argument("--allow-prompt-rewriting", action="store_true", help="Include complete Hunyuan autoregressive rewriting in the workflow timing")
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--timeout", type=int, default=7200)
    run(parser.parse_args())

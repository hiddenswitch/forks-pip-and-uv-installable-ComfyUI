"""Run bounded, weight-free coverage; reports include all shipped packages."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    "tests/execution/test_execution.py",
    "tests/execution/test_error_logging.py",
    "tests/execution/test_disabled_nodes.py",
    "tests/execution/test_jobs_from_execution.py",
    "tests/unit/test_coverage_processes.py",
    "tests/unit/test_sa_solver.py",
    "tests/unit/test_fast_samplers.py",
    "tests/unit/test_process_pool.py",
    "tests/unit/test_workflow_interrupt.py",
    "tests/unit/test_release_models.py",
    "tests/unit/test_flux3_integration.py",
    "tests/unit/test_fast_model_contracts.py",
    "tests/unit/test_model_downloader_uri.py",
    "tests/unit/test_cache_control.py",
    "tests/unit/test_cache_provider.py",
    "tests/unit/test_distributed_topology.py",
    "tests/unit/test_distributed_worker_configuration.py",
    "tests/unit/test_fast_coverage_runner.py",
    "tests/unit/test_validation.py",
    "tests/unit/test_execution_list.py",
    "tests/inference/test_supported_model_coverage.py",
]


def environment():
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("coverage", "pytest", "pytest-timeout", "torch", "numpy")
        },
        "selection": hashlib.sha256("\n".join(TESTS).encode()).hexdigest(),
    }


def compare_baseline(old, new):
    if old.get("environment") != new["environment"]:
        raise ValueError(
            "Baseline environment or test selection differs; measure a matching baseline"
        )
    if old.get("test_exit_code") != 0 or new["test_exit_code"] != 0:
        raise ValueError("Coverage baselines require successful test runs")
    for covered, total in [
        ("covered_lines", "num_statements"),
        ("covered_branches", "num_branches"),
    ]:
        before, after = old["totals"], new["totals"]
        if before[total] <= 0 or after[total] <= 0:
            raise ValueError("Coverage denominator is empty: " + total)
        if after[covered] * before[total] < before[covered] * after[total]:
            raise ValueError("Coverage regression: " + covered)


def run_command(command, *, env, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Fast coverage exceeded its ten-minute budget")
    with subprocess.Popen(
        command, cwd=ROOT, env=env, start_new_session=os.name == "posix"
    ) as process:
        try:
            return process.wait(timeout=remaining)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # Workers and HTTP servers inherit the new process group.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            raise
        finally:
            # pytest-timeout can exit without running fixtures' server cleanup.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def archive_process_data(output):
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    for previous in raw.glob(".coverage.*"):
        previous.unlink()
    # Reporting commands can auto-combine and delete root-level shards.
    for shard in output.glob(".coverage.*"):
        shard.rename(raw / shard.name)


def check_source_inventory(report):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    packages = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    expected = {
        path.relative_to(ROOT).as_posix()
        for package in packages
        for path in (ROOT / package).rglob("*.py")
    }
    missing = expected - report["files"].keys()
    assert not missing, f"Shipped Python files missing from coverage: {sorted(missing)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".coverage-results/fast")
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Previous fast coverage JSON; reject lower line or branch coverage",
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    current_environment = environment()
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    if baseline is not None and baseline.get("environment") != current_environment:
        parser.error(
            "Baseline environment or test selection differs; measure a matching baseline"
        )
    env = dict(
        os.environ,
        COVERAGE_FILE=str(output / ".coverage"),
        COMFYUI_UNIT_DISABLE_CUSTOM_NODES="1",
        HF_HUB_OFFLINE="1",
        COMFYUI_FAST_OFFLINE="1",
        PYTHONPATH=os.pathsep.join(
            [
                str(ROOT / "tests/fast_startup"),
                str(ROOT),
                os.environ.get("PYTHONPATH", ""),
            ]
        ),
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    deadline = time.monotonic() + 600

    def run(*command):
        return run_command([sys.executable, *command], env=env, deadline=deadline)

    if run("-m", "coverage", "erase"):
        return 1
    result = run(
        "-m",
        "coverage",
        "run",
        "--context=synthetic-cpu",
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "--timeout=30",
        "-ra",
        "--durations=10",
        "-m",
        "not timing and not requires_weights and not requires_network and not requires_accelerator",
        "--junitxml=" + str(output / "tests.xml"),
        *TESTS,
    )
    if run("-m", "coverage", "combine", "--keep"):
        return 1
    archive_process_data(output)
    for command, filename in [("json", "coverage.json"), ("xml", "coverage.xml")]:
        if run("-m", "coverage", command, "-o", str(output / filename)):
            return 1
    if run("-m", "coverage", "html", "-d", str(output / "html")):
        return 1
    # Fail if the real spawned worker body disappeared from the measurement.
    report = json.loads((output / "coverage.json").read_text())
    check_source_inventory(report)
    worker = report["files"]["comfy/distributed/process_pool_executor.py"]
    tree = ast.parse((ROOT / "comfy/distributed/process_pool_executor.py").read_text())
    wrapper = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_wrap_with_context"
    )
    assert {n.lineno for n in wrapper.body} <= set(worker["executed_lines"]), (
        "worker coverage missing"
    )
    report["environment"] = current_environment
    report["test_exit_code"] = result
    (output / "coverage.json").write_text(json.dumps(report))
    if baseline is not None:
        compare_baseline(baseline, report)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

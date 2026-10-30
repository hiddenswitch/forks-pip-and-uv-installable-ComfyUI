from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from coverage import CoverageData


SPEC = importlib.util.spec_from_file_location(
    "fast_coverage", Path(__file__).resolve().parents[2] / "scripts/fast_coverage.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def report():
    return {
        "environment": {"python": "3.14.7", "selection": "same-suite"},
        "test_exit_code": 0,
        "totals": {
            "covered_lines": 30,
            "num_statements": 100,
            "covered_branches": 10,
            "num_branches": 50,
        },
    }


def test_accepts_equal_and_improved_coverage(report):
    runner.compare_baseline(report, deepcopy(report))
    improved = deepcopy(report)
    improved["totals"]["covered_lines"] += 1
    improved["totals"]["covered_branches"] += 1
    runner.compare_baseline(report, improved)


@pytest.mark.parametrize("key", ["covered_lines", "covered_branches"])
def test_rejects_regression(report, key):
    lower = deepcopy(report)
    lower["totals"][key] -= 1
    with pytest.raises(ValueError, match="Coverage regression"):
        runner.compare_baseline(report, lower)


@pytest.mark.parametrize("key", ["python", "selection"])
def test_rejects_unrelated_measurements(report, key):
    changed = deepcopy(report)
    changed["environment"][key] = "different"
    with pytest.raises(ValueError, match="environment or test selection"):
        runner.compare_baseline(report, changed)


@pytest.mark.parametrize("failed_baseline", [False, True])
def test_rejects_failed_test_runs(report, failed_baseline):
    failed = deepcopy(report)
    failed["test_exit_code"] = 1
    with pytest.raises(ValueError, match="successful test runs"):
        runner.compare_baseline(
            failed, report
        ) if failed_baseline else runner.compare_baseline(report, failed)


@pytest.mark.parametrize("key", ["num_statements", "num_branches"])
def test_rejects_empty_denominators(report, key):
    empty = deepcopy(report)
    empty["totals"][key] = 0
    with pytest.raises(ValueError, match="denominator is empty"):
        runner.compare_baseline(report, empty)


def test_command_preserves_failure_status():
    assert runner.run_command(
        [sys.executable, "-c", "raise SystemExit(7)"],
        env=os.environ.copy(),
        deadline=time.monotonic() + 10,
    ) == 7


def test_expired_budget_does_not_launch_command(monkeypatch):
    def unexpected_launch(*args, **kwargs):
        pytest.fail("An expired budget must not start another process")

    monkeypatch.setattr(runner.subprocess, "Popen", unexpected_launch)
    with pytest.raises(TimeoutError, match="ten-minute budget"):
        runner.run_command([], env={}, deadline=time.monotonic() - 1)


def test_reporting_preserves_raw_process_data(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("value = 17\nassert value == 17\n")
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\nbranch = true\nparallel = true\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith("COVERAGE_")}
    env.update(COVERAGE_RCFILE=str(config), COVERAGE_FILE=str(tmp_path / ".coverage"))
    deadline = time.monotonic() + 15
    for arguments in [("run", str(probe)), ("combine", "--keep")]:
        assert runner.run_command(
            [sys.executable, "-m", "coverage", *arguments], env=env, deadline=deadline
        ) == 0
    runner.archive_process_data(tmp_path)
    shards = list((tmp_path / "raw").glob(".coverage.*"))
    assert shards, "The report must retain its source measurement"
    before = {shard: shard.read_bytes() for shard in shards}
    assert runner.run_command(
        [sys.executable, "-m", "coverage", "json", "-o", str(tmp_path / "coverage.json")],
        env=env, deadline=deadline,
    ) == 0
    for shard, contents in before.items():
        assert shard.read_bytes() == contents
        data = CoverageData(basename=str(shard))
        data.read()
        assert set(data.lines(str(probe))) == {1, 2}


def test_inventory_rejects_missing_unimported_source(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.hatch.build.targets.wheel]\npackages = ["package/"]\n'
    )
    source = tmp_path / "package" / "namespace" / "unimported.py"
    source.parent.mkdir(parents=True)
    source.write_text("def uncalled():\n    return 99\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(AssertionError, match="namespace/unimported.py"):
        runner.check_source_inventory({"files": {}})
    runner.check_source_inventory({"files": {"package/namespace/unimported.py": {}}})


@pytest.mark.skipif(sys.platform != "linux", reason="Checks Linux process state")
@pytest.mark.parametrize("parent_exit", [None, 0, 7])
def test_command_kills_spawned_descendants(tmp_path, parent_exit):
    pid_file = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"subprocess.Popen([sys.executable, {str(child)!r}])\n"
        f"while not Path({str(pid_file)!r}).exists():\n"
        "    time.sleep(0.01)\n"
        + ("time.sleep(60)\n" if parent_exit is None else f"raise SystemExit({parent_exit})\n")
    )
    try:
        def run():
            return runner.run_command(
                [sys.executable, str(parent)],
                env=os.environ.copy(),
                deadline=time.monotonic() + 5,
            )
        if parent_exit is None:
            with pytest.raises(subprocess.TimeoutExpired):
                run()
        else:
            assert run() == parent_exit
        assert pid_file.exists(), "The child must start before testing cleanup"
        state = Path("/proc") / pid_file.read_text() / "stat"
        deadline = time.monotonic() + 2
        while state.exists():
            try:
                status = state.read_text().split()[2]
            except FileNotFoundError:
                break
            if status == "Z":
                break
            assert time.monotonic() < deadline, "Timed-out child is still running"
            time.sleep(0.01)
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass

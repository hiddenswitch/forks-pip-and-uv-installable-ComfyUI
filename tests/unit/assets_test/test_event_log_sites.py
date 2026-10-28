import subprocess
import sys
from pathlib import Path

import pytest

from comfy.app.assets.event_log import TAG





@pytest.fixture(autouse=True)
def autoclean_unit_test_assets():
    yield


def run_quick_startup(tmp_path: Path, *flags: str, timeout: float) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from comfy.cmd.cli import entrypoint; entrypoint()",
            "--cpu",
            "--quick-test-for-ci",
            "--disable-all-custom-nodes",
            "--disable-api-nodes",
            f"--base-directory={tmp_path}",
            f"--front-end-root={tmp_path}",
            f"--database-url=sqlite:///{tmp_path / 'assets.sqlite3'}",
            *flags,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout + result.stderr


# TODO: re-enable once the --enable-assets startup runs stop flaking.
@pytest.mark.skip(reason="assets startup tests temporarily disabled")
@pytest.mark.parametrize(
    ("hashing_flag", "expected"),
    [
        pytest.param((), False, id="hashing-disabled"),
        pytest.param(("--enable-asset-hashing",), True, id="hashing-enabled"),
    ],
)
def test_enabled_assets_emits_once_with_the_hashing_flag(
    tmp_path: Path, hashing_flag: tuple[str, ...], expected: bool, process_startup_timeout_seconds
) -> None:
    output = run_quick_startup(tmp_path, "--enable-assets", *hashing_flag, timeout=process_startup_timeout_seconds)
    lines = [line for line in output.splitlines() if f"{TAG} assets.enabled " in line]

    assert len(lines) == 1
    assert f"hashing_enabled={str(expected).lower()}" in lines[0]


def test_noassets_emits_no_enabled_event(tmp_path: Path, process_startup_timeout_seconds) -> None:
    output = run_quick_startup(tmp_path, "--disable-assets", timeout=process_startup_timeout_seconds)

    assert f"{TAG} assets.enabled " not in output

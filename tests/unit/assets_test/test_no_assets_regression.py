import contextlib
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from tests.conftest import server_startup_timeout_seconds


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return server_socket.getsockname()[1]


@pytest.fixture
def no_assets_server(tmp_path: Path):
    for directory in ("models", "custom_nodes", "input", "output", "temp", "user"):
        (tmp_path / directory).mkdir()
    port = _free_port()
    log_path = tmp_path / "server.log"
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from comfy.cmd.cli import entrypoint; entrypoint()",
                "--disable-all-custom-nodes",
                f"--base-directory={tmp_path}",
                "--listen",
                "127.0.0.1",
                "--port",
                str(port),
                "--cpu",
                "--disable-assets",
            ],
            cwd=tmp_path,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + max(120, server_startup_timeout_seconds())
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"No-assets server exited with {process.returncode}\n{log_path.read_text()[-4000:]}")
            with contextlib.suppress(requests.RequestException):
                if requests.get(f"{base_url}/system_stats", timeout=1).status_code == 200:
                    break
            time.sleep(0.25)
        else:
            raise RuntimeError(f"No-assets server did not start\n{log_path.read_text()[-4000:]}")
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def test_no_assets_keeps_legacy_upload_and_view(no_assets_server: str):
    disabled = requests.get(f"{no_assets_server}/api/assets", timeout=10)
    assert disabled.status_code == 503
    assert "--enable-assets" in disabled.json()["error"]["message"]

    upload = requests.post(
        f"{no_assets_server}/upload/image",
        files={"image": ("legacy.png", b"legacy-bytes", "image/png")},
        data={"type": "output"},
        timeout=10,
    )

    assert upload.status_code == 200
    filename = upload.json()["name"]
    view = requests.get(
        f"{no_assets_server}/view",
        params={"filename": filename, "type": "output"},
        timeout=10,
    )
    assert view.status_code == 200
    assert view.content == b"legacy-bytes"

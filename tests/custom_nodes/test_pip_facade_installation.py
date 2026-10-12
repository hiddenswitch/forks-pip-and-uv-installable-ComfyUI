from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Generator

import pytest

_CUSTOM_SCRIPTS_REPO_URL = "https://github.com/pythongosssss/ComfyUI-Custom-Scripts"
_CUSTOM_SCRIPTS_ARCHIVE_URL = "https://github.com/pythongosssss/ComfyUI-Custom-Scripts/archive/refs/heads/main.zip"
_SRC_ROOT = Path(str(files("tests"))).parent


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http(url: str, process: subprocess.Popen[str], timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise AssertionError(f"serve-pip exited early with code {process.returncode}:\n{output}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError:
            time.sleep(0.5)
    output = process.stdout.read() if process.stdout is not None else ""
    raise AssertionError(f"Timed out waiting for {url}:\n{output}")


def _custom_scripts_expected_nodes() -> set[str]:
    import comfyui_manager

    mapping_path = Path(comfyui_manager.__file__).resolve().parent / "extension-node-map.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    class_types = mapping[_CUSTOM_SCRIPTS_REPO_URL][0]
    return {str(item) for item in class_types}


@pytest.fixture(scope="module")
def mock_registry_base_url() -> str:
    class _MockRegistryHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/nodes/comfyui-custom-scripts/versions"):
                payload = [
                    {
                        "version": "1.2.5",
                        "downloadUrl": _CUSTOM_SCRIPTS_ARCHIVE_URL,
                        "dependencies": [],
                        "deprecated": False,
                    }
                ]
                self._write_json(payload)
                return
            if self.path.startswith("/nodes"):
                payload = {
                    "totalPages": 1,
                    "nodes": [
                        {
                            "id": "comfyui-custom-scripts",
                            "name": "ComfyUI-Custom-Scripts",
                            "repository": _CUSTOM_SCRIPTS_REPO_URL,
                            "description": "Scripts",
                            "latest_version": {"version": "1.2.5", "status": "NodeVersionStatusActive"},
                        }
                    ],
                }
                self._write_json(payload)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            del format, args

        def _write_json(self, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockRegistryHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


@pytest.fixture(scope="module")
def facade_snapshot_path(tmp_path_factory: pytest.TempPathFactory, mock_registry_base_url: str) -> Path:
    output_dir = tmp_path_factory.mktemp("facade_snapshot")
    snapshot_path = output_dir / "registry.sqlite.xz"
    from comfy.component_model.configuration import Configuration
    from comfy.custom_node_facade.snapshot import snapshot_facade_registry

    configuration = Configuration()
    configuration.pip_facade_registry_base_url = mock_registry_base_url
    configuration.pip_facade_snapshot_output = str(snapshot_path)
    configuration.pip_facade_snapshot_compression = "auto"
    configuration.pip_facade_snapshot_overwrite = True
    configuration.pip_facade_only_known_nodes = True

    written = asyncio.run(snapshot_facade_registry(configuration))
    assert written == snapshot_path
    return snapshot_path


from dataclasses import dataclass


@dataclass
class FacadeServer:
    base_url: str
    env: dict[str, str]
    site_dir: Path
    src_root: Path = _SRC_ROOT


def _with_extra_site(env: dict[str, str], site_dir: Path) -> dict[str, str]:
    """Expose an installed ``--target`` directory to a probe subprocess."""
    probe_env = env.copy()
    existing = probe_env.get("PYTHONPATH", "")
    probe_env["PYTHONPATH"] = (
        str(site_dir) if not existing else f"{site_dir}{os.pathsep}{existing}"
    )
    return probe_env


@pytest.fixture
def facade_server(tmp_path: Path) -> Generator[FacadeServer, None, None]:
    """Start a serve-pip process and yield its base URL."""
    site_dir = tmp_path / "site"
    site_dir.mkdir()

    port = _find_free_port()
    env = os.environ.copy()

    process = subprocess.Popen(
        [
            sys.executable, "-m", "comfy.cmd.main", "serve-pip",
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--pip-facade-cache-prefix", str(tmp_path / "wheel-cache"),
            "--pip-facade-only-known-nodes",
            "--logging-level", "INFO",
        ],
        cwd=_SRC_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{port}/readyz", process)
        yield FacadeServer(
            base_url=f"http://127.0.0.1:{port}",
            env=env,
            site_dir=site_dir,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _uv_install(server: FacadeServer, *packages: str, no_deps: bool = False, index_url: str | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [
        "uv", "pip", "install",
        "--python", sys.executable,
        "--target", str(server.site_dir),
    ]
    if no_deps:
        cmd.append("--no-deps")
    if index_url:
        cmd.extend(["--index-url", index_url])
    else:
        cmd.extend(["--extra-index-url", f"{server.base_url}/simple/"])
    cmd.extend(packages)
    return subprocess.run(cmd, cwd=server.src_root, env=server.env, check=True, capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required for facade install test")
def test_serve_pip_can_install_and_load_custom_node(facade_server: FacadeServer):
    _uv_install(facade_server, "comfyui-custom-scripts==1.2.5")

    expected_nodes = sorted(_custom_scripts_expected_nodes())
    probe_script = f"""
from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

from comfy.cmd.node_info import node_info
from comfy.nodes.package import _extract_vanilla_custom_node_roots
from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1
from comfy_compatibility.vanilla import prepare_vanilla_environment

prepare_vanilla_environment()

entry_point = next(
    ep for ep in entry_points().select(group='comfyui.custom_nodes')
    if ep.name == 'comfyui-custom-scripts'
)
module = entry_point.load()
roots = _extract_vanilla_custom_node_roots(module)
assert roots, 'facade entry point did not expose any vanilla custom node roots'
repo_path = Path(roots[0]) / 'ComfyUI-Custom-Scripts'
assert repo_path.exists(), repo_path
exported = _vanilla_load_custom_nodes_1(str(repo_path))
keys = set(exported.NODE_CLASS_MAPPINGS.keys())
expected = set({expected_nodes!r})
matched = sorted(keys & expected)
assert matched, f'expected one of {{sorted(expected)}} in loaded keys'
info = node_info(matched[0], exported.NODE_CLASS_MAPPINGS, exported.NODE_DISPLAY_NAME_MAPPINGS)
assert isinstance(info["python_module"], str) and info["python_module"]
print('MATCHED', matched[0])
"""

    probe_env = _with_extra_site(facade_server.env, facade_server.site_dir)
    probe = subprocess.run(
        [sys.executable, "-c", probe_script],
        cwd=_SRC_ROOT,
        env=probe_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "MATCHED" in probe.stdout


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required for facade install test")
def test_serve_pip_can_install_comfyui_layerstyle(facade_server: FacadeServer):
    """Verify that comfyui-layerstyle can be downloaded and installed.

    This catches regressions like missing system tools (zip, sha256sum) in the
    container image that cause 500 errors during on-demand wheel builds.
    """
    _uv_install(facade_server, "comfyui-layerstyle", no_deps=True)

    installed = list(facade_server.site_dir.glob("_appmana_facade_*"))
    assert installed, f"Expected facade package in {facade_server.site_dir}, found: {list(facade_server.site_dir.iterdir())}"


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required for facade install test")
def test_serve_pip_rewrites_image_reward_with_relaxed_timm(facade_server: FacadeServer):
    """Verify that image-reward is served with relaxed timm/fairscale pins."""
    with urllib.request.urlopen(f"{facade_server.base_url}/simple/") as resp:
        index_html = resp.read().decode()
    assert "image-reward" in index_html, "image-reward not listed in simple index"

    _uv_install(facade_server, "image-reward==1.5", no_deps=True, index_url=f"{facade_server.base_url}/simple/")

    dist_info = list(facade_server.site_dir.glob("image_reward-*.dist-info"))
    assert dist_info, f"image-reward dist-info not found in {list(facade_server.site_dir.iterdir())}"
    metadata = (dist_info[0] / "METADATA").read_text(encoding="utf-8")
    assert "timm" in metadata, "timm not in dependencies"
    assert "timm (==0.6.13)" not in metadata, "timm pin was not relaxed"
    assert "fairscale (==0.4.13)" not in metadata, "fairscale pin was not relaxed"


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required for facade install test")
def test_serve_pip_can_install_comfyui_nunchaku_with_nunchaku_dep(facade_server: FacadeServer):
    """Verify that comfyui-nunchaku includes nunchaku as a dependency and that
    nunchaku resolves from the proxied GitHub Pages index."""
    with urllib.request.urlopen(f"{facade_server.base_url}/simple/nunchaku/") as resp:
        nunchaku_html = resp.read().decode()
    assert ".whl" in nunchaku_html, "nunchaku index should contain wheel links"

    _uv_install(facade_server, "comfyui-nunchaku", no_deps=True)

    installed = list(facade_server.site_dir.glob("_appmana_facade_*"))
    assert installed, f"Expected facade package in {facade_server.site_dir}, found: {list(facade_server.site_dir.iterdir())}"

    dist_info = list(facade_server.site_dir.glob("comfyui_nunchaku-*.dist-info"))
    assert dist_info, f"comfyui-nunchaku dist-info not found in {list(facade_server.site_dir.iterdir())}"
    metadata = (dist_info[0] / "METADATA").read_text(encoding="utf-8")
    assert "nunchaku" in metadata.lower(), "nunchaku should be in Requires-Dist"
    assert "darwin" in metadata, "nunchaku dep should have sys_platform != darwin marker"


def test_serve_pip_proxies_insightface_index(facade_server: FacadeServer):
    """The insightface index page must proxy through to the GitHub Pages site
    on the stable-ABI fork. Unlike sageattention/nunchaku, insightface has no
    CUDA dependency and the upstream is a flat URL, but the facade still
    serves it under both default and cu128/cu130 prefixes since the {cuda}
    placeholder is absent from the template (str.format is a no-op).
    """
    with urllib.request.urlopen(f"{facade_server.base_url}/simple/insightface/") as resp:
        insightface_html = resp.read().decode()
    assert ".whl" in insightface_html, "insightface index should contain wheel links"
    assert "appmana.github.io/forks-insightface-stable-abi" in insightface_html or \
           "github.com/AppMana/forks-insightface-stable-abi" in insightface_html, \
           "insightface index should resolve to the AppMana stable-ABI fork"

    for cuda in ("cu128", "cu130"):
        with urllib.request.urlopen(f"{facade_server.base_url}/simple/{cuda}/insightface/") as resp:
            cuda_html = resp.read().decode()
        assert ".whl" in cuda_html, f"insightface {cuda} index should contain wheel links"


@pytest.fixture
def facade_server_all_nodes(tmp_path: Path) -> Generator[FacadeServer, None, None]:
    """Start a serve-pip process without --pip-facade-only-known-nodes."""
    site_dir = tmp_path / "site"
    site_dir.mkdir()

    port = _find_free_port()
    env = os.environ.copy()

    process = subprocess.Popen(
        [
            sys.executable, "-m", "comfy.cmd.main", "serve-pip",
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--pip-facade-cache-prefix", str(tmp_path / "wheel-cache"),
            "--logging-level", "INFO",
        ],
        cwd=_SRC_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_http(f"http://127.0.0.1:{port}/readyz", process)
        yield FacadeServer(
            base_url=f"http://127.0.0.1:{port}",
            env=env,
            site_dir=site_dir,
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required for facade install test")
def test_serve_pip_can_install_comfyui_ltxvideo(facade_server_all_nodes: FacadeServer):
    """Verify that ComfyUI-LTXVideo can be downloaded, installed, and loaded.

    The Lightricks/ComfyUI-LTXVideo repo uses ``master`` as its default branch.
    This test catches regressions where the facade assumes ``main`` and gets a
    404 from GitHub, returning a 500 to the pip client.
    """
    _uv_install(facade_server_all_nodes, "comfyui-ltxvideo", no_deps=True)

    installed = list(facade_server_all_nodes.site_dir.glob("_appmana_facade_*"))
    assert installed, f"Expected facade package in {facade_server_all_nodes.site_dir}, found: {list(facade_server_all_nodes.site_dir.iterdir())}"

    probe_script = """
from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

from comfy.nodes.package import _extract_vanilla_custom_node_roots
from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1
from comfy_compatibility.vanilla import prepare_vanilla_environment

prepare_vanilla_environment()

entry_point = next(
    ep for ep in entry_points().select(group='comfyui.custom_nodes')
    if ep.name == 'comfyui-ltxvideo'
)
module = entry_point.load()
roots = _extract_vanilla_custom_node_roots(module)
assert roots, 'facade entry point did not expose any vanilla custom node roots'
repo_path = Path(roots[0]) / 'ComfyUI-LTXVideo'
assert repo_path.exists(), repo_path
exported = _vanilla_load_custom_nodes_1(str(repo_path))
keys = set(exported.NODE_CLASS_MAPPINGS.keys())
assert 'LTXVImgToVideoConditionOnly' in keys, f'LTXVImgToVideoConditionOnly not found in {sorted(keys)[:10]}'
print('MATCHED LTXVImgToVideoConditionOnly')
"""

    probe_env = _with_extra_site(
        facade_server_all_nodes.env,
        facade_server_all_nodes.site_dir,
    )
    probe = subprocess.run(
        [sys.executable, "-c", probe_script],
        cwd=_SRC_ROOT,
        env=probe_env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "MATCHED" in probe.stdout


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is required for facade install test")
def test_serve_pip_can_install_and_load_custom_node_from_snapshot(
    tmp_path: Path,
    facade_snapshot_path: Path,
):
    site_dir = tmp_path / "site"
    site_dir.mkdir()

    port = _find_free_port()
    env = os.environ.copy()

    process = subprocess.Popen(
        [
            sys.executable, "-m", "comfy.cmd.main", "serve-pip",
            "--listen", "127.0.0.1",
            "--port", str(port),
            "--pip-facade-cache-prefix", str(tmp_path / "wheel-cache"),
            "--pip-facade-only-known-nodes",
            "--pip-facade-snapshot-uri", facade_snapshot_path.as_uri(),
            "--logging-level", "INFO",
        ],
        cwd=_SRC_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        _wait_for_http(f"http://127.0.0.1:{port}/readyz", process)

        subprocess.run(
            [
                "uv", "pip", "install",
                "--python", sys.executable,
                "--target", str(site_dir),
                "--extra-index-url", f"http://127.0.0.1:{port}/simple/",
                "comfyui-custom-scripts==1.2.5",
            ],
            cwd=_SRC_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

        expected_nodes = sorted(_custom_scripts_expected_nodes())
        probe_script = f"""
from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

from comfy.cmd.node_info import node_info
from comfy.nodes.package import _extract_vanilla_custom_node_roots
from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1
from comfy_compatibility.vanilla import prepare_vanilla_environment

prepare_vanilla_environment()

entry_point = next(
    ep for ep in entry_points().select(group='comfyui.custom_nodes')
    if ep.name == 'comfyui-custom-scripts'
)
module = entry_point.load()
roots = _extract_vanilla_custom_node_roots(module)
assert roots, 'facade entry point did not expose any vanilla custom node roots'
repo_path = Path(roots[0]) / 'ComfyUI-Custom-Scripts'
assert repo_path.exists(), repo_path
exported = _vanilla_load_custom_nodes_1(str(repo_path))
keys = set(exported.NODE_CLASS_MAPPINGS.keys())
expected = set({expected_nodes!r})
matched = sorted(keys & expected)
assert matched, f'expected one of {{sorted(expected)}} in loaded keys'
info = node_info(matched[0], exported.NODE_CLASS_MAPPINGS, exported.NODE_DISPLAY_NAME_MAPPINGS)
assert isinstance(info["python_module"], str) and info["python_module"]
print('MATCHED', matched[0])
"""

        probe_env = _with_extra_site(env, site_dir)
        probe = subprocess.run(
            [sys.executable, "-c", probe_script],
            cwd=_SRC_ROOT,
            env=probe_env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "MATCHED" in probe.stdout
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

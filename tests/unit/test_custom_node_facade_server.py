from __future__ import annotations

from aiohttp.test_utils import TestClient, TestServer
import pytest

from comfy.component_model.configuration import Configuration
from comfy.custom_node_facade import server as facade_server
from comfy.custom_node_facade.rocm_index import RocmSimpleIndexProxy, _absolute_links
from comfy.custom_node_facade.snapshot import write_facade_registry_snapshot


class _FakeRegistry:
    calls = 0

    def __init__(self, *args, **kwargs):
        del args, kwargs

    async def list_projects(self):
        type(self).calls += 1
        return []


class _FakeSnapshotRegistry(_FakeRegistry):
    snapshot_uri = None

    def __init__(self, *args, snapshot_uri=None, **kwargs):
        del args, kwargs
        type(self).snapshot_uri = snapshot_uri


class _FakeBuilder:
    def __init__(self, *args, **kwargs):
        del args, kwargs


def test_local_cache_health_checks_usable_directory(tmp_path):
    root = tmp_path / "cache"
    cache = facade_server.FacadeCacheStore(root)
    cache.check_local_storage()
    assert list(root.iterdir()) == []
    root.rmdir()
    root.write_bytes(b"not a directory")
    with pytest.raises(OSError):
        cache.check_local_storage()
    root.unlink()
    cache.check_local_storage()


async def test_snapshot_health_tracks_mount_loss_and_recovers(tmp_path, monkeypatch):
    snapshot = tmp_path / "registry.sqlite"
    config = Configuration()
    config.pip_facade_snapshot_uri = str(snapshot)
    config.pip_facade_snapshot_max_age_seconds = 2100
    monkeypatch.setattr(facade_server, "triton_prewarm_targets", _no_prewarm)

    def write():
        write_facade_registry_snapshot(
            snapshot, projects=[], versions_by_node_id={},
            base_url="https://registry.example.invalid", only_known_nodes=True,
        )

    async with TestClient(TestServer(facade_server.create_facade_app(configuration=config))) as client:
        for path in ("/readyz", "/healthz", "/simple/"):
            response = await client.get(path)
            assert response.status == 503
            assert response.headers["Cache-Control"] == "no-store"
        assert (await client.get("/livez")).status == 200
        write()
        response = await client.get("/readyz")
        assert response.status == 200
        assert (await response.json())["snapshot_created_at"]
        assert (await client.get("/simple/")).status == 200
        snapshot.unlink()
        assert (await client.get("/readyz")).status == 503
        assert (await client.get("/simple/")).status == 503
        assert (await client.get("/livez")).status == 200
        write()
        assert (await client.get("/readyz")).status == 200


async def _no_prewarm(session):
    for item in ():
        yield item


async def test_readiness_detects_disconnected_wheel_cache(tmp_path, monkeypatch):
    config = Configuration()
    config.pip_facade_cache_prefix = str(tmp_path / "cache")
    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "triton_prewarm_targets", _no_prewarm)
    app = facade_server.create_facade_app(configuration=config)
    async with TestClient(TestServer(app)) as client:
        assert (await client.get("/readyz")).status == 200
        cache = app["facade_triton_cache"]
        original = cache.check_local_storage

        def disconnected():
            raise OSError(107, "Transport endpoint is not connected")

        monkeypatch.setattr(cache, "check_local_storage", disconnected)
        response = await client.get("/readyz")
        assert response.status == 503
        assert "Transport endpoint" in (await response.json())["cache_error"]
        assert response.headers["Cache-Control"] == "no-store"
        assert (await client.get("/livez")).status == 200
        monkeypatch.setattr(cache, "check_local_storage", original)
        assert (await client.get("/readyz")).status == 200


class _FakeProxy:
    def __init__(self, *cuda_variants: str):
        self.cuda_variants = set(cuda_variants)
        self.render_calls: list[str] = []

    def supports_cuda(self, cuda: str) -> bool:
        return cuda in self.cuda_variants

    async def render_index(self, session, cuda: str) -> str:
        del session
        self.render_calls.append(cuda)
        return f"<html><body>{cuda}</body></html>"


class _FakeSdistRewrite:
    name = "sam2"
    filename = "sam2-1.1.0.tar.gz"

    def __init__(self):
        self.calls = 0

    async def build_sdist(self, session) -> bytes:
        del session
        self.calls += 1
        return b"rewritten-sam2-sdist"


def test_cache_storage_options_sets_s3_endpoint_only_for_s3_prefix():
    config = Configuration()
    config.pip_facade_cache_s3_endpoint_url = "http://seaweedfs-s3:8333"

    assert facade_server._cache_storage_options(config, "s3://bucket/prefix") == {
        "client_kwargs": {"endpoint_url": "http://seaweedfs-s3:8333"}
    }
    assert facade_server._cache_storage_options(config, "/mnt/seaweedfs/prefix") == {}


async def test_serve_pip_warms_registry_before_reporting_ready(monkeypatch):
    _FakeRegistry.calls = 0
    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "FacadeWheelBuilder", _FakeBuilder)

    app = facade_server.create_facade_app(configuration=Configuration())
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        assert _FakeRegistry.calls == 1

        live = await client.get("/livez")
        assert live.status == 200
        assert await live.json() == {"ok": True, "live": True, "ready": True}

        ready = await client.get("/readyz")
        assert ready.status == 200
        assert await ready.json() == {"ok": True, "live": True, "ready": True}

        health = await client.get("/healthz")
        assert health.status == 200
        assert await health.json() == {"ok": True, "live": True, "ready": True}
    finally:
        await client.close()


def test_rocm_index_rewrites_relative_links_and_preserves_metadata():
    html = '<a data-requires-python="&gt;=3.10" href="torch/torch.whl#sha256=abc">torch</a>'
    rewritten = _absolute_links(html, "https://stable.repo.amd.com/rocm/whl-next/")
    assert 'data-requires-python="&gt;=3.10"' in rewritten
    assert 'href="https://stable.repo.amd.com/rocm/whl-next/torch/torch.whl#sha256=abc"' in rewritten


async def test_rocm_index_proxy_fetches_root_and_project_pages():
    class _Response:
        def __init__(self, body):
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def raise_for_status(self):
            return None

        async def text(self):
            return self.body

    class _Session:
        def get(self, url):
            if url.endswith("whl-next/"):
                return _Response('<a href="torch/">torch</a>')
            return _Response('<a data-variant="gfx1102" href="torch.whl">torch</a>')

    proxy = RocmSimpleIndexProxy()
    root = await proxy.render_root(_Session())
    project = await proxy.render_project(_Session(), "torch")
    assert "https://stable.repo.amd.com/rocm/whl-next/torch/" in root
    assert "https://stable.repo.amd.com/rocm/whl-next/torch/torch.whl" in project
    assert 'data-variant="gfx1102"' in project


async def test_rewritten_pypi_sdist_route_builds_once_and_uses_cache(monkeypatch, tmp_path):
    rewrite = _FakeSdistRewrite()
    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "FacadeWheelBuilder", _FakeBuilder)
    monkeypatch.setattr(
        facade_server,
        "PYPI_SDIST_REWRITE_FILENAME_INDEX",
        {rewrite.filename: rewrite},
    )
    configuration = Configuration()
    configuration.pip_facade_cache_prefix = str(tmp_path)
    app = facade_server.create_facade_app(configuration=configuration)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        url = f"/packages/pypi-rewrite/{rewrite.filename}"
        first = await client.get(url)
        second = await client.get(url)
        assert first.status == 200
        assert second.status == 200
        assert await first.read() == b"rewritten-sam2-sdist"
        assert await second.read() == b"rewritten-sam2-sdist"
        assert rewrite.calls == 1
        assert (await client.get("/packages/pypi-rewrite/unknown.tar.gz")).status == 404
    finally:
        await client.close()


async def test_cuda_simple_index_filters_proxy_projects_by_supported_cuda(monkeypatch):
    _FakeRegistry.calls = 0
    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "FacadeWheelBuilder", _FakeBuilder)
    monkeypatch.setattr(
        facade_server,
        "PYPI_PROXY_INDEX",
        {
            "flash-attn": _FakeProxy("cu118", "cu121", "cu124", "cu126", "cu128", "cu129", "cu130", "cu131", "cu132"),
            "flash-attn-3": _FakeProxy("cu124", "cu126", "cu128", "cu129", "cu130", "cu132"),
            "sageattention": _FakeProxy("cu128", "cu130"),
            "nunchaku": _FakeProxy("cu128", "cu130"),
        },
    )

    app = facade_server.create_facade_app(configuration=Configuration())
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        cu126 = await client.get("/simple/cu126/")
        assert cu126.status == 200
        cu126_html = await cu126.text()
        assert "/simple/cu126/flash-attn/" in cu126_html
        assert "/simple/cu126/flash-attn-3/" in cu126_html
        assert "sageattention" not in cu126_html
        assert "nunchaku" not in cu126_html

        cu131 = await client.get("/simple/cu131/")
        assert cu131.status == 200
        cu131_html = await cu131.text()
        assert "/simple/cu131/flash-attn/" in cu131_html
        assert "flash-attn-3" not in cu131_html
        assert "sageattention" not in cu131_html
        assert "nunchaku" not in cu131_html

        cu130 = await client.get("/simple/cu130/")
        assert cu130.status == 200
        cu130_html = await cu130.text()
        assert "/simple/flash-attn/" in cu130_html
        assert "/simple/flash-attn-3/" in cu130_html
        assert "/simple/sageattention/" in cu130_html
        assert "/simple/nunchaku/" in cu130_html
    finally:
        await client.close()


async def test_cuda_project_pages_reject_unsupported_proxy_cuda(monkeypatch):
    _FakeRegistry.calls = 0
    flash_attn = _FakeProxy("cu118", "cu121", "cu124", "cu126", "cu128", "cu129", "cu130", "cu131", "cu132")
    flash_attn_3 = _FakeProxy("cu124", "cu126", "cu128", "cu129", "cu130", "cu132")
    sageattention = _FakeProxy("cu128", "cu130")
    nunchaku = _FakeProxy("cu128", "cu130")
    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "FacadeWheelBuilder", _FakeBuilder)
    monkeypatch.setattr(
        facade_server,
        "PYPI_PROXY_INDEX",
        {
            "flash-attn": flash_attn,
            "flash-attn-3": flash_attn_3,
            "sageattention": sageattention,
            "nunchaku": nunchaku,
        },
    )

    app = facade_server.create_facade_app(configuration=Configuration())
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        assert (await client.get("/simple/cu126/flash-attn/")).status == 200
        assert (await client.get("/simple/cu126/flash-attn-3/")).status == 200
        assert (await client.get("/simple/cu126/sageattention/")).status == 404
        assert (await client.get("/simple/cu126/nunchaku/")).status == 404
        assert (await client.get("/simple/cu131/flash-attn-3/")).status == 404
        assert (await client.get("/simple/cu130/sageattention/")).status == 200
        assert (await client.get("/simple/cu130/nunchaku/")).status == 200

        assert flash_attn.render_calls == ["cu126"]
        assert flash_attn_3.render_calls == ["cu126"]
        assert sageattention.render_calls == ["cu130"]
        assert nunchaku.render_calls == ["cu130"]
    finally:
        await client.close()


def test_cuda_torch_variants_recognized_and_filter_flash_attention():
    from comfy.custom_node_facade.builder import (
        PYPI_PROXY_INDEX,
        is_index_variant,
    )

    # Combined cuXXXtorchY.Z tokens from the flash-attn snapshot are index
    # variants; a nonexistent torch token and a project name are not.
    assert is_index_variant("cu126torch2.12")
    assert is_index_variant("cu130")
    assert not is_index_variant("cu126torch9.9")
    assert not is_index_variant("flash-attn")

    flash_attn = PYPI_PROXY_INDEX["flash-attn"]
    flash_attn_3 = PYPI_PROXY_INDEX["flash-attn-3"]
    # Only projects that actually have a wheel for the exact CUDA+torch token
    # serve it.
    assert flash_attn.supports_cuda("cu126torch2.12")
    assert not flash_attn_3.supports_cuda("cu118torch2.0")

    body = await_render(flash_attn, "cu126torch2.12")
    wheels = [line for line in body.split("\n") if ".whl" in line]
    assert wheels, "expected wheels for cu126torch2.12"
    assert all("+cu126torch2.12-" in line for line in wheels)


def await_render(proxy, cuda):
    import asyncio

    return asyncio.new_event_loop().run_until_complete(proxy.render_index(None, cuda))


async def test_server_routes_cuda_torch_variant_segment(monkeypatch):
    _FakeRegistry.calls = 0
    flash_attn = _FakeProxy("cu126torch2.12")
    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "FacadeWheelBuilder", _FakeBuilder)
    monkeypatch.setattr(facade_server, "PYPI_PROXY_INDEX", {"flash-attn": flash_attn})

    app = facade_server.create_facade_app(configuration=Configuration())
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        # A real combined variant routes to the project page.
        assert (await client.get("/simple/cu126torch2.12/flash-attn/")).status == 200
        assert flash_attn.render_calls == ["cu126torch2.12"]
        # The combined-variant index lists the project under the variant prefix.
        index = await (await client.get("/simple/cu126torch2.12/")).text()
        assert "/simple/cu126torch2.12/flash-attn/" in index
    finally:
        await client.close()


class _FakeTritonBuilder:
    calls: list = []

    def __init__(self, *args, **kwargs):
        del args, kwargs

    async def build(self, *, served_filename, cuda):
        type(self).calls.append((served_filename, cuda))
        return b"PATCHED_WHEEL_BYTES"


async def test_triton_package_route_builds_and_caches(monkeypatch, tmp_path):
    _FakeRegistry.calls = 0
    _FakeTritonBuilder.calls = []
    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "FacadeWheelBuilder", _FakeBuilder)
    monkeypatch.setattr(facade_server, "TritonWheelBuilder", _FakeTritonBuilder)
    monkeypatch.setattr(facade_server, "PYPI_PROXY_INDEX", {})

    config = Configuration()
    config.pip_facade_cache_prefix = str(tmp_path)
    app = facade_server.create_facade_app(configuration=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        url = "/packages/triton/cu130/triton-3.7.0.post26-cp312-cp312-win_amd64.whl"
        first = await client.get(url)
        assert first.status == 200
        assert await first.read() == b"PATCHED_WHEEL_BYTES"
        # second request is served from cache; the builder is not invoked again
        second = await client.get(url)
        assert await second.read() == b"PATCHED_WHEEL_BYTES"
        assert _FakeTritonBuilder.calls == [
            ("triton-3.7.0.post26-cp312-cp312-win_amd64.whl", "cu130")
        ]
        # non-triton filenames are rejected
        assert (await client.get("/packages/triton/cu130/evil-1.0.whl")).status == 404
    finally:
        await client.close()


async def test_triton_prewarm_builds_into_cache(monkeypatch, tmp_path):
    _FakeRegistry.calls = 0
    _FakeTritonBuilder.calls = []

    async def fake_targets(session, **kwargs):
        del session, kwargs
        yield "triton-3.7.0.post26-cp312-cp312-win_amd64.whl", "cu130"
        yield "triton_windows-3.7.0.post26-cp312-cp312-win_amd64.whl", "cu130"

    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "FacadeWheelBuilder", _FakeBuilder)
    monkeypatch.setattr(facade_server, "TritonWheelBuilder", _FakeTritonBuilder)
    monkeypatch.setattr(facade_server, "triton_prewarm_targets", fake_targets)
    monkeypatch.setattr(facade_server, "PYPI_PROXY_INDEX", {})

    config = Configuration()
    config.pip_facade_cache_prefix = str(tmp_path)
    app = facade_server.create_facade_app(configuration=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        await app["facade_triton_prewarm"]  # let the background prewarm finish
        assert sorted(_FakeTritonBuilder.calls) == [
            ("triton-3.7.0.post26-cp312-cp312-win_amd64.whl", "cu130"),
            ("triton_windows-3.7.0.post26-cp312-cp312-win_amd64.whl", "cu130"),
        ]
        # a request is now served from the prewarmed cache without rebuilding
        _FakeTritonBuilder.calls = []
        resp = await client.get("/packages/triton/cu130/triton-3.7.0.post26-cp312-cp312-win_amd64.whl")
        assert resp.status == 200
        assert await resp.read() == b"PATCHED_WHEEL_BYTES"
        assert _FakeTritonBuilder.calls == []
    finally:
        await client.close()


async def test_serve_pip_uses_snapshot_registry_when_configured(monkeypatch):
    _FakeRegistry.calls = 0
    _FakeSnapshotRegistry.calls = 0
    _FakeSnapshotRegistry.snapshot_uri = None
    monkeypatch.setattr(facade_server, "FacadeRegistry", _FakeRegistry)
    monkeypatch.setattr(facade_server, "SnapshotFacadeRegistry", _FakeSnapshotRegistry)
    monkeypatch.setattr(facade_server, "FacadeWheelBuilder", _FakeBuilder)

    configuration = Configuration()
    configuration.pip_facade_snapshot_uri = "file:///tmp/registry.sqlite.xz"

    app = facade_server.create_facade_app(configuration=configuration)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        assert _FakeRegistry.calls == 0
        assert _FakeSnapshotRegistry.calls == 1
        assert _FakeSnapshotRegistry.snapshot_uri == configuration.pip_facade_snapshot_uri
    finally:
        await client.close()


async def test_alias_request_serves_wheels_named_after_requested_name(monkeypatch, tmp_path):
    """Regression: uv rejects an index that returns a distribution named
    differently from the requested one ("expected distribution for
    comfyui-ipadapter-plus, got distribution for comfyui-ipadapter-plus-fork").
    A project page requested under an alias must list wheels named after the
    requested name, and the built wheel's METADATA Name must match."""
    import io
    import zipfile

    from aiohttp import web

    from comfy.custom_node_facade.registry import FacadeProject, FacadeVersion, canonicalize_project_name

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("NeatNodes-fork/__init__.py", "NODE_CLASS_MAPPINGS = {}\n")

    archive_app = web.Application()

    async def serve_archive(_request):
        return web.Response(body=archive.getvalue(), content_type="application/zip")

    archive_app.router.add_get("/node.zip", serve_archive)
    archive_client = TestClient(TestServer(archive_app))
    await archive_client.start_server()

    project = FacadeProject(
        canonical_name="neat-nodes-fork",
        display_name="Neat Nodes (fork)",
        node_id="neat_nodes_fork",
        repo_url="https://example.invalid/NeatNodes-fork",
        repo_name="NeatNodes-fork",
        description="",
        aliases=("neat-nodes", "neat-nodes-fork"),
        extra_requirements=(),
        skip_requirements=frozenset(),
        depends_on=(),
        latest_version="1.0.0",
    )
    version = FacadeVersion(
        version="1.0.0",
        download_url=str(archive_client.make_url("/node.zip")),
        dependencies=("sentencepiece",),
        deprecated=False,
    )

    class _AliasRegistry:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def list_projects(self):
            return [project]

        async def get_project(self, name):
            if canonicalize_project_name(name) in project.aliases:
                return project
            return None

        async def list_versions(self, _project):
            return [version]

        async def get_version(self, _project, version_str):
            return version if version_str == version.version else None

        async def dependency_project_name(self, dependency_id):
            return canonicalize_project_name(dependency_id)

    monkeypatch.setattr(facade_server, "FacadeRegistry", _AliasRegistry)
    monkeypatch.setattr(facade_server, "PYPI_PROXY_INDEX", {})

    config = Configuration()
    config.pip_facade_cache_prefix = str(tmp_path)
    app = facade_server.create_facade_app(configuration=config)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        # Alias page lists wheels named after the requested name.
        alias_page = await client.get("/simple/neat-nodes/")
        assert alias_page.status == 200
        alias_html = await alias_page.text()
        assert "/packages/neat-nodes/1.0.0/neat_nodes-1.0.0-py3-none-any.whl" in alias_html
        assert "neat_nodes_fork" not in alias_html

        # Canonical page is unchanged.
        canonical_page = await client.get("/simple/neat-nodes-fork/")
        assert canonical_page.status == 200
        assert "neat_nodes_fork-1.0.0-py3-none-any.whl" in await canonical_page.text()

        # The wheel downloaded under the alias carries the alias as its
        # distribution name — this is what uv verifies.
        wheel_response = await client.get("/packages/neat-nodes/1.0.0/neat_nodes-1.0.0-py3-none-any.whl")
        assert wheel_response.status == 200
        wheel_bytes = await wheel_response.read()
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as wheel:
            names = wheel.namelist()
            assert "neat_nodes-1.0.0.dist-info/METADATA" in names
            metadata = wheel.read("neat_nodes-1.0.0.dist-info/METADATA").decode("utf-8")
            assert "Name: neat-nodes" in metadata
            assert "Name: neat-nodes-fork" not in metadata

        # The canonical wheel is also still served under its own name.
        canonical_wheel = await client.get("/packages/neat-nodes-fork/1.0.0/neat_nodes_fork-1.0.0-py3-none-any.whl")
        assert canonical_wheel.status == 200
        with zipfile.ZipFile(io.BytesIO(await canonical_wheel.read())) as wheel:
            metadata = wheel.read("neat_nodes_fork-1.0.0.dist-info/METADATA").decode("utf-8")
            assert "Name: neat-nodes-fork" in metadata
    finally:
        await client.close()
        await archive_client.close()

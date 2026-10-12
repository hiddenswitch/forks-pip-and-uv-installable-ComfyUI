from __future__ import annotations

import asyncio
import concurrent.futures
import io
import os
import shutil
import tarfile
import types
import zipfile
from pathlib import Path

import pytest
import s3fs
from docker.errors import DockerException
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from comfy.custom_node_facade import flash_attention_wheels
from comfy.custom_node_facade.builder import FacadeWheelBuilder
from comfy.custom_node_facade.builder import FlashAttentionProxySpec
from comfy.custom_node_facade.builder import FacadeCacheStore
from comfy.custom_node_facade.builder import PYPI_PROXY_INDEX
from comfy.custom_node_facade.builder import PyPIProxySpec
from comfy.custom_node_facade.builder import PyPISdistRewriteProxySpec
from comfy.custom_node_facade.registry import FacadeProject, FacadeVersion
from comfy.cmd.node_info import node_info
from comfy.nodes.package import _extract_vanilla_custom_node_roots
from comfy.nodes.vanilla_node_importing import _stamp_relative_python_modules


def _make_zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    return buffer.getvalue()


def _make_tar_gz_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path, data in files.items():
            info = tarfile.TarInfo(path)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_sam2_proxy_removes_torch_only_from_isolated_build_requirements():
    proxy = PYPI_PROXY_INDEX["sam2"]
    assert isinstance(proxy, PyPISdistRewriteProxySpec)
    source = _make_tar_gz_bytes(
        {
            "sam2-1.1.0/pyproject.toml": b"""[build-system]\nrequires = [\n    \"setuptools>=61.0\",\n    \"torch>=2.5.1\",\n]\nbuild-backend = \"setuptools.build_meta\"\n""",
            "sam2-1.1.0/setup.py": b"REQUIRED_PACKAGES = ['torch>=2.5.1', 'torchvision>=0.20.1']\n",
        }
    )

    rewritten = proxy.rewrite_sdist(source)

    with tarfile.open(fileobj=io.BytesIO(rewritten), mode="r:gz") as archive:
        pyproject = (
            archive.extractfile("sam2-1.1.0/pyproject.toml").read().decode("utf-8")
        )
        setup_py = archive.extractfile("sam2-1.1.0/setup.py").read().decode("utf-8")
    assert "torch" not in pyproject
    assert "setuptools>=61.0" in pyproject
    assert "'torch>=2.5.1'" in setup_py
    assert "/packages/pypi-rewrite/sam2-1.1.0.tar.gz" in asyncio.run(
        proxy.render_index(None, "cu130")  # type: ignore[arg-type]
    )


from comfy.custom_node_facade.builder import _strip_url_dependency


class _NoNetworkSession:
    def get(self, url: str):
        raise AssertionError(
            f"Flash Attention proxy should not fetch at request time: {url}"
        )


_FLASH_ATTENTION_WHEEL_URLS = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.8.2/flash_attn-2.8.3+cu126torch2.8-cp39-abi3-linux_x86_64.whl",
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.8.2/flash_attn-2.8.3+cu128torch2.8-cp39-abi3-linux_x86_64.whl",
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.8.2/flash_attn_3-3.0.0+cu128torch2.8-cp39-abi3-linux_x86_64.whl",
    "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.29/flash_attn_3-3.0.0+cu124torch2.5-cp39-abi3-linux_x86_64.whl",
)


class SeaweedFSS3Container(DockerContainer):
    def __init__(self):
        super().__init__("chrislusf/seaweedfs:latest")
        self.with_exposed_ports(8333)
        self.with_command("server -ip=0.0.0.0 -dir=/data -s3 -s3.port=8333")
        self.waiting_for(LogMessageWaitStrategy("Start Seaweed S3 API Server"))

    def endpoint_url(self) -> str:
        return f"http://127.0.0.1:{self.get_exposed_port(8333)}"


@pytest.fixture(name="seaweedfs_s3_endpoint")
def fixture_seaweedfs_s3_endpoint():
    if shutil.which("docker") is None:
        pytest.skip("Docker is required for SeaweedFS S3 testcontainer")

    try:
        container = SeaweedFSS3Container()
        container.start()
    except DockerException as exc:
        pytest.skip(f"Docker is unavailable for SeaweedFS S3 testcontainer: {exc}")

    try:
        yield container.endpoint_url()
    finally:
        container.stop()


def test_strip_url_dependency_rewrites_sam2():
    assert (
        _strip_url_dependency("sam2 @ git+https://github.com/facebookresearch/sam2")
        == "sam2"
    )


def test_strip_url_dependency_preserves_plain_requirement():
    assert _strip_url_dependency("requests>=2.28") == "requests>=2.28"


def test_strip_url_dependency_preserves_extras_and_markers():
    result = _strip_url_dependency(
        "pkg[extra1] @ https://example.com/pkg.tar.gz ; python_version >= '3.10'"
    )
    assert result == 'pkg[extra1]; python_version >= "3.10"'


async def test_flash_attention_proxy_filters_packages_by_cuda(monkeypatch):
    monkeypatch.setattr(
        flash_attention_wheels,
        "FLASH_ATTENTION_WHEEL_URLS",
        _FLASH_ATTENTION_WHEEL_URLS,
    )
    session = _NoNetworkSession()
    proxy = FlashAttentionProxySpec(
        name="flash-attn", wheel_project_prefix="flash_attn"
    )

    html = await proxy.render_index(session, "cu128")  # type: ignore[arg-type]

    assert "flash_attn-2.8.3+cu128torch2.8" in html
    assert "flash_attn-2.8.3+cu126torch2.8" not in html
    assert "flash_attn_3-3.0.0+cu128torch2.8" not in html
    assert (
        "github.com/mjun0812/flash-attention-prebuild-wheels/releases/download" in html
    )


async def test_flash_attention_3_proxy_filters_packages_by_distribution(monkeypatch):
    monkeypatch.setattr(
        flash_attention_wheels,
        "FLASH_ATTENTION_WHEEL_URLS",
        _FLASH_ATTENTION_WHEEL_URLS,
    )
    session = _NoNetworkSession()
    proxy = FlashAttentionProxySpec(
        name="flash-attn-3",
        wheel_project_prefix="flash_attn_3",
        cuda_variants=("cu124", "cu126", "cu128", "cu129", "cu130", "cu132"),
    )

    html = await proxy.render_index(session, "cu128")  # type: ignore[arg-type]

    assert "flash_attn_3-3.0.0+cu128torch2.8" in html
    assert "flash_attn-2.8.3+cu128torch2.8" not in html
    assert "flash_attn_3-3.0.0+cu124torch2.5" not in html


async def test_flash_attention_3_proxy_rejects_unsupported_cuda(monkeypatch):
    monkeypatch.setattr(
        flash_attention_wheels,
        "FLASH_ATTENTION_WHEEL_URLS",
        _FLASH_ATTENTION_WHEEL_URLS,
    )
    session = _NoNetworkSession()
    proxy = FlashAttentionProxySpec(
        name="flash-attn-3",
        wheel_project_prefix="flash_attn_3",
        cuda_variants=("cu124", "cu126", "cu128", "cu129", "cu130", "cu132"),
    )

    html = await proxy.render_index(session, "cu121")  # type: ignore[arg-type]

    assert ".whl" not in html


def test_cuda_specific_proxy_support_matrix():
    expected = {
        "flash-attn": {
            "cu118",
            "cu121",
            "cu124",
            "cu126",
            "cu128",
            "cu129",
            "cu130",
            "cu131",
            "cu132",
        },
        "flash-attn-3": {"cu124", "cu126", "cu128", "cu129", "cu130", "cu132"},
        "sageattention": {"cu128", "cu130"},
        "nunchaku": {"cu128", "cu130"},
    }
    checked_cuda_variants = {
        "cu118",
        "cu121",
        "cu124",
        "cu126",
        "cu128",
        "cu129",
        "cu130",
        "cu131",
        "cu132",
    }

    for project_name, supported_cuda_variants in expected.items():
        proxy = PYPI_PROXY_INDEX[project_name]
        assert {
            cuda for cuda in checked_cuda_variants if proxy.supports_cuda(cuda)
        } == supported_cuda_variants

    assert all(
        PYPI_PROXY_INDEX["insightface"].supports_cuda(cuda)
        for cuda in checked_cuda_variants
    )


def test_triton_xpu_proxy_uses_pytorch_xpu_index():
    proxy = PYPI_PROXY_INDEX["triton-xpu"]

    assert isinstance(proxy, PyPIProxySpec)
    assert proxy.supports_cuda("cu130")
    assert (
        proxy.upstream_index_url("cu130")
        == "https://download.pytorch.org/whl/xpu/triton-xpu/"
    )


def test_url_dependency_stripped_from_wheel_metadata(tmp_path: Path):
    project = FacadeProject(
        canonical_name="comfyui-impact-pack",
        display_name="ComfyUI-Impact-Pack",
        node_id="comfyui-impact-pack",
        repo_url="https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        repo_name="ComfyUI-Impact-Pack",
        description="Impact pack",
        aliases=(),
        extra_requirements=(),
        skip_requirements=frozenset({"torch", "comfyui"}),
        depends_on=(),
        latest_version="8.28.2",
    )
    version = FacadeVersion(
        version="8.28.2",
        download_url="https://example.invalid/node.zip",
        dependencies=(
            "segment-anything>=1.0",
            "sam2 @ git+https://github.com/facebookresearch/sam2",
            "ultralytics>=8.0",
        ),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes(
        {"ComfyUI-Impact-Pack/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n"}
    )

    builder = FacadeWheelBuilder(
        session=None, registry=None, cache_prefix=str(tmp_path)
    )  # type: ignore[arg-type]
    wheel_path = str(
        tmp_path / "comfyui-impact-pack" / "comfyui_impact_pack-8.28.2-py3-none-any.whl"
    )

    builder._build_wheel_from_archive(project, version, archive_bytes, wheel_path, [])

    with zipfile.ZipFile(wheel_path) as wheel:
        metadata = wheel.read("comfyui_impact_pack-8.28.2.dist-info/METADATA").decode(
            "utf-8"
        )
        assert "Requires-Dist: sam2" in metadata
        assert "git+https" not in metadata
        assert "Requires-Dist: segment-anything>=1.0" in metadata
        assert "Requires-Dist: ultralytics>=8.0" in metadata


def test_version_constraints_stripped_from_pinned_dependencies(tmp_path: Path):
    project = FacadeProject(
        canonical_name="test-node",
        display_name="Test",
        node_id="test-node",
        repo_url="https://github.com/test/test",
        repo_name="test",
        description="",
        aliases=(),
        extra_requirements=(),
        skip_requirements=frozenset({"torch"}),
        depends_on=(),
        latest_version="1.0.0",
    )
    version = FacadeVersion(
        version="1.0.0",
        download_url="https://example.invalid/node.zip",
        dependencies=(
            "numpy<2",
            "jax>=0.4,<0.8",
            "jaxlib>=0.4,<0.8",
            "opencv-python-headless==4.7.0.72",
            "scipy>=1.0",
        ),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes({"test/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n"})
    builder = FacadeWheelBuilder(
        session=None, registry=None, cache_prefix=str(tmp_path)
    )  # type: ignore[arg-type]
    wheel_path = str(tmp_path / "test-node" / "test_node-1.0.0-py3-none-any.whl")
    builder._build_wheel_from_archive(project, version, archive_bytes, wheel_path, [])

    with zipfile.ZipFile(wheel_path) as wheel:
        metadata = wheel.read("test_node-1.0.0.dist-info/METADATA").decode("utf-8")
        assert "Requires-Dist: numpy\n" in metadata
        assert "numpy<" not in metadata
        assert "Requires-Dist: jax\n" in metadata
        assert "jax>" not in metadata
        assert "Requires-Dist: jaxlib\n" in metadata
        assert "Requires-Dist: opencv-contrib-python-headless\n" in metadata
        assert "opencv-python-headless" not in metadata
        assert "Requires-Dist: scipy>=1.0" in metadata


def test_onnxruntime_expanded_to_platform_variants(tmp_path: Path):
    project = FacadeProject(
        canonical_name="test-onnx-node",
        display_name="Test",
        node_id="test-onnx-node",
        repo_url="https://github.com/test/test",
        repo_name="test",
        description="",
        aliases=(),
        extra_requirements=(),
        skip_requirements=frozenset({"torch"}),
        depends_on=(),
        latest_version="1.0.0",
    )
    version = FacadeVersion(
        version="1.0.0",
        download_url="https://example.invalid/node.zip",
        dependencies=("onnxruntime", "pillow"),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes({"test/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n"})
    builder = FacadeWheelBuilder(
        session=None, registry=None, cache_prefix=str(tmp_path)
    )  # type: ignore[arg-type]
    wheel_path = str(
        tmp_path / "test-onnx-node" / "test_onnx_node-1.0.0-py3-none-any.whl"
    )
    builder._build_wheel_from_archive(project, version, archive_bytes, wheel_path, [])

    with zipfile.ZipFile(wheel_path) as wheel:
        metadata = wheel.read("test_onnx_node-1.0.0.dist-info/METADATA").decode("utf-8")
        assert 'sys_platform == "darwin"' in metadata
        assert "onnxruntime-gpu" in metadata
        assert 'sys_platform == "win32"' in metadata
        assert "Requires-Dist: pillow" in metadata
        bare = [l for l in metadata.splitlines() if l == "Requires-Dist: onnxruntime"]
        assert bare == [], f"bare onnxruntime without marker: {bare}"


def test_pynvml_rewritten_to_nvidia_ml_py(tmp_path: Path):
    """Any custom node that lists pynvml as a requirement should have its
    facade wheel METADATA rewritten to nvidia-ml-py. The two packages ship
    an identical ``pynvml.py`` module, but pynvml adds a .pth hook that
    warns on every import — silencing that is the whole point.
    """
    project = FacadeProject(
        canonical_name="test-pynvml-node",
        display_name="Test",
        node_id="test-pynvml-node",
        repo_url="https://github.com/test/test",
        repo_name="test",
        description="",
        aliases=(),
        extra_requirements=(),
        skip_requirements=frozenset({"torch"}),
        depends_on=(),
        latest_version="1.0.0",
    )
    version = FacadeVersion(
        version="1.0.0",
        download_url="https://example.invalid/node.zip",
        dependencies=("pynvml", "pillow"),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes({"test/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n"})
    builder = FacadeWheelBuilder(
        session=None, registry=None, cache_prefix=str(tmp_path)
    )  # type: ignore[arg-type]
    wheel_path = str(
        tmp_path / "test-pynvml-node" / "test_pynvml_node-1.0.0-py3-none-any.whl"
    )
    builder._build_wheel_from_archive(project, version, archive_bytes, wheel_path, [])

    with zipfile.ZipFile(wheel_path) as wheel:
        metadata = wheel.read("test_pynvml_node-1.0.0.dist-info/METADATA").decode(
            "utf-8"
        )
        assert "Requires-Dist: nvidia-ml-py" in metadata
        assert "Requires-Dist: pillow" in metadata
        bare = [l for l in metadata.splitlines() if l == "Requires-Dist: pynvml"]
        assert bare == [], f"pynvml should have been rewritten, got: {bare}"


def test_injected_project_has_github_archive_version():
    from comfy.custom_node_facade.registry import FacadeRegistry
    from comfy.component_model.node_registry import CustomNodeSpec

    spec = CustomNodeSpec(
        node_id="test-injected-node",
        repo_url="https://github.com/TestOrg/TestRepo",
        display_name="Test Injected Node",
        inject_version="0.1.0",
        git_ref="main",
    )
    registry = FacadeRegistry.__new__(FacadeRegistry)
    registry._versions_cache = {}
    registry._overlay_index = None

    project = registry._build_injected_project(spec)

    assert project.canonical_name == "test-injected-node"
    assert project.latest_version == "0.1.0"
    assert project.repo_url == "https://github.com/TestOrg/TestRepo"

    versions = registry._versions_cache["test-injected-node"]
    assert len(versions) == 1
    assert versions[0].version == "0.1.0"
    assert (
        versions[0].download_url
        == "https://api.github.com/repos/TestOrg/TestRepo/zipball/main"
    )


async def test_int8_fast_is_served_as_injected_facade_project(monkeypatch):
    from comfy.custom_node_facade import registry as registry_module
    from comfy.custom_node_facade.registry import FacadeRegistry

    async def empty_manager_registry(_session):
        return []

    async def empty_cnr_nodes(_self):
        return []

    monkeypatch.setattr(
        registry_module, "_load_manager_registry", empty_manager_registry
    )
    monkeypatch.setattr(FacadeRegistry, "_fetch_cnr_nodes", empty_cnr_nodes)

    registry = FacadeRegistry(session=None, only_known_nodes=True)  # type: ignore[arg-type]
    project = await registry.get_project("comfyui-int8-fast")

    assert project is not None
    assert project.canonical_name == "comfyui-int8-fast"
    assert project.repo_url == "https://github.com/BobJohnson24/ComfyUI-INT8-Fast"
    assert "comfyui-int8-fast" in project.aliases

    versions = await registry.list_versions(project)
    assert len(versions) == 1
    assert versions[0].version == "0.1.0"
    assert (
        versions[0].download_url
        == "https://api.github.com/repos/BobJohnson24/ComfyUI-INT8-Fast/zipball"
    )


def test_github_archive_url_uses_zipball_for_default_branch():
    from comfy.custom_node_facade.registry import FacadeRegistry

    # No ref: uses zipball endpoint that resolves to the repo's default branch
    url = FacadeRegistry._github_archive_url(
        "https://github.com/Lightricks/ComfyUI-LTXVideo"
    )
    assert url == "https://api.github.com/repos/Lightricks/ComfyUI-LTXVideo/zipball"

    # Explicit ref: uses zipball with that ref
    url = FacadeRegistry._github_archive_url(
        "https://github.com/Lightricks/ComfyUI-LTXVideo", "master"
    )
    assert (
        url == "https://api.github.com/repos/Lightricks/ComfyUI-LTXVideo/zipball/master"
    )

    # Strips trailing .git
    url = FacadeRegistry._github_archive_url(
        "https://github.com/Lightricks/ComfyUI-LTXVideo.git"
    )
    assert url == "https://api.github.com/repos/Lightricks/ComfyUI-LTXVideo/zipball"


def test_extract_vanilla_custom_node_roots(tmp_path: Path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_c = tmp_path / "c"
    root_a.mkdir()
    root_b.mkdir()
    root_c.mkdir()

    module = types.ModuleType("facade_marker")
    module.COMFYUI_VANILLA_NODE_PATH = str(root_a)
    module.COMFYUI_VANILLA_NODE_PATHS = [str(root_a), str(root_b)]
    module.get_vanilla_custom_node_paths = lambda: [str(root_c)]

    assert _extract_vanilla_custom_node_roots(module) == [
        str(root_a.resolve()),
        str(root_b.resolve()),
        str(root_c.resolve()),
    ]


def test_build_wheel_from_archive_injects_entrypoint_and_metadata(tmp_path: Path):
    project = FacadeProject(
        canonical_name="comfyui-wanvideowrapper",
        display_name="ComfyUI-WanVideoWrapper",
        node_id="comfyui-wanvideowrapper",
        repo_url="https://github.com/kijai/ComfyUI-WanVideoWrapper",
        repo_name="ComfyUI-WanVideoWrapper",
        description="Wrapper nodes",
        aliases=("comfyui-wanvideowrapper",),
        extra_requirements=("mediapipe<=0.10.21",),
        skip_requirements=frozenset({"torch", "comfyui"}),
        depends_on=("ComfyUI-KJNodes",),
        latest_version="3.3.3",
    )
    version = FacadeVersion(
        version="3.3.3",
        download_url="https://example.invalid/node.zip",
        dependencies=(
            "requests>=2",
            "torch",
            "numpy<2",
            "opencv-python-headless==4.7.0.72",
        ),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes(
        {
            "ComfyUI-WanVideoWrapper/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n",
            "ComfyUI-WanVideoWrapper/helper.py": b"value = 1\n",
        }
    )

    builder = FacadeWheelBuilder(
        session=None, registry=None, cache_prefix=str(tmp_path)
    )  # type: ignore[arg-type]
    wheel_path = str(
        tmp_path
        / "comfyui-wanvideowrapper"
        / "comfyui_wanvideowrapper-3.3.3-py3-none-any.whl"
    )

    built_path = builder._build_wheel_from_archive(
        project,
        version,
        archive_bytes,
        wheel_path,
        ["comfyui-kjnodes"],
    )

    assert built_path.cache_path == wheel_path
    assert os.path.normpath(built_path.local_path or "") == os.path.normpath(wheel_path)
    assert Path(wheel_path).exists()

    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        assert "_appmana_facade_comfyui_wanvideowrapper/entrypoint.py" in names
        assert (
            "_appmana_facade_comfyui_wanvideowrapper/_vendor/ComfyUI-WanVideoWrapper/__init__.py"
            in names
        )

        metadata_name = "comfyui_wanvideowrapper-3.3.3.dist-info/METADATA"
        entry_points_name = "comfyui_wanvideowrapper-3.3.3.dist-info/entry_points.txt"

        metadata = wheel.read(metadata_name).decode("utf-8")
        assert "Name: comfyui-wanvideowrapper" in metadata
        assert "Version: 3.3.3" in metadata
        assert "Requires-Dist: requests>=2" in metadata
        assert "Requires-Dist: mediapipe<=0.10.21" in metadata
        assert "Requires-Dist: comfyui-kjnodes" in metadata
        assert "Requires-Dist: torch" not in metadata
        assert "Requires-Dist: numpy<2" not in metadata
        assert "Requires-Dist: numpy\n" in metadata
        assert "Requires-Dist: opencv-python-headless==4.7.0.72" not in metadata
        assert "Requires-Dist: opencv-contrib-python-headless\n" in metadata

        entry_points = wheel.read(entry_points_name).decode("utf-8")
        assert "[comfyui.custom_nodes]" in entry_points
        assert (
            "comfyui-wanvideowrapper = _appmana_facade_comfyui_wanvideowrapper.entrypoint"
            in entry_points
        )

        entrypoint_module = wheel.read(
            "_appmana_facade_comfyui_wanvideowrapper/entrypoint.py"
        ).decode("utf-8")
        assert "COMFYUI_VANILLA_NODE_PATHS" in entrypoint_module


def test_build_wheel_to_memory_fs_prefix(tmp_path: Path):
    project = FacadeProject(
        canonical_name="comfyui-custom-scripts",
        display_name="ComfyUI-Custom-Scripts",
        node_id="comfyui-custom-scripts",
        repo_url="https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
        repo_name="ComfyUI-Custom-Scripts",
        description="Scripts",
        aliases=("comfyui-custom-scripts",),
        extra_requirements=(),
        skip_requirements=frozenset({"torch"}),
        depends_on=(),
        latest_version="1.2.5",
    )
    version = FacadeVersion(
        version="1.2.5",
        download_url="https://example.invalid/node.zip",
        dependencies=(),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes(
        {"ComfyUI-Custom-Scripts/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n"}
    )

    builder = FacadeWheelBuilder(
        session=None, registry=None, cache_prefix="memory://facade-cache"
    )  # type: ignore[arg-type]
    wheel = builder._build_wheel_from_archive(
        project,
        version,
        archive_bytes,
        "facade-cache/comfyui-custom-scripts/comfyui_custom_scripts-1.2.5-py3-none-any.whl",
        [],
    )

    assert wheel.local_path is None
    payload = builder._cache.read_bytes(wheel.cache_path)  # type: ignore[attr-defined]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert (
            "_appmana_facade_comfyui_custom_scripts/entrypoint.py" in archive.namelist()
        )


def test_facade_cache_store_handles_windows_local_parent(tmp_path: Path):
    builder = FacadeWheelBuilder(
        session=None, registry=None, cache_prefix=str(tmp_path)
    )  # type: ignore[arg-type]
    cache = builder._cache  # type: ignore[attr-defined]

    windows_style_path = str(tmp_path / "facade" / "wheel.whl").replace("/", "\\")

    assert cache._parent(windows_style_path).endswith("\\facade")


def test_facade_cache_store_filesystem_copy_from_installs_complete_file(tmp_path: Path):
    """Filesystem cache writes are installed with temp+rename semantics, so the
    destination is always one complete source, never a mixture or truncation."""
    import threading

    builder = FacadeWheelBuilder(
        session=None, registry=None, cache_prefix=str(tmp_path)
    )  # type: ignore[arg-type]
    cache = builder._cache  # type: ignore[attr-defined]

    size = 2 * 1024 * 1024
    workers = 8
    # Each worker copies a source that is entirely one distinct byte value, so
    # any interleave of two writers is detectable as a destination containing
    # more than one byte value.
    sources = []
    for i in range(workers):
        src = tmp_path / f"src-{i}.bin"
        src.write_bytes(bytes([i + 1]) * size)
        sources.append(str(src))

    dest = cache.wheel_path(  # a realistic nested cache path
        types.SimpleNamespace(canonical_name="comfyui-kjnodes"),
        "comfyui_kjnodes-1.4.7-py3-none-any.whl",
    )

    barrier = threading.Barrier(workers)

    def worker(src: str) -> None:
        barrier.wait()
        cache.copy_from(src, dest)

    threads = [threading.Thread(target=worker, args=(s,)) for s in sources]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    data = Path(dest).read_bytes()
    assert len(data) == size, f"truncated/torn write: {len(data)} != {size}"
    distinct = set(data)
    assert len(distinct) == 1, (
        f"interleaved write: destination mixes {len(distinct)} byte values"
    )
    assert (distinct.pop() - 1) in range(workers)

    # No temp siblings left behind after a clean run.
    leftovers = list(Path(dest).parent.glob(".facade-*.tmp"))
    assert leftovers == [], f"leftover temp files: {leftovers}"


def test_facade_cache_store_write_bytes_is_atomic(tmp_path: Path):
    builder = FacadeWheelBuilder(
        session=None, registry=None, cache_prefix=str(tmp_path)
    )  # type: ignore[arg-type]
    cache = builder._cache  # type: ignore[attr-defined]

    dest = cache.custom_path(
        "triton", "cu130", "triton-3.7.0-cp312-cp312-win_amd64.whl"
    )
    cache.write_bytes(dest, b"PK\x03\x04payload")
    assert Path(dest).read_bytes() == b"PK\x03\x04payload"
    assert list(Path(dest).parent.glob(".facade-*.tmp")) == []


def test_facade_cache_store_object_cache_writes_final_key_without_rename(
    tmp_path: Path, monkeypatch
):
    builder = FacadeWheelBuilder(
        session=None, registry=None, cache_prefix="memory://facade-object-cache"
    )  # type: ignore[arg-type]
    cache = builder._cache  # type: ignore[attr-defined]
    cache._is_object_cache = (
        True  # exercise S3-style object-store behavior without s3fs
    )

    def fail_mv(*_args, **_kwargs):
        raise AssertionError("object cache writes must not rename or move temp objects")

    monkeypatch.setattr(cache._fs, "mv", fail_mv)

    source = tmp_path / "source.whl"
    source.write_bytes(b"PK\x03\x04wheel")
    dest = cache.wheel_path(
        types.SimpleNamespace(canonical_name="comfyui-kjnodes"),
        "comfyui_kjnodes-1.4.7-py3-none-any.whl",
    )

    cache.copy_from(str(source), dest)
    assert cache.read_bytes(dest) == b"PK\x03\x04wheel"

    cache.write_bytes(dest, b"PK\x03\x04replacement")
    assert cache.read_bytes(dest) == b"PK\x03\x04replacement"


def test_facade_cache_store_seaweedfs_s3_concurrent_puts_publish_complete_objects(
    tmp_path: Path,
    seaweedfs_s3_endpoint: str,
):
    """The production SeaweedFS cache backend should be the S3-compatible API.

    Concurrent writers to the same generated wheel key are duplicate work, but
    the final object must always be one complete PUT payload, never a mixed or
    truncated wheel.
    """
    bucket = "facade-concurrent-put-test"
    storage_options = {
        "anon": True,
        "client_kwargs": {"endpoint_url": seaweedfs_s3_endpoint},
    }
    s3fs.S3FileSystem(**storage_options).mkdir(bucket)
    cache = FacadeCacheStore(
        f"s3://{bucket}/pip-facade", storage_options=storage_options
    )

    size = 1024 * 1024
    workers = 8
    sources: list[Path] = []
    for i in range(workers):
        source = tmp_path / f"source-{i}.whl"
        source.write_bytes(bytes([i + 1]) * size)
        sources.append(source)

    dest = cache.wheel_path(
        types.SimpleNamespace(canonical_name="comfyui-kjnodes"),
        "comfyui_kjnodes-1.4.7-py3-none-any.whl",
    )

    for _ in range(3):
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda source: cache.copy_from(str(source), dest), sources))

        payload = cache.read_bytes(dest)
        distinct = set(payload)
        assert len(payload) == size
        assert len(distinct) == 1
        assert (distinct.pop() - 1) in range(workers)


def test_stamp_relative_python_modules_uses_class_module():
    class ExampleNode:
        __module__ = "vendor.custom_scripts.nodes"
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"
        CATEGORY = "tests"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    _stamp_relative_python_modules({"ExampleNode": ExampleNode})

    assert (
        getattr(ExampleNode, "RELATIVE_PYTHON_MODULE", None)
        == "vendor.custom_scripts.nodes"
    )


def test_node_info_python_module_falls_back_to_class_module():
    class ExampleNode:
        __module__ = "vendor.custom_scripts.nodes"
        RELATIVE_PYTHON_MODULE = None
        ESSENTIALS_CATEGORY = "Tests"
        RETURN_TYPES = ("STRING",)
        FUNCTION = "run"
        CATEGORY = "tests"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}

    info = node_info("ExampleNode", {"ExampleNode": ExampleNode}, {})

    assert info["python_module"] == "vendor.custom_scripts.nodes"
    assert info["essentials_category"] == "Tests"


async def test_cnr_only_node_becomes_facade_project(monkeypatch):
    from comfy.custom_node_facade import registry as registry_module
    from comfy.custom_node_facade.registry import FacadeRegistry

    async def empty_manager_registry(_session):
        return []

    async def cnr_nodes(_self):
        return [
            {
                "id": "qwen-whitebg-detector",
                "name": "qwen-whitebg-detector",
                "repository": "https://github.com/Holonica-Development-Team/QwenWhiteBgDetector",
                "description": "White background detector",
                "status": "NodeStatusActive",
                "latest_version": {"version": "1.0.0", "status": "NodeVersionStatusActive"},
            },
            {
                "id": "banned-node",
                "repository": "https://github.com/example/banned",
                "status": "NodeStatusBanned",
            },
            {
                "id": "gguf",
                "repository": "https://github.com/calcuis/gguf",
                "status": "NodeStatusActive",
            },
        ]

    monkeypatch.setattr(
        registry_module, "_load_manager_registry", empty_manager_registry
    )
    monkeypatch.setattr(FacadeRegistry, "_fetch_cnr_nodes", cnr_nodes)

    registry = FacadeRegistry(session=None)  # type: ignore[arg-type]

    project = await registry.get_project("qwen-whitebg-detector")
    assert project is not None
    assert project.canonical_name == "qwen-whitebg-detector"
    assert project.node_id == "qwen-whitebg-detector"
    assert (
        project.repo_url
        == "https://github.com/Holonica-Development-Team/QwenWhiteBgDetector"
    )
    assert project.latest_version == "1.0.0"
    assert "qwenwhitebgdetector" in project.aliases

    assert await registry.get_project("banned-node") is None
    assert await registry.get_project("gguf") is None


async def test_flagged_registry_latest_version_is_not_a_declared_latest(monkeypatch):
    """The registry reports a flagged version as a node's latest_version, but
    the versions the facade serves are only the active and pending ones."""
    from comfy.custom_node_facade import registry as registry_module
    from comfy.custom_node_facade.registry import FacadeRegistry
    from comfy.custom_node_facade.snapshot import _validated_installable_projects

    async def manager_registry(_session):
        return [
            {
                "title": "Anthropic Claude",
                "reference": "https://github.com/alexmunteanu/comfyui-anthropic-claude",
                "description": "",
            }
        ]

    def flagged_latest(node_id: str, version: str) -> dict:
        return {
            "id": "ffbd422c-f87f-4119-b869-3ead9ffa6a6d",
            "node_id": node_id,
            "version": version,
            "status": "NodeVersionStatusFlagged",
            "downloadUrl": "",
            "dependencies": [],
            "deprecated": False,
        }

    async def cnr_nodes(_self):
        return [
            {
                "id": "anthropic-claude",
                "name": "Anthropic Claude",
                "repository": "https://github.com/alexmunteanu/comfyui-anthropic-claude",
                "status": "NodeStatusActive",
                "latest_version": flagged_latest("anthropic-claude", "1.5.25"),
            },
            {
                "id": "4a-prompt-manager",
                "name": "4A Prompt Manager",
                "repository": "https://github.com/tsukino4a/ComfyUI-4A-Prompt-Manager",
                "status": "NodeStatusActive",
                "latest_version": flagged_latest("4a-prompt-manager", "1.4.4"),
            },
            {
                "id": "qwen-whitebg-detector",
                "name": "qwen-whitebg-detector",
                "repository": "https://github.com/Holonica-Development-Team/QwenWhiteBgDetector",
                "status": "NodeStatusActive",
                "latest_version": {
                    "version": "1.0.0",
                    "status": "NodeVersionStatusActive",
                    "downloadUrl": "",
                },
            },
        ]

    installable = {
        "anthropic-claude": "1.5.22",
        "4a-prompt-manager": "1.4.3",
        "qwen-whitebg-detector": "1.0.0",
    }

    class _VersionsResponse:
        def __init__(self, node_id: str):
            self._node_id = node_id

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            version = installable[self._node_id]
            return [
                {
                    "version": version,
                    "status": "NodeVersionStatusActive",
                    "downloadUrl": f"https://cdn.comfy.org/example/{self._node_id}/{version}/node.zip",
                    "dependencies": [],
                    "deprecated": False,
                }
            ]

    class _VersionsSession:
        def get(self, url, params=None):
            assert params == [
                ("statuses", "NodeVersionStatusActive"),
                ("statuses", "NodeVersionStatusPending"),
            ]
            return _VersionsResponse(url.split("/nodes/")[1].removesuffix("/versions"))

    monkeypatch.setattr(registry_module, "_load_manager_registry", manager_registry)
    monkeypatch.setattr(FacadeRegistry, "_fetch_cnr_nodes", cnr_nodes)

    registry = FacadeRegistry(session=_VersionsSession())  # type: ignore[arg-type]
    projects = await registry.list_projects()
    by_node_id = {project.node_id: project for project in projects}
    assert by_node_id["anthropic-claude"].latest_version is None
    assert by_node_id["4a-prompt-manager"].latest_version is None
    assert by_node_id["qwen-whitebg-detector"].latest_version == "1.0.0"

    versions = {
        project.node_id: await registry.list_versions(project) for project in projects
    }
    assert _validated_installable_projects(projects, versions) == projects


async def test_registry_id_wins_name_collision_and_displaced_project_rekeys(
    monkeypatch,
):
    from comfy.custom_node_facade import registry as registry_module
    from comfy.custom_node_facade.registry import FacadeRegistry

    async def manager_registry(_session):
        return [
            {
                "title": "LongCat Avatar (smthemex)",
                "reference": "https://github.com/smthemex/ComfyUI_LongCat_Avatar",
                "description": "",
            }
        ]

    async def cnr_nodes(_self):
        return [
            {
                "id": "longcat_avatar",
                "name": "longcat_avatar",
                "repository": "https://github.com/smthemex/ComfyUI_LongCat_Avatar",
                "status": "NodeStatusActive",
                "latest_version": {"version": "0.2.0", "status": "NodeVersionStatusActive"},
            },
            {
                "id": "comfyui-longcat-avatar",
                "name": "comfyui-longcat-avatar",
                "repository": "https://github.com/rookiestar28/ComfyUI-LongCat-Avatar",
                "status": "NodeStatusActive",
                "latest_version": {"version": "0.2.0", "status": "NodeVersionStatusActive"},
            },
        ]

    monkeypatch.setattr(registry_module, "_load_manager_registry", manager_registry)
    monkeypatch.setattr(FacadeRegistry, "_fetch_cnr_nodes", cnr_nodes)

    registry = FacadeRegistry(session=None)  # type: ignore[arg-type]

    # The registry node with this exact id owns the name.
    winner = await registry.get_project("comfyui-longcat-avatar")
    assert winner is not None
    assert winner.repo_url == "https://github.com/rookiestar28/ComfyUI-LongCat-Avatar"
    assert winner.node_id == "comfyui-longcat-avatar"

    # The manager-derived project is re-keyed to its own registry id.
    displaced = await registry.get_project("longcat-avatar")
    assert displaced is not None
    assert displaced.repo_url == "https://github.com/smthemex/ComfyUI_LongCat_Avatar"
    assert displaced.node_id == "longcat_avatar"
    assert "comfyui-longcat-avatar" not in displaced.aliases


async def test_manager_fallback_does_not_poison_registry_owner_versions(monkeypatch):
    """A stale Manager repo with the same basename must not replace versions
    belonging to the authoritative Comfy Registry project."""
    from comfy.custom_node_facade import registry as registry_module
    from comfy.custom_node_facade.registry import FacadeRegistry

    async def manager_registry(_session):
        return [
            {
                "title": "ComfyUI BAGEL",
                "reference": "https://github.com/neverbiasu/ComfyUI-BAGEL",
                "description": "authoritative",
            },
            {
                "title": "Old ComfyUI BAGEL fork",
                "reference": "https://github.com/Yuan-ManX/ComfyUI_Bagel",
                "description": "stale manager-only collision",
            },
        ]

    async def cnr_nodes(_self):
        return [
            {
                "id": "comfyui-bagel",
                "name": "ComfyUI-BAGEL",
                "repository": "https://github.com/neverbiasu/ComfyUI-BAGEL",
                "status": "NodeStatusActive",
                "latest_version": {"version": "1.1.0", "status": "NodeVersionStatusActive"},
            }
        ]

    class _VersionsResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def raise_for_status(self):
            return None

        async def json(self):
            return [
                {
                    "version": "1.1.0",
                    "downloadUrl": "https://cdn.comfy.org/faych/comfyui-bagel/1.1.0/node.zip",
                    "dependencies": [],
                    "deprecated": False,
                }
            ]

    class _VersionsSession:
        def get(self, url, params=None):
            assert url.endswith("/nodes/comfyui-bagel/versions")
            assert params is not None
            return _VersionsResponse()

    monkeypatch.setattr(registry_module, "_load_manager_registry", manager_registry)
    monkeypatch.setattr(FacadeRegistry, "_fetch_cnr_nodes", cnr_nodes)

    registry = FacadeRegistry(session=_VersionsSession())  # type: ignore[arg-type]
    project = await registry.get_project("comfyui-bagel")
    assert project is not None
    assert project.repo_url == "https://github.com/neverbiasu/ComfyUI-BAGEL"

    versions = await registry.list_versions(project)
    assert [item.version for item in versions] == ["1.1.0"]
    assert "Yuan-ManX" not in versions[0].download_url


async def test_manager_only_gitlab_project_served(monkeypatch):
    from comfy.custom_node_facade import registry as registry_module
    from comfy.custom_node_facade.registry import FacadeRegistry

    async def manager_registry(_session):
        return [
            {
                "title": "Pixaroma",
                "reference": "https://gitlab.com/pixaroma/comfyui-pixaroma",
                "description": "Pixaroma nodes",
            }
        ]

    async def empty_cnr_nodes(_self):
        return []

    monkeypatch.setattr(registry_module, "_load_manager_registry", manager_registry)
    monkeypatch.setattr(FacadeRegistry, "_fetch_cnr_nodes", empty_cnr_nodes)

    registry = FacadeRegistry(session=None)  # type: ignore[arg-type]

    project = await registry.get_project("comfyui-pixaroma")
    assert project is not None
    assert project.repo_url == "https://gitlab.com/pixaroma/comfyui-pixaroma"

    versions = await registry.list_versions(project)
    assert len(versions) == 1
    assert versions[0].download_url == (
        "https://gitlab.com/api/v4/projects/pixaroma%2Fcomfyui-pixaroma/repository/archive.zip?sha=HEAD"
    )


def test_gguf_stripped_from_wheel_metadata(tmp_path: Path):
    project = FacadeProject(
        canonical_name="comfyui-gguf",
        display_name="ComfyUI-GGUF",
        node_id="comfyui-gguf",
        repo_url="https://github.com/city96/ComfyUI-GGUF",
        repo_name="ComfyUI-GGUF",
        description="GGUF loaders",
        aliases=(),
        extra_requirements=(),
        skip_requirements=frozenset({"torch", "comfyui"}),
        depends_on=(),
        latest_version="1.1.7",
    )
    version = FacadeVersion(
        version="1.1.7",
        download_url="https://example.invalid/node.zip",
        dependencies=("gguf>=0.13.0", "sentencepiece"),
        deprecated=False,
    )
    archive_bytes = _make_zip_bytes(
        {"ComfyUI-GGUF/__init__.py": b"NODE_CLASS_MAPPINGS = {}\n"}
    )

    builder = FacadeWheelBuilder(
        session=None, registry=None, cache_prefix=str(tmp_path)
    )  # type: ignore[arg-type]
    wheel_path = str(tmp_path / "comfyui-gguf" / "comfyui_gguf-1.1.7-py3-none-any.whl")

    builder._build_wheel_from_archive(project, version, archive_bytes, wheel_path, [])

    with zipfile.ZipFile(wheel_path) as wheel:
        metadata = wheel.read("comfyui_gguf-1.1.7.dist-info/METADATA").decode("utf-8")
        requires = [
            line for line in metadata.splitlines() if line.startswith("Requires-Dist:")
        ]
        assert requires == ["Requires-Dist: sentencepiece"]


# ---------------------------------------------------------------------------
# GithubReleaseWheelProxySpec (the `comfyui` package served from GH releases)
# ---------------------------------------------------------------------------
class _GithubResp:
    def __init__(self, obj):
        self._obj = obj
        self.headers_seen = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return self._obj


class _GithubSession:
    def __init__(self, obj):
        self._obj = obj
        self.last_headers = None
        self.last_url = None
        self.calls = 0

    def get(self, url, headers=None):
        self.calls += 1
        self.last_url = url
        self.last_headers = headers
        return _GithubResp(self._obj)


async def test_comfyui_proxy_lists_release_wheels():
    from comfy.custom_node_facade.builder import (
        GithubReleaseWheelProxySpec,
        PYPI_PROXY_INDEX,
        _GITHUB_RELEASE_INDEX_CACHE,
        _GITHUB_RELEASE_INDEX_LOCKS,
    )

    _GITHUB_RELEASE_INDEX_CACHE.clear()
    _GITHUB_RELEASE_INDEX_LOCKS.clear()

    # registered as a default proxy project
    assert "comfyui" in PYPI_PROXY_INDEX

    releases = [
        {
            "assets": [
                {
                    "name": "comfyui-0.24.0.4-py3-none-any.whl",
                    "browser_download_url": "https://github.com/o/r/releases/download/v0.24.0.4/comfyui-0.24.0.4-py3-none-any.whl",
                },
                {
                    "name": "comfyui-0.24.0.4.tar.gz",
                    "browser_download_url": "https://github.com/o/r/releases/download/v0.24.0.4/comfyui-0.24.0.4.tar.gz",
                },
                {
                    "name": "some-other-thing.zip",
                    "browser_download_url": "https://github.com/o/r/releases/download/v0.24.0.4/some-other-thing.zip",
                },
            ]
        },
        {
            "assets": [
                {
                    "name": "comfyui-0.24.0.3-py3-none-any.whl",
                    "browser_download_url": "https://github.com/o/r/releases/download/v0.24.0.3/comfyui-0.24.0.3-py3-none-any.whl",
                },
            ]
        },
    ]
    proxy = GithubReleaseWheelProxySpec(
        name="comfyui", repo="o/r", asset_prefix="comfyui-"
    )
    session = _GithubSession(releases)

    # CUDA-agnostic: in every plain index, not the flash-attn torch indexes
    assert proxy.supports_cuda("cu130")
    assert not proxy.supports_cuda("cu130torch2.12")

    body = await proxy.render_index(session, "cu130")
    cached_body = await proxy.render_index(session, "cu128")
    assert "comfyui-0.24.0.4-py3-none-any.whl" in body
    assert "comfyui-0.24.0.4.tar.gz" in body
    assert "comfyui-0.24.0.3-py3-none-any.whl" in body
    # non-comfyui asset filtered out
    assert "some-other-thing.zip" not in body
    # queries the configured repo
    assert "/repos/o/r/releases" in session.last_url
    assert cached_body == body
    assert session.calls == 1


async def test_manager_name_collision_resolves_to_registry_owner(monkeypatch):
    """Regression for the CI failure where /simple/comfyui-ipadapter-plus/
    served pamparamm's fork: two manager entries whose repo basenames
    canonicalize identically must resolve so that the registry node owning the
    exact id gets the name and the fork keeps its own registry id."""
    from comfy.custom_node_facade import registry as registry_module
    from comfy.custom_node_facade.registry import FacadeRegistry

    async def manager_registry(_session):
        return [
            {
                "title": "ComfyUI IPAdapter plus",
                "reference": "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
                "description": "",
            },
            {
                "title": "ComfyUI IPAdapter plus fork",
                "reference": "https://github.com/pamparamm/ComfyUI_IPAdapter_plus",
                "description": "",
            },
        ]

    async def cnr_nodes(_self):
        return [
            {
                "id": "comfyui_ipadapter_plus",
                "name": "ComfyUI_IPAdapter_plus",
                "repository": "https://github.com/cubiq/ComfyUI_IPAdapter_plus",
                "status": "NodeStatusActive",
                "latest_version": {"version": "2.0.0", "status": "NodeVersionStatusActive"},
            },
            {
                "id": "comfyui_ipadapter_plus_fork",
                "name": "ComfyUI_IPAdapter_plus_fork",
                "repository": "https://github.com/pamparamm/ComfyUI_IPAdapter_plus",
                "status": "NodeStatusActive",
                "latest_version": {"version": "2.0.1", "status": "NodeVersionStatusActive"},
            },
        ]

    monkeypatch.setattr(registry_module, "_load_manager_registry", manager_registry)
    monkeypatch.setattr(FacadeRegistry, "_fetch_cnr_nodes", cnr_nodes)

    registry = FacadeRegistry(session=None)  # type: ignore[arg-type]

    plus = await registry.get_project("comfyui-ipadapter-plus")
    assert plus is not None
    assert plus.repo_url == "https://github.com/cubiq/ComfyUI_IPAdapter_plus"
    assert plus.node_id == "comfyui_ipadapter_plus"

    fork = await registry.get_project("comfyui-ipadapter-plus-fork")
    assert fork is not None
    assert fork.repo_url == "https://github.com/pamparamm/ComfyUI_IPAdapter_plus"
    assert fork.node_id == "comfyui_ipadapter_plus_fork"

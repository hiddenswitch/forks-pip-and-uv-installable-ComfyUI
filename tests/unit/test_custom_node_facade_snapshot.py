from __future__ import annotations

import json
import lzma
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fsspec
import pytest

from comfy.custom_node_facade.registry import (
    FacadeProject,
    FacadeVersion,
    SnapshotFacadeRegistry,
    SnapshotUnavailableError,
    is_excluded_facade_project,
    _normalize_pep440_versions,
    _sort_versions,
)
from comfy.custom_node_facade.snapshot import (
    _build_class_type_rows,
    _collect_versions,
    _load_previous_snapshot,
    _validated_installable_projects,
    write_facade_registry_snapshot,
)

from .facade_storage import (  # noqa: F401  pylint: disable=unused-import
    fixture_facade_storage_url,
    fixture_seaweedfs_s3_endpoint,
)


def _sample_project() -> FacadeProject:
    return FacadeProject(
        canonical_name="comfyui-custom-scripts",
        display_name="ComfyUI-Custom-Scripts",
        node_id="comfyui-custom-scripts",
        repo_url="https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
        repo_name="ComfyUI-Custom-Scripts",
        description="Scripts",
        aliases=("comfyui-custom-scripts", "pysssss"),
        extra_requirements=("deepdiff",),
        skip_requirements=frozenset({"torch", "numpy"}),
        depends_on=("comfyui-kjnodes",),
        latest_version="1.2.5",
    )


def _sample_version() -> FacadeVersion:
    return FacadeVersion(
        version="1.2.5",
        download_url="https://example.invalid/node.zip",
        dependencies=("requests>=2",),
        deprecated=False,
    )


async def test_snapshot_age_limit_is_optional(tmp_path: Path):
    output = tmp_path / "registry.sqlite"
    project = _sample_project()
    write_facade_registry_snapshot(
        output, projects=[project], versions_by_node_id={project.node_id: [_sample_version()]},
        base_url="https://registry.example.invalid", only_known_nodes=True,
    )
    with closing(sqlite3.connect(output)) as connection, connection:
        connection.execute("UPDATE metadata SET value = '2000-01-01T00:00:00+00:00' WHERE key = 'created_at'")
    registry = SnapshotFacadeRegistry(snapshot_uri=str(output))
    assert await registry.get_project(project.node_id) == project


def test_write_facade_registry_snapshot_sqlite(tmp_path: Path):
    output = tmp_path / "registry.sqlite"
    project = _sample_project()
    version = _sample_version()

    written = write_facade_registry_snapshot(
        output,
        projects=[project],
        versions_by_node_id={project.node_id: [version]},
        base_url="https://registry.example.invalid",
        only_known_nodes=True,
    )

    assert written == output
    assert output.exists()

    with sqlite3.connect(output) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        assert metadata["format"] == "appmana-comfyui-pip-facade-registry-snapshot"
        assert metadata["registry_base_url"] == "https://registry.example.invalid"
        assert metadata["only_known_nodes"] == "1"

        project_row = connection.execute(
            """
            SELECT display_name, aliases_json, extra_requirements_json, skip_requirements_json, depends_on_json, latest_version
            FROM projects
            WHERE canonical_name = ?
            """,
            (project.canonical_name,),
        ).fetchone()
        assert project_row is not None
        assert project_row[0] == project.display_name
        assert json.loads(project_row[1]) == list(project.aliases)
        assert json.loads(project_row[2]) == list(project.extra_requirements)
        assert json.loads(project_row[3]) == sorted(project.skip_requirements)
        assert json.loads(project_row[4]) == list(project.depends_on)
        assert project_row[5] == project.latest_version

        version_row = connection.execute(
            "SELECT version, download_url, dependencies_json, deprecated FROM versions WHERE node_id = ?",
            (project.node_id,),
        ).fetchone()
        assert version_row == (
            version.version,
            version.download_url,
            json.dumps(list(version.dependencies), separators=(",", ":")),
            0,
        )


def test_write_facade_registry_snapshot_xz(tmp_path: Path):
    output = tmp_path / "registry.sqlite.xz"
    project = _sample_project()
    version = _sample_version()

    written = write_facade_registry_snapshot(
        output,
        projects=[project],
        versions_by_node_id={project.node_id: [version]},
        base_url="https://registry.example.invalid",
        only_known_nodes=False,
        compression="auto",
    )

    assert written == output
    assert output.exists()

    extracted = tmp_path / "extracted.sqlite"
    with lzma.open(output, "rb") as source, extracted.open("wb") as destination:
        destination.write(source.read())

    with sqlite3.connect(extracted) as connection:
        assert connection.execute("SELECT COUNT(*) FROM projects").fetchone() == (1,)


def test_sort_versions_tolerates_non_pep440_versions():
    versions = [
        FacadeVersion(
            version="0.8.17-bugfix",
            download_url="https://example.invalid/b",
            dependencies=(),
            deprecated=False,
        ),
        FacadeVersion(
            version="1.2.0",
            download_url="https://example.invalid/a",
            dependencies=(),
            deprecated=False,
        ),
        FacadeVersion(
            version="1.1.0",
            download_url="https://example.invalid/c",
            dependencies=(),
            deprecated=False,
        ),
    ]

    sorted_versions = _sort_versions(versions)

    assert [item.version for item in sorted_versions] == [
        "1.2.0",
        "1.1.0",
        "0.8.17-bugfix",
    ]


def test_normalize_pep440_versions_preserves_suffixed_registry_releases():
    versions = [
        FacadeVersion(
            version="1.2.0",
            download_url="https://example.invalid/a",
            dependencies=(),
            deprecated=False,
        ),
        FacadeVersion(
            version="0.8.17-bugfix",
            download_url="https://example.invalid/b",
            dependencies=(),
            deprecated=False,
        ),
    ]

    normalized = _normalize_pep440_versions(versions)

    assert [item.version for item in normalized] == ["1.2.0", "0.8.17+bugfix"]


async def test_snapshot_registry_reads_plain_sqlite(tmp_path: Path):
    output = tmp_path / "registry.sqlite"
    project = _sample_project()
    version = _sample_version()
    write_facade_registry_snapshot(
        output,
        projects=[project],
        versions_by_node_id={project.node_id: [version]},
        base_url="https://registry.example.invalid",
        only_known_nodes=True,
    )

    registry = SnapshotFacadeRegistry(snapshot_uri=str(output))
    projects = await registry.list_projects()

    assert project.canonical_name in [item.canonical_name for item in projects]
    loaded_version = await registry.get_version(project.canonical_name, version.version)
    assert loaded_version == version


async def test_snapshot_registry_reads_pkg_xz_uri(tmp_path: Path, monkeypatch):
    package_root = tmp_path / "snapshot_pkg"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")

    snapshot_path = package_root / "registry.sqlite.xz"
    project = _sample_project()
    version = _sample_version()
    write_facade_registry_snapshot(
        snapshot_path,
        projects=[project],
        versions_by_node_id={project.node_id: [version]},
        base_url="https://registry.example.invalid",
        only_known_nodes=False,
        compression="auto",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("snapshot_pkg", None)

    registry = SnapshotFacadeRegistry(
        snapshot_uri="pkg://snapshot_pkg/registry.sqlite.xz"
    )
    projects = await registry.list_projects()

    assert project.canonical_name in [item.canonical_name for item in projects]
    versions = await registry.list_versions(project.canonical_name)
    assert version in versions


async def test_snapshot_registry_normalizes_non_pep440_versions(tmp_path: Path):
    output = tmp_path / "registry.sqlite"
    project = _sample_project()
    valid = _sample_version()
    invalid = FacadeVersion(
        version="0.8.17-bugfix",
        download_url="https://example.invalid/node-invalid.zip",
        dependencies=(),
        deprecated=False,
    )
    write_facade_registry_snapshot(
        output,
        projects=[project],
        versions_by_node_id={project.node_id: [valid, invalid]},
        base_url="https://registry.example.invalid",
        only_known_nodes=True,
    )

    registry = SnapshotFacadeRegistry(snapshot_uri=str(output))
    versions = await registry.list_versions(project.canonical_name)

    assert [item.version for item in versions] == ["1.2.5", "0.8.17+bugfix"]


async def test_snapshot_registry_deduplicates_bundled_rewrite_version(tmp_path: Path):
    output = tmp_path / "registry.sqlite"
    project = FacadeProject(
        canonical_name="image-reward",
        display_name="image-reward",
        node_id="image-reward",
        repo_url="",
        repo_name="image-reward",
        description="Patched image-reward",
        aliases=("image-reward",),
        extra_requirements=(),
        skip_requirements=frozenset(),
        depends_on=(),
        latest_version="1.5",
    )
    version = FacadeVersion(
        version="1.5",
        download_url="https://files.pythonhosted.org/image_reward-1.5.whl",
        dependencies=(),
        deprecated=False,
    )
    write_facade_registry_snapshot(
        output,
        projects=[project],
        versions_by_node_id={project.node_id: [version]},
        base_url="https://registry.example.invalid",
        only_known_nodes=False,
    )

    registry = SnapshotFacadeRegistry(snapshot_uri=str(output))

    assert [item.version for item in await registry.list_versions(project)] == ["1.5"]


async def test_collect_versions_reuses_complete_unchanged_snapshot():
    project = _sample_project()
    version = _sample_version()

    class _NoFetchRegistry:
        async def list_versions(self, _project):
            raise AssertionError("unchanged project should reuse its previous versions")

    versions = await _collect_versions(
        _NoFetchRegistry(),  # type: ignore[arg-type]
        [project],
        previous={
            (
                project.node_id,
                "https://github.com/pythongosssss/comfyui-custom-scripts",
            ): (project.latest_version, [version])
        },
    )

    assert versions == {project.node_id: [version]}


async def test_collect_versions_refreshes_snapshot_missing_declared_latest():
    project = _sample_project()
    latest = _sample_version()
    stale = FacadeVersion(
        version="0.0.1",
        download_url="https://example.invalid/stale.zip",
        dependencies=(),
        deprecated=False,
    )

    class _FetchRegistry:
        calls = 0

        async def list_versions(self, _project):
            self.calls += 1
            return [latest]

    registry = _FetchRegistry()
    versions = await _collect_versions(
        registry,  # type: ignore[arg-type]
        [project],
        previous={
            (
                project.node_id,
                "https://github.com/pythongosssss/comfyui-custom-scripts",
            ): (project.latest_version, [stale])
        },
    )

    assert registry.calls == 1
    assert versions == {project.node_id: [latest]}


def test_snapshot_validation_rejects_a_missing_declared_latest():
    project = _sample_project()
    stale = FacadeVersion(
        version="1.2.4",
        download_url="https://example.invalid/stale.zip",
        dependencies=(),
        deprecated=False,
    )

    try:
        _validated_installable_projects([project], {project.node_id: [stale]})
    except RuntimeError as exc:
        assert "comfyui-custom-scripts=1.2.5" in str(exc)
    else:
        raise AssertionError(
            "missing declared latest version should reject the snapshot"
        )


async def test_snapshot_registry_hides_excluded_facade_projects(tmp_path: Path):
    output = tmp_path / "registry.sqlite"
    excluded_projects = [
        FacadeProject(
            canonical_name="comfyui-manager",
            display_name="ComfyUI-Manager",
            node_id="comfyui-manager",
            repo_url="https://github.com/Comfy-Org/ComfyUI-Manager",
            repo_name="ComfyUI-Manager",
            description="",
            aliases=("comfyui-manager", "comfyui-manager"),
            extra_requirements=(),
            skip_requirements=frozenset(),
            depends_on=(),
            latest_version="3.0.1",
        ),
        FacadeProject(
            canonical_name="gguf",
            display_name="gguf",
            node_id="gguf",
            repo_url="https://github.com/calcuis/gguf",
            repo_name="gguf",
            description="",
            aliases=("gguf",),
            extra_requirements=(),
            skip_requirements=frozenset(),
            depends_on=(),
            latest_version="0.0.1",
        ),
    ]
    normal_project = _sample_project()
    normal_version = _sample_version()
    write_facade_registry_snapshot(
        output,
        projects=[*excluded_projects, normal_project],
        versions_by_node_id={
            project.node_id: [
                FacadeVersion(
                    version=project.latest_version or "0.0.1",
                    download_url=f"https://example.invalid/{project.canonical_name}.zip",
                    dependencies=(),
                    deprecated=False,
                )
            ]
            for project in excluded_projects
        }
        | {
            normal_project.node_id: [normal_version],
        },
        base_url="https://registry.example.invalid",
        only_known_nodes=True,
    )

    registry = SnapshotFacadeRegistry(snapshot_uri=str(output))

    assert await registry.get_project("comfyui-manager") is None
    assert await registry.get_project("comfyui_manager") is None
    assert await registry.get_project("gguf") is None
    assert await registry.list_versions("comfyui-manager") == []
    assert await registry.list_versions("gguf") == []
    projects = await registry.list_projects()
    assert "comfyui-manager" not in [item.canonical_name for item in projects]
    assert "gguf" not in [item.canonical_name for item in projects]
    assert normal_project.canonical_name in [item.canonical_name for item in projects]


def test_snapshot_class_type_rows_exclude_facade_shadow_packages():
    projects = [_sample_project()]
    rows = _build_class_type_rows(
        projects,
        {
            "ManagerNode": "https://github.com/Comfy-Org/ComfyUI-Manager",
            "GGUFNode": "https://github.com/calcuis/gguf",
            "CustomScriptsNode": "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
        },
    )

    assert ("ManagerNode", "comfyui-manager") not in rows
    assert ("GGUFNode", "gguf") not in rows
    assert all(not is_excluded_facade_project(pkg) for _, pkg in rows)
    assert ("CustomScriptsNode", "comfyui-custom-scripts") in rows


def _write(output, *, projects, created_at=None, overwrite=True, compression="auto"):
    return write_facade_registry_snapshot(
        output,
        projects=projects,
        versions_by_node_id={
            project.node_id: [
                FacadeVersion(
                    version=project.latest_version or "1.0.0",
                    download_url=f"https://example.invalid/{project.node_id}.zip",
                    dependencies=(),
                    deprecated=False,
                )
            ]
            for project in projects
        },
        base_url="https://registry.example.invalid",
        only_known_nodes=True,
        compression=compression,
        overwrite=overwrite,
        created_at=created_at,
    )


def _other_project() -> FacadeProject:
    return FacadeProject(
        canonical_name="new-project",
        display_name="New Project",
        node_id="new-project",
        repo_url="https://github.com/example/new-project",
        repo_name="new-project",
        description="New snapshot generation",
        aliases=("new-project",),
        extra_requirements=(),
        skip_requirements=frozenset(),
        depends_on=(),
        latest_version="2.0.0",
    )


@pytest.mark.parametrize("name", ["registry.sqlite", "registry.sqlite.xz"])
async def test_snapshot_round_trips_through_any_fsspec_url(facade_storage_url, name):
    output = f"{facade_storage_url}/{name}"
    project = _sample_project()

    assert _write(output, projects=[project], overwrite=False) == output
    with pytest.raises(FileExistsError):
        _write(output, projects=[project], overwrite=False)

    registry = SnapshotFacadeRegistry(snapshot_uri=output)
    assert await registry.get_project(project.node_id) == project
    assert (await registry.get_version(project.node_id, "1.2.5")).download_url.endswith(
        f"{project.node_id}.zip"
    )
    fs, path = fsspec.core.url_to_fs(output)
    assert [item for item in fs.ls(fs._parent(path), detail=False) if item.endswith(".tmp")] == []


async def test_snapshot_registry_serves_its_last_generation_while_the_source_is_unavailable(
    facade_storage_url,
):
    output = f"{facade_storage_url}/registry.sqlite"
    project = _sample_project()
    _write(output, projects=[project])
    registry = SnapshotFacadeRegistry(snapshot_uri=output, max_age_seconds=2100)
    assert await registry.refresh()

    fs, path = fsspec.core.url_to_fs(output)
    fs.rm(path)
    assert not await registry.refresh()
    health = registry.health()
    assert health["snapshot_error"] is None
    assert health["snapshot_refresh_error"]
    assert await registry.get_project(project.node_id) == project

    _write(output, projects=[_other_project()])
    assert await registry.refresh()
    assert registry.health()["snapshot_refresh_error"] is None
    assert await registry.get_project("new-project") == _other_project()
    assert await registry.get_project(project.node_id) is None


async def test_snapshot_registry_loads_a_replaced_generation_and_skips_an_unchanged_one(
    facade_storage_url, monkeypatch
):
    output = f"{facade_storage_url}/registry.sqlite.xz"
    _write(output, projects=[_sample_project()])
    registry = SnapshotFacadeRegistry(snapshot_uri=output)
    assert await registry.refresh()

    downloads = 0
    original = registry._download  # pylint: disable=protected-access

    def counting_download(destination):
        nonlocal downloads
        downloads += 1
        return original(destination)

    monkeypatch.setattr(registry, "_download", counting_download)
    assert not await registry.refresh()
    assert downloads == 0

    _write(output, projects=[_other_project()])
    assert await registry.refresh()
    assert downloads == 1
    names = [item.canonical_name for item in await registry.list_projects()]
    assert "new-project" in names
    assert _sample_project().canonical_name not in names


async def test_snapshot_registry_retries_a_generation_replaced_during_download(
    facade_storage_url, monkeypatch
):
    output = f"{facade_storage_url}/registry.sqlite"
    _write(output, projects=[_sample_project()])
    registry = SnapshotFacadeRegistry(snapshot_uri=output)
    original = registry._download  # pylint: disable=protected-access
    downloads = 0

    def replace_after_first_download(destination):
        nonlocal downloads
        original(destination)
        downloads += 1
        if downloads == 1:
            _write(output, projects=[_other_project()])

    monkeypatch.setattr(registry, "_download", replace_after_first_download)
    projects = await registry.list_projects()

    assert downloads == 2
    names = [item.canonical_name for item in projects]
    assert "new-project" in names
    assert _sample_project().canonical_name not in names


async def test_snapshot_registry_requests_and_health_do_no_storage_io(
    facade_storage_url, monkeypatch
):
    output = f"{facade_storage_url}/registry.sqlite"
    project = _sample_project()
    _write(output, projects=[project])
    registry = SnapshotFacadeRegistry(snapshot_uri=output, max_age_seconds=2100)
    assert await registry.refresh()

    def no_io(*_args, **_kwargs):
        raise AssertionError("request and health paths must not touch snapshot storage")

    fs = registry._fs  # pylint: disable=protected-access
    with monkeypatch.context() as patch:
        for method in ("info", "open", "_open", "exists", "cat_file", "get_file"):
            patch.setattr(fs, method, no_io, raising=False)

        assert await registry.get_project(project.node_id) == project
        assert await registry.list_versions(project.node_id)
        assert registry.health()["snapshot_error"] is None


async def test_snapshot_registry_reports_unavailable_before_the_first_load_and_expired_generations(
    facade_storage_url,
):
    output = f"{facade_storage_url}/registry.sqlite"
    registry = SnapshotFacadeRegistry(snapshot_uri=output, max_age_seconds=2100)
    assert registry.health()["snapshot_error"]
    with pytest.raises(SnapshotUnavailableError):
        await registry.list_projects()

    _write(output, projects=[_sample_project()],
           created_at=datetime.now(timezone.utc) - timedelta(hours=1))
    with pytest.raises(SnapshotUnavailableError, match="expired"):
        await registry.list_projects()
    assert registry.health()["snapshot_age_seconds"] >= 3600
    assert "expired" in registry.health()["snapshot_error"]

    _write(output, projects=[_sample_project()])
    assert await registry.refresh()
    assert registry.health()["snapshot_error"] is None
    assert await registry.get_project(_sample_project().node_id) == _sample_project()

    fs, path = fsspec.core.url_to_fs(output)
    fs.pipe_file(path, b"invalid snapshot")
    assert not await registry.refresh()
    assert registry.health()["snapshot_refresh_error"]
    assert await registry.get_project(_sample_project().node_id) == _sample_project()


def test_snapshot_replacement_is_atomic_when_publishing_fails(facade_storage_url, monkeypatch):
    output = f"{facade_storage_url}/registry.sqlite"
    _write(output, projects=[_sample_project()])
    fs, path = fsspec.core.url_to_fs(output)
    previous = fs.cat_file(path)

    def interrupted_put(lpath, rpath, *args, **kwargs):
        del lpath, args, kwargs
        fs.pipe_file(rpath, b"partial")
        raise OSError("simulated interrupted upload")

    with monkeypatch.context() as patch:
        patch.setattr(fs, "put_file", interrupted_put)
        with pytest.raises(OSError, match="simulated interrupted"):
            _write(output, projects=[_other_project()])

    fs, path = fsspec.core.url_to_fs(output)
    assert fs.cat_file(path) == previous
    assert [item for item in fs.ls(fs._parent(path), detail=False) if item.endswith(".tmp")] == []


async def test_snapshot_command_reuses_the_previous_snapshot_at_any_fsspec_url(facade_storage_url):
    output = f"{facade_storage_url}/registry.sqlite.xz"
    assert await _load_previous_snapshot(output) == {}

    project = _sample_project()
    _write(output, projects=[project])
    previous = await _load_previous_snapshot(output)
    key = (project.node_id, "https://github.com/pythongosssss/comfyui-custom-scripts")
    assert previous[key][0] == "1.2.5"
    assert [item.version for item in previous[key][1]] == ["1.2.5"]

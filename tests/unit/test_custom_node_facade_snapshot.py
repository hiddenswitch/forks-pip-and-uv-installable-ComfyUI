from __future__ import annotations

import json
import lzma
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
    _validated_installable_projects,
    write_facade_registry_snapshot,
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


async def test_snapshot_unavailable_and_expired_generations_recover(tmp_path: Path):
    output = tmp_path / "registry.sqlite"
    project = _sample_project()

    def write():
        write_facade_registry_snapshot(
            output, projects=[project],
            versions_by_node_id={project.node_id: [_sample_version()]},
            base_url="https://registry.example.invalid", only_known_nodes=True,
            overwrite=True,
        )

    write()
    registry = SnapshotFacadeRegistry(snapshot_uri=str(output), max_age_seconds=2100)
    assert await registry.get_project(project.node_id) == project
    output.unlink()
    with pytest.raises(SnapshotUnavailableError):
        await registry.list_projects()
    assert not output.exists()
    assert (await registry.health())["snapshot_error"]

    output.write_bytes(b"invalid snapshot")
    with pytest.raises(SnapshotUnavailableError):
        await registry.list_projects()
    write()
    with sqlite3.connect(output) as connection:
        connection.execute("UPDATE metadata SET value = ? WHERE key = 'created_at'", (
            (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        ))
    with pytest.raises(SnapshotUnavailableError, match="expired"):
        await registry.list_projects()
    assert (await registry.health())["snapshot_age_seconds"] >= 3600
    write()
    assert await registry.get_project(project.node_id) == project
    assert (await registry.health())["snapshot_error"] is None


async def test_snapshot_age_limit_is_optional(tmp_path: Path):
    output = tmp_path / "registry.sqlite"
    project = _sample_project()
    write_facade_registry_snapshot(
        output, projects=[project], versions_by_node_id={project.node_id: [_sample_version()]},
        base_url="https://registry.example.invalid", only_known_nodes=True,
    )
    with sqlite3.connect(output) as connection:
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


async def test_snapshot_registry_retries_replacement_during_load(
    tmp_path: Path, monkeypatch
):
    output = tmp_path / "registry.sqlite"
    old_project = _sample_project()
    old_version = _sample_version()
    write_facade_registry_snapshot(
        output,
        projects=[old_project],
        versions_by_node_id={old_project.node_id: [old_version]},
        base_url="https://registry.example.invalid",
        only_known_nodes=True,
    )

    new_project = FacadeProject(
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
    new_version = FacadeVersion(
        version="2.0.0",
        download_url="https://example.invalid/new-project.zip",
        dependencies=(),
        deprecated=False,
    )
    replacement = tmp_path / "replacement.sqlite"
    write_facade_registry_snapshot(
        replacement,
        projects=[new_project],
        versions_by_node_id={new_project.node_id: [new_version]},
        base_url="https://registry.example.invalid",
        only_known_nodes=True,
    )

    registry = SnapshotFacadeRegistry(snapshot_uri=str(output))
    original_load = registry._load_all  # pylint: disable=protected-access
    load_count = 0

    def replace_after_first_load():
        nonlocal load_count
        loaded = original_load()
        load_count += 1
        if load_count == 1:
            os.replace(replacement, output)
        return loaded

    monkeypatch.setattr(registry, "_load_all", replace_after_first_load)

    projects = await registry.list_projects()

    assert load_count == 2
    project_names = {project.canonical_name for project in projects}
    assert "new-project" in project_names
    assert old_project.canonical_name not in project_names
    assert await registry.get_version("new-project", "2.0.0") == new_version


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


def test_snapshot_replacement_is_atomic_when_staging_fails(tmp_path: Path, monkeypatch):
    output = tmp_path / "registry.sqlite"
    output.write_bytes(b"previous-valid-snapshot")
    project = _sample_project()

    def fail_staging(_source, staged):
        Path(staged).write_bytes(b"partial")
        raise OSError("simulated interrupted network-filesystem write")

    monkeypatch.setattr(
        "comfy.custom_node_facade.snapshot.shutil.copyfile", fail_staging
    )

    try:
        write_facade_registry_snapshot(
            output,
            projects=[project],
            versions_by_node_id={project.node_id: [_sample_version()]},
            base_url="https://registry.example.invalid",
            only_known_nodes=False,
            overwrite=True,
        )
    except OSError as exc:
        assert "simulated interrupted" in str(exc)
    else:
        raise AssertionError("staging failure should propagate")

    assert output.read_bytes() == b"previous-valid-snapshot"
    assert list(tmp_path.glob(".registry.sqlite.*.tmp")) == []


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

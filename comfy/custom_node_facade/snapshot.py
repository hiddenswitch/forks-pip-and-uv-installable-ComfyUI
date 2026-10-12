from __future__ import annotations

from ..cmd.main_pre import tracer

import asyncio
import json
import logging
import lzma
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

from ..component_model.configuration import Configuration
from ..component_model.node_registry import CUSTOM_NODE_REGISTRY
from .registry import (
    FacadeProject,
    FacadeRegistry,
    FacadeVersion,
    SnapshotFacadeRegistry,
    canonicalize_project_name,
    is_excluded_facade_project,
    normalize_repo_url,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1"


def _compact_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=False)


def _resolve_compression(output_path: Path, compression: str) -> str:
    if compression == "auto":
        return "xz" if output_path.suffix == ".xz" else "none"
    return compression


_EXTENSION_NODE_MAP_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json"


def _parse_extension_node_map(data: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for repo_url, value in data.items():
        if isinstance(value, list) and value:
            for class_type in value[0]:
                mapping.setdefault(class_type, repo_url)
    return mapping


async def _load_class_type_to_repo(session: aiohttp.ClientSession) -> dict[str, str]:
    """Return *class_type -> repo_url* from the latest extension-node-map, with bundled fallback."""
    try:
        async with session.get(_EXTENSION_NODE_MAP_URL) as response:
            response.raise_for_status()
            data = await response.json(content_type=None)
            return _parse_extension_node_map(data)
    except Exception:
        logger.debug("Failed to fetch live extension-node-map, using bundled fallback")
        from importlib.resources import files as resource_files

        path = resource_files("comfyui_manager").joinpath("extension-node-map.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        return _parse_extension_node_map(data)


def _build_class_type_rows(
    projects: list[FacadeProject],
    class_type_to_repo: dict[str, str],
) -> list[tuple[str, str]]:
    """Build ``(class_type, canonical_name)`` rows by joining extension-node-map against projects."""
    from .registry import repo_basename

    repo_to_canonical: dict[str, str] = {}
    for project in projects:
        if project.repo_url:
            repo_to_canonical[normalize_repo_url(project.repo_url)] = (
                project.canonical_name
            )

    class_type_to_repo = dict(class_type_to_repo)
    for spec in CUSTOM_NODE_REGISTRY:
        if normalize_repo_url(spec.repo_url) not in repo_to_canonical:
            continue
        for class_type in spec.expected_node_types:
            class_type_to_repo[class_type] = spec.repo_url
    rows: list[tuple[str, str]] = []
    for class_type, repo_url in class_type_to_repo.items():
        norm = normalize_repo_url(repo_url)
        canonical = repo_to_canonical.get(norm)
        if canonical is None:
            canonical = canonicalize_project_name(repo_basename(repo_url))
        if is_excluded_facade_project(canonical):
            continue
        rows.append((class_type, canonical))
    return rows


def _write_snapshot_sqlite(
    sqlite_path: Path,
    *,
    projects: list[FacadeProject],
    versions_by_node_id: dict[str, list[FacadeVersion]],
    class_type_rows: list[tuple[str, str]],
    base_url: str,
    only_known_nodes: bool,
) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(sqlite_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA page_size=4096")
        connection.execute("""
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID
        """)
        connection.execute("""
            CREATE TABLE projects (
                canonical_name TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                node_id TEXT NOT NULL,
                repo_url TEXT NOT NULL,
                repo_name TEXT NOT NULL,
                description TEXT NOT NULL,
                aliases_json TEXT NOT NULL,
                extra_requirements_json TEXT NOT NULL,
                skip_requirements_json TEXT NOT NULL,
                depends_on_json TEXT NOT NULL,
                latest_version TEXT
            ) WITHOUT ROWID
        """)
        connection.execute("""
            CREATE TABLE versions (
                node_id TEXT NOT NULL,
                version TEXT NOT NULL,
                download_url TEXT NOT NULL,
                dependencies_json TEXT NOT NULL,
                deprecated INTEGER NOT NULL,
                PRIMARY KEY (node_id, version)
            ) WITHOUT ROWID
        """)
        connection.execute("CREATE INDEX versions_by_node_id ON versions (node_id)")
        connection.execute("""
            CREATE TABLE IF NOT EXISTS class_types (
                class_type TEXT PRIMARY KEY,
                canonical_name TEXT NOT NULL
            ) WITHOUT ROWID
        """)

        project_rows = [
            (
                project.canonical_name,
                project.display_name,
                project.node_id,
                project.repo_url,
                project.repo_name,
                project.description,
                _compact_json(list(project.aliases)),
                _compact_json(list(project.extra_requirements)),
                _compact_json(sorted(project.skip_requirements)),
                _compact_json(list(project.depends_on)),
                project.latest_version,
            )
            for project in projects
        ]
        version_rows = [
            (
                project.node_id,
                version.version,
                version.download_url,
                _compact_json(list(version.dependencies)),
                int(version.deprecated),
            )
            for project in projects
            for version in versions_by_node_id.get(project.node_id, [])
        ]

        connection.executemany(
            """
            INSERT INTO projects (
                canonical_name, display_name, node_id, repo_url, repo_name, description,
                aliases_json, extra_requirements_json, skip_requirements_json, depends_on_json,
                latest_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            project_rows,
        )
        connection.executemany(
            """
            INSERT INTO versions (
                node_id, version, download_url, dependencies_json, deprecated
            ) VALUES (?, ?, ?, ?, ?)
            """,
            version_rows,
        )

        connection.executemany(
            "INSERT OR IGNORE INTO class_types (class_type, canonical_name) VALUES (?, ?)",
            class_type_rows,
        )

        metadata_rows = [
            ("format", "appmana-comfyui-pip-facade-registry-snapshot"),
            ("schema_version", _SCHEMA_VERSION),
            ("created_at", datetime.now(timezone.utc).isoformat()),
            ("registry_base_url", base_url),
            ("only_known_nodes", "1" if only_known_nodes else "0"),
            ("project_count", str(len(projects))),
            ("version_count", str(len(version_rows))),
        ]
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)", metadata_rows
        )
        connection.commit()
        connection.execute("VACUUM")
    finally:
        connection.close()


def write_facade_registry_snapshot(
    output_path: str | Path,
    *,
    projects: list[FacadeProject],
    versions_by_node_id: dict[str, list[FacadeVersion]],
    class_type_rows: list[tuple[str, str]] | None = None,
    base_url: str,
    only_known_nodes: bool,
    compression: str = "auto",
    overwrite: bool = False,
) -> Path:
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Snapshot output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_compression = _resolve_compression(destination, compression)

    with tempfile.TemporaryDirectory(prefix="comfyui_facade_snapshot_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        sqlite_path = temp_dir / "registry.sqlite"
        _write_snapshot_sqlite(
            sqlite_path,
            projects=projects,
            versions_by_node_id=versions_by_node_id,
            class_type_rows=class_type_rows or [],
            base_url=base_url,
            only_known_nodes=only_known_nodes,
        )
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as staged_file:
            staged_path = Path(staged_file.name)
        try:
            if resolved_compression == "xz":
                with (
                    sqlite_path.open("rb") as source,
                    lzma.open(staged_path, "wb", preset=9) as compressed,
                ):
                    shutil.copyfileobj(source, compressed)
            else:
                shutil.copyfile(sqlite_path, staged_path)
            os.replace(staged_path, destination)
        finally:
            staged_path.unlink(missing_ok=True)
    return destination


async def snapshot_facade_registry(configuration: Configuration) -> Path:
    output_path = configuration.pip_facade_snapshot_output
    if not output_path:
        raise ValueError("pip_facade_snapshot_output must be set")

    timeout = aiohttp.ClientTimeout(total=10 * 60.0, connect=60.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        previous = await _load_previous_snapshot(Path(output_path))
        registry = FacadeRegistry(
            session,
            base_url=configuration.pip_facade_registry_base_url,
            only_known_nodes=configuration.pip_facade_only_known_nodes,
        )
        with tracer.start_as_current_span("Snapshot Facade Registry") as span:
            span.set_attribute(
                "facade.registry_base_url", configuration.pip_facade_registry_base_url
            )
            span.set_attribute(
                "facade.only_known_nodes", configuration.pip_facade_only_known_nodes
            )
            projects = await registry.list_projects()
            versions_by_node_id = await _collect_versions(
                registry, projects, previous=previous
            )
            projects = _validated_installable_projects(projects, versions_by_node_id)
            versions_by_node_id = {
                project.node_id: versions_by_node_id[project.node_id]
                for project in projects
            }
            class_type_to_repo = await _load_class_type_to_repo(session)
            class_type_rows = _build_class_type_rows(projects, class_type_to_repo)
            span.set_attribute("facade.class_type_count", len(class_type_rows))
            snapshot_path = await asyncio.to_thread(
                write_facade_registry_snapshot,
                output_path,
                projects=projects,
                versions_by_node_id=versions_by_node_id,
                class_type_rows=class_type_rows,
                base_url=configuration.pip_facade_registry_base_url,
                only_known_nodes=configuration.pip_facade_only_known_nodes,
                compression=configuration.pip_facade_snapshot_compression,
                overwrite=configuration.pip_facade_snapshot_overwrite,
            )
            span.set_attribute("facade.project_count", len(projects))
            span.set_attribute(
                "facade.version_count",
                sum(len(items) for items in versions_by_node_id.values()),
            )
            span.set_attribute("facade.snapshot_path", str(snapshot_path))
            return snapshot_path


def _validated_installable_projects(
    projects: list[FacadeProject],
    versions_by_node_id: dict[str, list[FacadeVersion]],
) -> list[FacadeProject]:
    missing_latest = [
        project
        for project in projects
        if project.latest_version is not None
        and not any(
            version.version == project.latest_version
            for version in versions_by_node_id.get(project.node_id, ())
        )
    ]
    if missing_latest:
        details = ", ".join(
            f"{project.canonical_name}={project.latest_version}"
            for project in missing_latest[:20]
        )
        raise RuntimeError(f"Snapshot omitted declared latest version(s): {details}")

    unavailable = [
        project.canonical_name
        for project in projects
        if not versions_by_node_id.get(project.node_id)
    ]
    if unavailable:
        logger.warning(
            "Omitting %d facade projects without installable versions: %s",
            len(unavailable),
            ", ".join(unavailable[:20]),
        )
    return [project for project in projects if versions_by_node_id.get(project.node_id)]


async def _collect_versions(
    registry: FacadeRegistry,
    projects: list[FacadeProject],
    *,
    previous: dict[tuple[str, str], tuple[str | None, list[FacadeVersion]]]
    | None = None,
) -> dict[str, list[FacadeVersion]]:
    semaphore = asyncio.Semaphore(12)
    previous = previous or {}

    async def load(project: FacadeProject) -> tuple[str, list[FacadeVersion]]:
        previous_project = previous.get(
            (project.node_id, normalize_repo_url(project.repo_url))
        )
        if previous_project is not None:
            previous_latest, previous_versions = previous_project
            has_latest = project.latest_version is None or any(
                version.version == project.latest_version
                for version in previous_versions
            )
            if (
                previous_latest == project.latest_version
                and previous_versions
                and has_latest
            ):
                return project.node_id, previous_versions
        async with semaphore:
            try:
                return project.node_id, await registry.list_versions(project)
            except Exception:
                logger.warning(
                    "Failed to list versions for %s, omitting from snapshot",
                    project.node_id,
                    exc_info=True,
                )
                return project.node_id, []

    results = await asyncio.gather(*(load(project) for project in projects))
    return dict(results)


async def _load_previous_snapshot(
    output_path: Path,
) -> dict[tuple[str, str], tuple[str | None, list[FacadeVersion]]]:
    if not output_path.exists():
        return {}
    try:
        registry = SnapshotFacadeRegistry(snapshot_uri=str(output_path))
        projects = await registry.list_projects()
        return {
            (project.node_id, normalize_repo_url(project.repo_url)): (
                project.latest_version,
                await registry.list_versions(project),
            )
            for project in projects
        }
    except Exception:
        logger.warning(
            "Could not reuse previous facade snapshot; performing a full refresh",
            exc_info=True,
        )
        return {}

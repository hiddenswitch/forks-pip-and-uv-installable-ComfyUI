from __future__ import annotations

from ..cmd.main_pre import tracer

import asyncio
import json
import logging
import os
import platform
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import aiohttp
import fsspec

from ..app.custom_node_manager import CustomNodeManager
from ..component_model.node_registry import CUSTOM_NODE_REGISTRY, CustomNodeSpec
from ..nodes.custom_node_dependencies import CUSTOM_NODE_RUNTIME_DEPS

logger = logging.getLogger(__name__)

_SIMPLE_NAME_RE = re.compile(r"[-_.]+")
EXCLUDED_FACADE_PROJECT_NAMES: frozenset[str] = frozenset(
    {
        # ComfyUI-Manager is a first-class dependency of this fork, not a facade
        # package. Publishing a facade with the same normalized name shadows the
        # real comfyui_manager distribution when uv resolves extra indexes.
        "comfyui-manager",
        # gguf is a PyPI runtime dependency imported by core ComfyUI code. The
        # custom-node facade entry with this name vendors an unrelated node repo
        # and breaks `import gguf` when it wins dependency resolution.
        "gguf",
    }
)
EXCLUDED_FACADE_REPOSITORIES: frozenset[str] = frozenset(
    {
        # The Manager registry still advertises this repository, but GitHub has
        # removed it.  Keeping it in the facade creates a project page whose only
        # wheel fails to build with a 404.
        "https://github.com/kayarte/geonodes",
    }
)


def canonicalize_project_name(name: str) -> str:
    return _SIMPLE_NAME_RE.sub("-", name).strip("-").lower()


def is_excluded_facade_project(name: str) -> bool:
    return canonicalize_project_name(name) in EXCLUDED_FACADE_PROJECT_NAMES


def normalize_repo_url(url: str) -> str:
    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    netloc = parsed.netloc.lower()
    return f"{scheme}://{netloc}{path.lower()}"


def is_excluded_facade_repository(url: str) -> bool:
    return normalize_repo_url(url) in EXCLUDED_FACADE_REPOSITORIES


def repo_basename(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    if not path:
        return "custom-node"
    name = path.rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name or "custom-node"


def _sort_versions(versions: list["FacadeVersion"]) -> list["FacadeVersion"]:
    try:
        from packaging.version import Version
    except Exception:
        return sorted(versions, key=lambda item: item.version, reverse=True)
    valid: list[tuple[object, FacadeVersion]] = []
    invalid: list[FacadeVersion] = []
    for item in versions:
        try:
            valid.append((Version(item.version), item))
        except Exception:
            invalid.append(item)
    valid_sorted = [
        item for _, item in sorted(valid, key=lambda pair: pair[0], reverse=True)
    ]
    invalid_sorted = sorted(invalid, key=lambda item: item.version, reverse=True)
    return valid_sorted + invalid_sorted


def _is_pep440_version(version: str) -> bool:
    try:
        from packaging.version import Version

        Version(version)
        return True
    except Exception:
        return False


def _normalize_pep440_version(version: str) -> str | None:
    if _is_pep440_version(version):
        return version

    # Comfy Registry contains a small number of otherwise conventional
    # versions with a free-form suffix separated by a hyphen, for example
    # ``2.0.0-inprogress``.  PEP 440 represents that information as a local
    # version, so expose ``2.0.0+inprogress`` instead of silently dropping the
    # newest release from the package index.
    base, separator, suffix = version.partition("-")
    if not separator or not _is_pep440_version(base):
        return None
    local = re.sub(r"[^A-Za-z0-9]+", ".", suffix).strip(".").lower()
    if not local:
        return None
    normalized = f"{base}+{local}"
    return normalized if _is_pep440_version(normalized) else None


def _normalize_pep440_versions(
    versions: list["FacadeVersion"],
) -> list["FacadeVersion"]:
    normalized: list[FacadeVersion] = []
    seen: set[str] = set()
    for item in versions:
        version = _normalize_pep440_version(item.version)
        if version is None or version in seen:
            continue
        seen.add(version)
        normalized.append(
            item if version == item.version else replace(item, version=version)
        )
    return normalized


_MANAGER_REGISTRY_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json"


def _manager_registry_path() -> Path:
    return Path(files("comfyui_manager").joinpath("custom-node-list.json"))


def _load_bundled_manager_registry() -> list[dict[str, Any]]:
    path = _manager_registry_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("custom_nodes", ()))


async def _load_manager_registry(
    session: aiohttp.ClientSession,
) -> list[dict[str, Any]]:
    try:
        async with session.get(_MANAGER_REGISTRY_URL) as response:
            response.raise_for_status()
            data = await response.json(content_type=None)
            return list(data.get("custom_nodes", ()))
    except Exception:
        logger.debug(
            "Failed to fetch live custom-node-list.json, using bundled fallback"
        )
        return _load_bundled_manager_registry()


def _build_alias_cache(projects: list["FacadeProject"]) -> dict[str, str]:
    """Map alias -> canonical name, never letting an alias shadow another
    project's canonical name; ambiguous aliases go to the first project in
    canonical-name order."""
    canonical_names = {project.canonical_name for project in projects}
    aliases: dict[str, str] = {}
    for project in sorted(projects, key=lambda item: item.canonical_name):
        for alias in project.aliases:
            if alias != project.canonical_name and alias in canonical_names:
                continue
            aliases.setdefault(alias, project.canonical_name)
    return aliases


@dataclass(frozen=True)
class FacadeVersion:
    version: str
    download_url: str
    dependencies: tuple[str, ...]
    deprecated: bool


@dataclass(frozen=True)
class FacadeProject:
    canonical_name: str
    display_name: str
    node_id: str
    repo_url: str
    repo_name: str
    description: str
    aliases: tuple[str, ...]
    extra_requirements: tuple[str, ...]
    skip_requirements: frozenset[str]
    depends_on: tuple[str, ...]
    latest_version: str | None = None


class FacadeRegistryProtocol(Protocol):
    async def list_projects(self) -> list["FacadeProject"]: ...

    async def get_project(self, name: str) -> "FacadeProject | None": ...

    async def list_versions(
        self, project: "FacadeProject | str"
    ) -> list["FacadeVersion"]: ...

    async def get_version(
        self, project: "FacadeProject | str", version: str
    ) -> "FacadeVersion | None": ...

    async def dependency_project_name(self, dependency_id: str) -> str: ...


class SnapshotUnavailableError(RuntimeError):
    """The configured registry snapshot cannot safely serve metadata."""


class SnapshotFacadeRegistry:
    def __init__(self, *, snapshot_uri: str, max_age_seconds: int = 0) -> None:
        if max_age_seconds < 0:
            raise ValueError("Snapshot maximum age must be nonnegative")
        self._snapshot_uri = snapshot_uri
        self._max_age_seconds = max_age_seconds
        self._created_at: datetime | None = None
        self._db_path: str | None = None
        self._lock = asyncio.Lock()
        self._projects: list[FacadeProject] = []
        self._alias_cache: dict[str, str] = {}
        self._versions_cache: dict[str, list[FacadeVersion]] = {}
        self._loaded_signature: tuple[int, int, int, int, int] | None = None

    def _file_path(self) -> str | None:
        if self._snapshot_uri.startswith("file://"):
            return self._snapshot_uri.removeprefix("file://")
        if not self._snapshot_uri.startswith(
            ("pkg://", "s3://", "http://", "https://")
        ):
            return self._snapshot_uri
        return None

    def _needs_reload(self) -> bool:
        path = self._file_path()
        if path is None:
            return not self._projects or bool(self._max_age_seconds and self.snapshot_age_seconds > self._max_age_seconds)
        signature = self._file_signature()
        return signature != self._loaded_signature

    def _file_signature(self) -> tuple[int, int, int, int, int] | None:
        path = self._file_path()
        if path is None:
            return None
        # A cached generation must not hide a disconnected shared-volume mount.
        with open(path, "rb") as source:
            source.read(1)
            stat = os.fstat(source.fileno())
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    async def _ensure_loaded(self) -> None:
        try:
            await self._reload_if_needed()
            if self._max_age_seconds and self.snapshot_age_seconds > self._max_age_seconds:
                raise SnapshotUnavailableError("Registry snapshot has expired")
        except (OSError, sqlite3.Error, ValueError, KeyError) as exc:
            raise SnapshotUnavailableError(f"Registry snapshot is unavailable: {exc}") from exc

    @property
    def snapshot_age_seconds(self) -> float:
        if self._created_at is None:
            return 0.0
        return max(0.0, (datetime.now(timezone.utc) - self._created_at).total_seconds())

    async def health(self) -> dict[str, Any]:
        error = None
        try:
            await self._ensure_loaded()
        except SnapshotUnavailableError as exc:
            error = str(exc)
        return {
            "snapshot_created_at": self._created_at.isoformat() if self._created_at else None,
            "snapshot_age_seconds": self.snapshot_age_seconds if self._created_at else None,
            "snapshot_error": error,
        }

    async def _reload_if_needed(self) -> None:
        if not self._needs_reload():
            return
        async with self._lock:
            if not self._needs_reload():
                return
            while True:
                signature_before = self._file_signature()
                projects, aliases, versions, created_at = await asyncio.to_thread(self._load_all)
                signature_after = self._file_signature()
                if (
                    self._file_path() is not None
                    and signature_before != signature_after
                ):
                    logger.debug(
                        "Facade snapshot changed while loading; retrying the new generation"
                    )
                    continue
                self._projects = projects
                self._alias_cache = aliases
                self._versions_cache = versions
                self._loaded_signature = signature_after
                self._created_at = created_at
                return

    async def list_projects(self) -> list[FacadeProject]:
        await self._ensure_loaded()
        return self._projects

    async def get_project(self, name: str) -> FacadeProject | None:
        await self._ensure_loaded()
        canonical = canonicalize_project_name(name)
        if canonical in EXCLUDED_FACADE_PROJECT_NAMES:
            return None
        target = self._alias_cache.get(canonical, canonical)
        for project in self._projects:
            if project.canonical_name == target:
                return project
        return None

    async def list_versions(self, project: FacadeProject | str) -> list[FacadeVersion]:
        resolved = (
            await self.get_project(project) if isinstance(project, str) else project
        )
        if resolved is None:
            return []
        await self._ensure_loaded()
        return self._versions_cache.get(resolved.node_id, [])

    async def get_version(
        self, project: FacadeProject | str, version: str
    ) -> FacadeVersion | None:
        for item in await self.list_versions(project):
            if item.version == version:
                return item
        return None

    async def dependency_project_name(self, dependency_id: str) -> str:
        project = await self.get_project(dependency_id)
        if project is not None:
            return project.canonical_name
        return canonicalize_project_name(dependency_id)

    @staticmethod
    def _needs_decompression(path: str) -> bool:
        return path.endswith((".xz", ".gz", ".bz2", ".zst"))

    def _load_all(
        self,
    ) -> tuple[list[FacadeProject], dict[str, str], dict[str, list[FacadeVersion]], datetime]:
        path = self._file_path()
        if path is not None and not self._needs_decompression(path):
            db_path = path
            cleanup = False
        else:
            db_path = self._materialize_snapshot()
            cleanup = True
        try:
            conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                metadata = dict(conn.execute("SELECT key, value FROM metadata"))
                if metadata.get("format") != "appmana-comfyui-pip-facade-registry-snapshot" or metadata.get("schema_version") != "1":
                    raise ValueError("Unsupported registry snapshot format or schema")
                created_at = datetime.fromisoformat(metadata["created_at"])
                if created_at.tzinfo is None:
                    raise ValueError("Registry snapshot created_at must include a timezone")
                projects = [
                    FacadeProject(
                        canonical_name=row["canonical_name"],
                        display_name=row["display_name"],
                        node_id=row["node_id"],
                        repo_url=row["repo_url"],
                        repo_name=row["repo_name"],
                        description=row["description"],
                        aliases=tuple(json.loads(row["aliases_json"])),
                        extra_requirements=tuple(
                            json.loads(row["extra_requirements_json"])
                        ),
                        skip_requirements=frozenset(
                            json.loads(row["skip_requirements_json"])
                        ),
                        depends_on=tuple(json.loads(row["depends_on_json"])),
                        latest_version=row["latest_version"],
                    )
                    for row in conn.execute(
                        "SELECT * FROM projects ORDER BY canonical_name"
                    )
                    if row["canonical_name"] not in EXCLUDED_FACADE_PROJECT_NAMES
                ]
                raw_versions: dict[str, list[FacadeVersion]] = {}
                for row in conn.execute(
                    "SELECT * FROM versions ORDER BY node_id, version"
                ):
                    if row["node_id"] in EXCLUDED_FACADE_PROJECT_NAMES:
                        continue
                    raw_versions.setdefault(row["node_id"], []).append(
                        FacadeVersion(
                            version=row["version"],
                            download_url=row["download_url"],
                            dependencies=tuple(json.loads(row["dependencies_json"])),
                            deprecated=bool(row["deprecated"]),
                        )
                    )
            finally:
                conn.close()
        finally:
            if cleanup:
                Path(db_path).unlink(missing_ok=True)

        versions_cache: dict[str, list[FacadeVersion]] = {}
        for node_id, versions in raw_versions.items():
            filtered = _normalize_pep440_versions(versions)
            if filtered:
                versions_cache[node_id] = _sort_versions(filtered)

        rewrite_projects, rewrite_versions = _rewrite_projects_and_versions()
        for rp in rewrite_projects:
            if not any(p.canonical_name == rp.canonical_name for p in projects):
                projects.append(rp)
        for node_id, rv in rewrite_versions.items():
            versions_cache[node_id] = _sort_versions(
                _normalize_pep440_versions([*versions_cache.get(node_id, ()), *rv])
            )

        aliases = _build_alias_cache(projects)
        return projects, aliases, versions_cache, created_at

    def _materialize_snapshot(self) -> str:
        with tempfile.NamedTemporaryFile(
            prefix="comfyui_facade_snapshot_",
            suffix=".sqlite",
            delete=False,
        ) as temp_file:
            temp_path = temp_file.name

        try:
            with fsspec.open(
                self._snapshot_uri, mode="rb", compression="infer"
            ) as source:
                with open(temp_path, "wb") as destination:
                    shutil.copyfileobj(source, destination)
        except Exception:
            Path(temp_path).unlink(missing_ok=True)
            raise
        return temp_path


def _rewrite_projects_and_versions() -> tuple[
    list[FacadeProject], dict[str, list[FacadeVersion]]
]:
    """Build synthetic FacadeProject/FacadeVersion entries for PyPI rewrite packages."""
    from .builder import PYPI_REWRITE_PACKAGES

    projects: list[FacadeProject] = []
    versions: dict[str, list[FacadeVersion]] = {}
    for spec in PYPI_REWRITE_PACKAGES:
        canonical = canonicalize_project_name(spec.name)
        project = FacadeProject(
            canonical_name=canonical,
            display_name=spec.name,
            node_id=canonical,
            repo_url="",
            repo_name=canonical,
            description=f"Patched {spec.name} with relaxed dependency pins",
            aliases=(canonical,),
            extra_requirements=(),
            skip_requirements=frozenset(),
            depends_on=(),
            latest_version=spec.version,
        )
        projects.append(project)
        versions[canonical] = [
            FacadeVersion(
                version=spec.version,
                download_url=spec.wheel_url,
                dependencies=spec.dependencies,
                deprecated=False,
            )
        ]
    return projects, versions


class _OverlayIndex:
    def __init__(self) -> None:
        self._by_node_id: dict[str, CustomNodeSpec] = {}
        self._by_repo: dict[str, CustomNodeSpec] = {}
        for spec in CUSTOM_NODE_REGISTRY:
            self._by_node_id[canonicalize_project_name(spec.node_id)] = spec
            self._by_repo[normalize_repo_url(spec.repo_url)] = spec

    def match(
        self, node_id: str | None, repo_url: str, repo_name: str, title: str
    ) -> CustomNodeSpec | None:
        keys = []
        if node_id:
            keys.append(canonicalize_project_name(node_id))
        keys.extend(
            canonicalize_project_name(value) for value in (repo_name, title) if value
        )
        for key in keys:
            spec = self._by_node_id.get(key)
            if spec is not None:
                return spec
        return self._by_repo.get(normalize_repo_url(repo_url))


class FacadeRegistry:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        base_url: str = "https://api.comfy.org",
        only_known_nodes: bool = False,
    ) -> None:
        self._session = session
        self._base_url = base_url.rstrip("/")
        self._only_known_nodes = only_known_nodes
        self._overlay_index = _OverlayIndex()
        self._project_cache: list[FacadeProject] | None = None
        self._alias_cache: dict[str, str] = {}
        self._versions_cache: dict[str, list[FacadeVersion]] = {}
        self._manager_versions_by_repo: dict[str, list[FacadeVersion]] = {}
        self._projects_lock = asyncio.Lock()
        self._versions_lock = asyncio.Lock()

    async def list_projects(self) -> list[FacadeProject]:
        if self._project_cache is not None:
            return self._project_cache
        async with self._projects_lock:
            if self._project_cache is not None:
                return self._project_cache
            with tracer.start_as_current_span("List Facade Projects") as span:
                span.set_attribute("facade.only_known_nodes", self._only_known_nodes)
                span.set_attribute("facade.registry_base_url", self._base_url)
                projects = await self._build_projects()
                span.set_attribute("facade.project_count", len(projects))
            self._project_cache = projects
            return projects

    async def get_project(self, name: str) -> FacadeProject | None:
        canonical = canonicalize_project_name(name)
        if canonical in EXCLUDED_FACADE_PROJECT_NAMES:
            return None
        projects = await self.list_projects()
        target = self._alias_cache.get(canonical, canonical)
        for project in projects:
            if project.canonical_name == target:
                return project
        return None

    async def list_versions(self, project: FacadeProject | str) -> list[FacadeVersion]:
        resolved = (
            await self.get_project(project) if isinstance(project, str) else project
        )
        if resolved is None:
            return []
        if resolved.node_id in self._versions_cache:
            return self._versions_cache[resolved.node_id]
        manager_versions = self._manager_versions_by_repo.get(
            normalize_repo_url(resolved.repo_url)
        )
        if manager_versions is not None:
            return manager_versions
        async with self._versions_lock:
            if resolved.node_id in self._versions_cache:
                return self._versions_cache[resolved.node_id]
            params = [
                ("statuses", "NodeVersionStatusActive"),
                ("statuses", "NodeVersionStatusPending"),
            ]
            with tracer.start_as_current_span("List Facade Project Versions") as span:
                span.set_attribute("facade.project_name", resolved.canonical_name)
                span.set_attribute("facade.node_id", resolved.node_id)
                async with self._session.get(
                    f"{self._base_url}/nodes/{resolved.node_id}/versions",
                    params=params,
                ) as response:
                    response.raise_for_status()
                    payload = await response.json()
                versions = [
                    FacadeVersion(
                        version=item["version"],
                        download_url=item["downloadUrl"],
                        dependencies=tuple(item.get("dependencies") or ()),
                        deprecated=bool(item.get("deprecated", False)),
                    )
                    for item in payload
                    if item.get("version") and item.get("downloadUrl")
                ]
                dropped_versions = len(versions)
                versions = _normalize_pep440_versions(versions)
                dropped_versions -= len(versions)
                if dropped_versions:
                    span.set_attribute(
                        "facade.dropped_non_pep440_versions", dropped_versions
                    )
            if not versions and resolved.repo_url:
                fallback = self._fallback_version_from_repo(resolved)
                if fallback is not None:
                    versions = [fallback]
            self._versions_cache[resolved.node_id] = _sort_versions(versions)
            return self._versions_cache[resolved.node_id]

    async def get_version(
        self, project: FacadeProject | str, version: str
    ) -> FacadeVersion | None:
        for item in await self.list_versions(project):
            if item.version == version:
                return item
        return None

    async def dependency_project_name(self, dependency_id: str) -> str:
        project = await self.get_project(dependency_id)
        if project is not None:
            return project.canonical_name
        return canonicalize_project_name(dependency_id)

    async def _build_projects(self) -> list[FacadeProject]:
        with tracer.start_as_current_span("Build Facade Project Registry") as span:
            manager_nodes = await _load_manager_registry(self._session)
            span.set_attribute("facade.manager_registry_items", len(manager_nodes))
            cnr_nodes = await self._fetch_cnr_nodes()
            span.set_attribute("facade.cnr_nodes", len(cnr_nodes))
        repo_to_cnr = {
            normalize_repo_url(node["repository"]): node
            for node in cnr_nodes
            if node.get("repository")
        }

        projects_by_name: dict[str, FacadeProject] = {}

        for item in manager_nodes:
            repo_url = self._extract_repo_url(item)
            if repo_url is None:
                continue
            if is_excluded_facade_repository(repo_url):
                continue
            if is_excluded_facade_project(repo_basename(repo_url)):
                continue
            cnr = repo_to_cnr.get(normalize_repo_url(repo_url))
            if cnr is not None:
                project = self._build_project(item, cnr, repo_url)
                if project is not None:
                    projects_by_name[project.canonical_name] = project
            elif self._is_supported_archive_host(repo_url):
                project = self._build_manager_only_project(item, repo_url)
                if project is not None:
                    projects_by_name.setdefault(project.canonical_name, project)

        for spec in CUSTOM_NODE_REGISTRY:
            cnr = repo_to_cnr.get(normalize_repo_url(spec.repo_url))
            if cnr is not None:
                item = {
                    "title": spec.display_name,
                    "reference": spec.repo_url,
                    "description": "",
                }
                project = self._build_project(item, cnr, spec.repo_url)
                if project is not None:
                    projects_by_name.setdefault(project.canonical_name, project)
            elif spec.inject_version is not None:
                project = self._build_injected_project(spec)
                projects_by_name.setdefault(project.canonical_name, project)

        self._ingest_cnr_only_nodes(projects_by_name, cnr_nodes)

        rewrite_projects, rewrite_versions = _rewrite_projects_and_versions()
        for rp in rewrite_projects:
            projects_by_name.setdefault(rp.canonical_name, rp)
        for node_id, rv in rewrite_versions.items():
            self._versions_cache.setdefault(node_id, []).extend(rv)

        projects = sorted(
            projects_by_name.values(), key=lambda project: project.canonical_name
        )
        self._alias_cache = _build_alias_cache(projects)
        return projects

    def _ingest_cnr_only_nodes(
        self,
        projects_by_name: dict[str, FacadeProject],
        cnr_nodes: list[dict[str, Any]],
    ) -> None:
        """Create projects for registry nodes not reachable through the manager list.

        The registry is the authoritative namespace for its own node ids: when a
        node's id collides with a name another repo claimed via its repo
        basename, the registry node takes the name and the other project is
        re-keyed to its own registry id.
        """
        if self._only_known_nodes:
            return
        represented_node_ids = {
            canonicalize_project_name(project.node_id)
            for project in projects_by_name.values()
        }
        for cnr in cnr_nodes:
            node_id = cnr.get("id")
            if not node_id:
                continue
            canonical_id = canonicalize_project_name(node_id)
            if canonical_id in represented_node_ids:
                continue
            if is_excluded_facade_project(node_id):
                continue
            if cnr.get("repository") and is_excluded_facade_repository(
                cnr["repository"]
            ):
                continue
            if cnr.get("status") in ("NodeStatusBanned", "NodeStatusDeleted"):
                continue
            project = self._build_cnr_only_project(cnr)
            existing = projects_by_name.get(project.canonical_name)
            projects_by_name[project.canonical_name] = project
            represented_node_ids.add(canonical_id)
            if existing is not None:
                self._rekey_displaced_project(
                    projects_by_name, existing, project.canonical_name
                )

    @staticmethod
    def _rekey_displaced_project(
        projects_by_name: dict[str, FacadeProject],
        displaced: FacadeProject,
        lost_name: str,
    ) -> None:
        fallback = canonicalize_project_name(displaced.node_id)
        if fallback == lost_name or fallback in projects_by_name:
            logger.warning(
                "Dropping facade project for %s: registry node owns the name %s and no free fallback name exists",
                displaced.repo_url or displaced.node_id,
                lost_name,
            )
            return
        aliases = {alias for alias in displaced.aliases if alias != lost_name}
        aliases.add(fallback)
        projects_by_name[fallback] = replace(
            displaced,
            canonical_name=fallback,
            aliases=tuple(sorted(aliases)),
        )
        logger.info(
            "Re-keyed facade project %s -> %s: registry node owns the name",
            lost_name,
            fallback,
        )

    def _build_cnr_only_project(self, cnr: dict[str, Any]) -> FacadeProject:
        node_id = cnr["id"]
        canonical_name = canonicalize_project_name(node_id)
        repo_url = cnr.get("repository") or ""
        repo_name = repo_basename(repo_url) if repo_url else canonical_name
        display_name = cnr.get("name") or node_id
        aliases = {
            canonical_name,
            canonicalize_project_name(repo_name),
            canonicalize_project_name(display_name),
        }
        latest = cnr.get("latest_version")
        latest_version = (
            _normalize_pep440_version(latest.get("version", ""))
            if isinstance(latest, dict)
            else None
        )
        return FacadeProject(
            canonical_name=canonical_name,
            display_name=display_name,
            node_id=node_id,
            repo_url=repo_url,
            repo_name=repo_name,
            description=cnr.get("description") or "",
            aliases=tuple(sorted(aliases)),
            extra_requirements=tuple(
                self._runtime_dependencies(None, node_id, repo_name)
            ),
            skip_requirements=frozenset(CustomNodeManager.DEFAULT_SKIP),
            depends_on=(),
            latest_version=latest_version,
        )

    async def _fetch_cnr_nodes(self) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        page = 1
        total_pages = 1
        with tracer.start_as_current_span("Fetch CNR Nodes") as span:
            span.set_attribute("facade.registry_base_url", self._base_url)
            span.set_attribute("facade.form_factor", self._form_factor())
            while page <= total_pages:
                params = {
                    "page": page,
                    "limit": 100,
                    "comfyui_version": "unknown",
                    "form_factor": self._form_factor(),
                }
                async with self._session.get(
                    f"{self._base_url}/nodes", params=params
                ) as response:
                    response.raise_for_status()
                    payload = await response.json()
                total_pages = int(payload.get("totalPages", 1))
                nodes.extend(payload.get("nodes") or ())
                page += 1
            span.set_attribute("facade.cnr_node_count", len(nodes))
        return nodes

    def _build_project(
        self, item: dict[str, Any], cnr: dict[str, Any], repo_url: str
    ) -> FacadeProject | None:
        repo_name = repo_basename(repo_url)
        overlay = self._overlay_index.match(
            cnr.get("id"),
            repo_url,
            repo_name,
            item.get("title") or cnr.get("name") or repo_name,
        )
        if self._only_known_nodes and overlay is None:
            return None

        canonical_name = canonicalize_project_name(
            overlay.node_id if overlay is not None else repo_name
        )
        aliases = {
            canonical_name,
            canonicalize_project_name(cnr["id"]),
            canonicalize_project_name(repo_name),
        }
        if title := item.get("title"):
            aliases.add(canonicalize_project_name(title))

        extra_requirements = list(
            self._runtime_dependencies(overlay, cnr["id"], repo_name)
        )
        skip_requirements = set(CustomNodeManager.DEFAULT_SKIP)
        depends_on: tuple[str, ...] = ()
        display_name = item.get("title") or cnr.get("name") or repo_name
        description = cnr.get("description") or item.get("description") or ""
        if overlay is not None:
            display_name = overlay.display_name
            extra_requirements.extend(overlay.extra_requirements)
            skip_requirements.update(overlay.skip_requirements)
            depends_on = tuple(overlay.depends_on)

        deduped_requirements: list[str] = []
        seen_requirements: set[str] = set()
        for requirement in extra_requirements:
            normalized = requirement.strip()
            if not normalized or normalized in seen_requirements:
                continue
            seen_requirements.add(normalized)
            deduped_requirements.append(normalized)

        latest_version = None
        latest = cnr.get("latest_version")
        if isinstance(latest, dict):
            latest_version = _normalize_pep440_version(latest.get("version", ""))

        return FacadeProject(
            canonical_name=canonical_name,
            display_name=display_name,
            node_id=cnr["id"],
            repo_url=repo_url,
            repo_name=repo_name,
            description=description,
            aliases=tuple(sorted(aliases)),
            extra_requirements=tuple(deduped_requirements),
            skip_requirements=frozenset(skip_requirements),
            depends_on=depends_on,
            latest_version=latest_version,
        )

    def _seed_injected_version(self, spec: CustomNodeSpec, node_id: str) -> None:
        download_url = self._github_archive_url(spec.repo_url, spec.git_ref)
        version = FacadeVersion(
            version=spec.inject_version,
            download_url=download_url,
            dependencies=tuple(spec.extra_requirements),
            deprecated=False,
        )
        self._versions_cache.setdefault(node_id, [version])

    def _build_manager_only_project(
        self, item: dict[str, Any], repo_url: str
    ) -> FacadeProject | None:
        if self._only_known_nodes:
            return None
        repo_name = repo_basename(repo_url)
        canonical_name = canonicalize_project_name(repo_name)
        display_name = item.get("title") or repo_name
        description = item.get("description") or ""
        aliases = {canonical_name, canonicalize_project_name(display_name)}
        download_url = self._github_archive_url(repo_url)
        version = FacadeVersion(
            version="0.0.1",
            download_url=download_url,
            dependencies=(),
            deprecated=False,
        )
        self._manager_versions_by_repo[normalize_repo_url(repo_url)] = [version]
        return FacadeProject(
            canonical_name=canonical_name,
            display_name=display_name,
            node_id=canonical_name,
            repo_url=repo_url,
            repo_name=repo_name,
            description=description,
            aliases=tuple(sorted(aliases)),
            extra_requirements=(),
            skip_requirements=frozenset(CustomNodeManager.DEFAULT_SKIP),
            depends_on=(),
            latest_version="0.0.1",
        )

    @staticmethod
    def _is_supported_archive_host(repo_url: str) -> bool:
        host = (urlparse(repo_url).hostname or "").lower()
        return host in ("github.com", "gitlab.com")

    @staticmethod
    def _fallback_version_from_repo(project: FacadeProject) -> FacadeVersion | None:
        url = project.repo_url.rstrip("/")
        if not FacadeRegistry._is_supported_archive_host(url):
            return None
        download_url = FacadeRegistry._github_archive_url(url)
        return FacadeVersion(
            version="0.0.1",
            download_url=download_url,
            dependencies=tuple(project.extra_requirements),
            deprecated=False,
        )

    @staticmethod
    def _github_archive_url(repo_url: str, ref: str | None = None) -> str:
        """Return an archive download URL for *repo_url* (GitHub or GitLab).

        The historical name kept for backward compatibility — handles both
        ``github.com`` and ``gitlab.com`` archive endpoints. Other Git hosts
        return the original URL so the caller can fall back to direct
        ``pip install git+<url>``.
        """
        url = repo_url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path.strip("/")
        if host == "github.com":
            if ref is not None:
                return f"https://api.github.com/repos/{path}/zipball/{ref}"
            return f"https://api.github.com/repos/{path}/zipball"
        if host == "gitlab.com":
            # GitLab's archive URL: encoded project ID-or-path + /archive.zip
            # The path-encoded form works without auth for public repos.
            import urllib.parse as _up

            encoded = _up.quote(path, safe="")
            r = ref or "HEAD"
            return f"https://gitlab.com/api/v4/projects/{encoded}/repository/archive.zip?sha={r}"
        # Unknown host — return the original URL; FacadeWheelBuilder will
        # fail to extract it and the caller can fall back to git+url install.
        return url

    def _build_injected_project(self, spec: CustomNodeSpec) -> FacadeProject:
        """Create a FacadeProject for a spec that has no CNR entry."""
        repo_name = repo_basename(spec.repo_url)
        canonical_name = canonicalize_project_name(spec.node_id)
        aliases = {canonical_name, canonicalize_project_name(repo_name)}

        skip_requirements = set(CustomNodeManager.DEFAULT_SKIP)
        skip_requirements.update(spec.skip_requirements)

        download_url = self._github_archive_url(spec.repo_url, spec.git_ref)
        version = FacadeVersion(
            version=spec.inject_version,
            download_url=download_url,
            dependencies=tuple(spec.extra_requirements),
            deprecated=False,
        )
        self._versions_cache[spec.node_id] = [version]

        return FacadeProject(
            canonical_name=canonical_name,
            display_name=spec.display_name,
            node_id=spec.node_id,
            repo_url=spec.repo_url,
            repo_name=repo_name,
            description=f"{spec.display_name} (injected)",
            aliases=tuple(sorted(aliases)),
            extra_requirements=tuple(spec.extra_requirements),
            skip_requirements=frozenset(skip_requirements),
            depends_on=spec.depends_on,
            latest_version=spec.inject_version,
        )

    def _runtime_dependencies(
        self,
        overlay: CustomNodeSpec | None,
        node_id: str,
        repo_name: str,
    ) -> tuple[str, ...]:
        keys = {
            node_id,
            canonicalize_project_name(node_id),
            repo_name,
            canonicalize_project_name(repo_name),
        }
        if overlay is not None:
            keys.update(
                {
                    overlay.node_id,
                    canonicalize_project_name(overlay.node_id),
                    overlay.display_name,
                    canonicalize_project_name(overlay.display_name),
                }
            )

        deps: list[str] = []
        for candidate, requirements in CUSTOM_NODE_RUNTIME_DEPS.items():
            normalized = canonicalize_project_name(candidate)
            if candidate in keys or normalized in keys:
                deps.extend(requirements)
        return tuple(deps)

    @staticmethod
    def _extract_repo_url(item: dict[str, Any]) -> str | None:
        candidate = item.get("reference") or item.get("repository")
        if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
            return candidate

        for entry in item.get("files") or ():
            if not isinstance(entry, str):
                continue
            if not entry.startswith(("http://", "https://")):
                continue
            if entry.endswith((".py", ".js")):
                continue
            return entry
        return None

    @staticmethod
    def _form_factor() -> str:
        system = platform.system().lower()
        if system == "windows":
            return "git-windows"
        if system == "darwin":
            return "git-mac"
        if system == "linux":
            return "git-linux"
        return "other"

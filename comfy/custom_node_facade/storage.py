"""fsspec storage shared by the facade's snapshot, wheel cache and snapshot output.

Every location is an fsspec URL (a local path, ``file://``, ``s3://``,
``gs://``, ``memory://`` ...) resolved with :func:`fsspec.core.url_to_fs`.
Backend settings such as an S3 endpoint come from fsspec's own configuration
(``fsspec.config.conf``, filled from ``FSSPEC_<PROTOCOL>_<OPTION>`` environment
variables and fsspec config files) or from explicit ``storage_options``.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Mapping

import fsspec
import fsspec.config
from fsspec.implementations.local import LocalFileSystem

from ..component_model.configuration import Configuration


def apply_fsspec_configuration(configuration: Configuration) -> None:
    """Map the facade's legacy endpoint flag onto fsspec's s3 configuration.

    ``--pip-facade-cache-s3-endpoint-url`` (``PIP_FACADE_CACHE_S3_ENDPOINT_URL``)
    is equivalent to ``FSSPEC_S3_ENDPOINT_URL`` and applies to every s3 URL the
    facade opens.
    """
    endpoint_url = getattr(configuration, "pip_facade_cache_s3_endpoint_url", None)
    if endpoint_url:
        fsspec.config.conf.setdefault("s3", {})["endpoint_url"] = endpoint_url


def url_to_fs(url: str | os.PathLike[str], storage_options: Mapping[str, Any] | None = None):
    return fsspec.core.url_to_fs(os.fspath(url), **(storage_options or {}))


def is_local(fs: fsspec.AbstractFileSystem) -> bool:
    return isinstance(fs, LocalFileSystem)


def publish_file(fs: fsspec.AbstractFileSystem, source: str, destination: str, *, staged: bool) -> None:
    """Upload the local file *source* to *destination* on *fs*.

    With ``staged`` the upload goes to a temporary sibling that is then moved
    over the destination, so readers see either the previous object or the
    complete new one on every filesystem. Unstaged uploads rely on the
    filesystem publishing an object only when its upload completes, which
    object stores do; a local filesystem is always staged.
    """
    # A file:// URL or a native Windows path becomes the filesystem's own
    # path form, so the temporary sibling's name and the rename target agree.
    destination = fs._strip_protocol(destination)  # pylint: disable=protected-access
    if not (staged or is_local(fs)):
        fs.put_file(source, destination)
        return
    parent = fs._parent(destination)  # pylint: disable=protected-access
    if parent and not fs.exists(parent):
        fs.makedirs(parent, exist_ok=True)
    name = destination.rstrip("/").rsplit("/", 1)[-1]
    temporary = f"{parent}/.{name}.{uuid.uuid4().hex}.tmp" if parent else f".{name}.{uuid.uuid4().hex}.tmp"
    try:
        fs.put_file(source, temporary)
        if is_local(fs):
            os.replace(temporary, destination)
        else:
            fs.mv(temporary, destination)
    except BaseException:
        try:
            if fs.exists(temporary):
                fs.rm(temporary)
        except Exception:  # noqa: BLE001 - best-effort temporary cleanup
            pass
        raise


def signature(info: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """An identity for one generation of an object from its fsspec ``info``.

    Uses the size and whichever modification time, inode, checksum or version
    fields the backend reports.
    """
    keys = (
        "size", "mtime", "created", "ino", "LastModified", "last_modified", "ETag", "etag",
        "md5Hash", "md5", "generation", "VersionId", "version_id",
    )
    return tuple((key, str(info[key])) for key in keys if key in info and info[key] is not None)

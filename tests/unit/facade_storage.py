"""Storage backends the pip facade is tested against.

The facade addresses its registry snapshot, its wheel cache and the
snapshot-pip-registry output as fsspec URLs through one code path. These
fixtures yield the same location as a plain local path, a ``file://`` URL, an
in-memory filesystem and a real S3 API (SeaweedFS in a container), so every
test parametrized over them exercises that one path with each backend.
"""
from __future__ import annotations

import shutil
import uuid

import fsspec
import fsspec.config
import pytest
import s3fs
from docker.errors import DockerException
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy


class SeaweedFSS3Container(DockerContainer):
    def __init__(self):
        super().__init__("chrislusf/seaweedfs:latest")
        self.with_exposed_ports(8333)
        self.with_command("server -ip=0.0.0.0 -dir=/data -s3 -s3.port=8333")
        self.waiting_for(LogMessageWaitStrategy("Start Seaweed S3 API Server"))

    def endpoint_url(self) -> str:
        return f"http://127.0.0.1:{self.get_exposed_port(8333)}"


@pytest.fixture(name="seaweedfs_s3_endpoint", scope="session")
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


FACADE_STORAGE_BACKENDS = ("path", "file-url", "memory", "s3")


@pytest.fixture(name="facade_storage_url", params=FACADE_STORAGE_BACKENDS)
def fixture_facade_storage_url(request, tmp_path, monkeypatch):
    """A fresh, empty directory-like location as an fsspec URL.

    The s3 backend gets its endpoint from fsspec's own configuration
    (``fsspec.config.conf``, which ``FSSPEC_S3_*`` environment variables fill),
    not from an argument, the way a deployment configures it.
    """
    backend = request.param
    if backend == "path":
        yield str(tmp_path / "storage")
    elif backend == "file-url":
        yield (tmp_path / "storage").as_uri()
    elif backend == "memory":
        root = f"memory://facade-{uuid.uuid4().hex}"
        try:
            yield root
        finally:
            memory = fsspec.filesystem("memory")
            if memory.exists(root):
                memory.rm(root, recursive=True)
    else:
        endpoint = request.getfixturevalue("seaweedfs_s3_endpoint")
        monkeypatch.setitem(
            fsspec.config.conf, "s3", {"endpoint_url": endpoint, "anon": True}
        )
        bucket = f"facade-{uuid.uuid4().hex[:20]}"
        s3fs.S3FileSystem(endpoint_url=endpoint, anon=True, skip_instance_cache=True).mkdir(bucket)
        yield f"s3://{bucket}/storage"

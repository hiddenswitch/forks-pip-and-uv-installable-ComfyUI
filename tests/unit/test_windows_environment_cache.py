from pathlib import Path
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
import shutil
import threading
import zipfile

import pytest

from ci import windows_environment_cache as cache


def environment(root, contents=b"prepared dependency"):
    for name in ("venv/Scripts/python.exe", "venv/Lib/site-packages/dependency.py", "ci-python/python.exe"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)


def test_restore_prepared_environment_onto_local_disk(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    environment(source)
    environment(destination, b"old environment")
    archive = tmp_path / "seaweedfs" / "environment.zip"
    cache.save(archive, source)
    assert cache.restore(archive, destination)
    assert (destination / "venv/Lib/site-packages/dependency.py").read_bytes() == b"prepared dependency"
    assert (destination / "ci-python/python.exe").read_bytes() == b"prepared dependency"
    assert not list(destination.glob("comfy-env-*"))
    assert (destination / ".comfy-environment-cache-hit").is_file()


@pytest.mark.parametrize("kind", ["missing", "corrupt", "incomplete", "traversal"])
def test_invalid_cache_keeps_existing_environment(tmp_path, kind):
    root = tmp_path / "local"
    root.mkdir()
    environment(root)
    (root / ".comfy-environment-cache-hit").touch()
    archive = tmp_path / "environment.zip"
    if kind == "corrupt":
        archive.write_bytes(b"interrupted upload")
    elif kind in ("incomplete", "traversal"):
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../outside" if kind == "traversal" else "venv/readme", b"bad cache")
    assert not cache.restore(archive, root)
    assert not (root / ".comfy-environment-cache-hit").exists()
    assert (root / "venv/Scripts/python.exe").read_bytes() == b"prepared dependency"
    assert not (tmp_path / "outside").exists()


def test_failed_upload_does_not_replace_previous_archive(tmp_path, monkeypatch):
    root = tmp_path / "local"
    root.mkdir()
    environment(root)
    archive = tmp_path / "cache.zip"
    archive.write_bytes(b"previous archive")

    def interrupted_copy(source, destination):
        Path(destination).write_bytes(b"partial upload")
        raise OSError("disconnected")

    monkeypatch.setattr(cache.shutil, "copyfile", interrupted_copy)
    with pytest.raises(OSError, match="disconnected"):
        cache.save(archive, root)
    assert archive.read_bytes() == b"previous archive"
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("changed", cache.INPUTS)
def test_dependency_inputs_invalidate_cache_but_checkout_changes_do_not(tmp_path, changed):
    for name in cache.INPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original\n")
    original = cache.cache_key(tmp_path)
    (tmp_path / "README.md").write_text("new commit", encoding="utf-8")
    workflow = tmp_path / '.github/workflows/test.yml'
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text('unrelated CPU job change', encoding='utf-8')
    (tmp_path / 'ci/windows_environment_cache.py').write_text('transport-only change', encoding='utf-8')
    assert cache.cache_key(tmp_path) == original
    (tmp_path / changed).write_bytes(b"original\r\n")
    assert cache.cache_key(tmp_path) == original
    (tmp_path / changed).write_text("updated dependencies", encoding="utf-8")
    assert cache.cache_key(tmp_path) != original


def test_cache_compresses_installed_binaries_and_restores_bytes(tmp_path):
    root = tmp_path / 'local'
    root.mkdir()
    contents = b'MZ' + bytes(1024 * 1024)
    environment(root, contents)
    archive = tmp_path / 'environment.zip'
    cache.save(archive, root)
    assert archive.stat().st_size < len(contents) // 10
    destination = tmp_path / 'restored'
    destination.mkdir()
    assert cache.restore(archive, destination)
    assert (destination / 'venv/Scripts/python.exe').read_bytes() == contents


@pytest.mark.parametrize('http_available', [True, False])
def test_http_restore_and_mounted_cache_fallback(tmp_path, http_available):
    source, destination, published = (tmp_path / name for name in ('source', 'destination', 'http'))
    for directory in (source, destination, published):
        directory.mkdir()
    environment(source)
    archive = tmp_path / 'mounted/environment.zip'
    cache.save(archive, source)
    if http_available:
        shutil.move(archive, published / archive.name)
    server = HTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(published)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert cache.restore(archive, destination, f'http://127.0.0.1:{server.server_port}')
        assert (destination / 'venv/Scripts/python.exe').read_bytes() == b'prepared dependency'
        assert (destination / '.comfy-environment-cache-hit').is_file()
    finally:
        server.shutdown()
        thread.join()
        server.server_close()

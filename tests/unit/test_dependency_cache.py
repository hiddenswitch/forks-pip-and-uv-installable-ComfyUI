import os
from pathlib import Path
import shutil
import subprocess

import pytest

from ci import dependency_cache as cache

pytestmark = pytest.mark.skipif(os.name != 'posix' or not shutil.which('zstd'), reason='Linux runner archive tools required')


def test_cache_roundtrip_preserves_relative_links_and_hardlink_install_source(tmp_path):
    source = tmp_path / 'source'
    (source / 'archive-v0/package').mkdir(parents=True)
    package = source / 'archive-v0/package/module.py'
    package.write_text('VALUE = 42\n')
    package.chmod(0o755)
    (source / 'wheels-v5').mkdir()
    (source / 'wheels-v5/package').symlink_to('../archive-v0/package')
    archive = tmp_path / 'shared/cache.tar.zst'
    cache.save(archive, source)
    restored = tmp_path / 'restored'
    restored.mkdir()
    assert cache.restore(archive, restored)
    restored_package = restored / 'wheels-v5/package/module.py'
    assert restored_package.read_bytes() == package.read_bytes()
    assert restored_package.stat().st_mode & 0o111
    installed = tmp_path / 'installed.py'
    os.link(restored_package, installed)
    assert installed.stat().st_ino == restored_package.stat().st_ino
    assert not list(tmp_path.glob('comfy-cache-*'))
    assert not list(archive.parent.glob('*.tmp'))


def test_failed_publication_keeps_previous_shared_entries(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'package').write_bytes(b'package contents')
    archive = tmp_path / 'shared/new.tar.zst'
    archive.parent.mkdir()
    old = archive.parent / 'previous.tar.zst'
    old.write_bytes(b'previous')

    def interrupted(source, destination):
        Path(destination).write_bytes(b'partial')
        raise OSError('connection lost')

    monkeypatch.setattr(cache.shutil, 'copyfile', interrupted)
    with pytest.raises(OSError, match='connection lost'):
        cache.save(archive, source)
    assert not archive.exists()
    assert old.read_bytes() == b'previous'
    assert list(archive.parent.iterdir()) == [old]


def test_corrupt_archive_is_not_used_as_a_partial_uv_cache(tmp_path):
    archive = tmp_path / 'broken.tar.zst'
    archive.write_bytes(b'incomplete upload')
    restored = tmp_path / 'restored'
    restored.mkdir()
    (restored / 'partial').write_bytes(b'partial')
    result = subprocess.run([os.sys.executable, str(Path(cache.__file__)), 'restore', str(archive), str(restored)], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'failed' in result.stderr
    assert list(restored.iterdir()) == []


def test_cache_key_tracks_lock_and_backend_abi(tmp_path, monkeypatch):
    lock = tmp_path / 'pylock.toml'
    lock.write_text('original')
    root = tmp_path / 'archives'
    key = cache.archive_path(lock, root)
    lock.write_text('changed')
    assert cache.archive_path(lock, root) != key
    lock.write_text('original')
    monkeypatch.setattr(cache.importlib.metadata, 'version', lambda name: 'different-torch-build')
    assert cache.archive_path(lock, root) != key

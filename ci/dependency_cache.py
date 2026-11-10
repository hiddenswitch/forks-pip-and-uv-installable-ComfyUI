"""Move immutable uv-cache snapshots between the shared volume and local disk."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sysconfig
import tempfile
import time
import uuid


CACHE_ROOT = Path('/opt/comfyui-ci-cache/dependency-archives')


def archive_path(lock_file, root=CACHE_ROOT):
    if not root.parent.is_dir() or not shutil.which('zstd'):
        return None
    try:
        torch = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
        torch = None
    environment = {
        'python': platform.python_version(), 'platform': sysconfig.get_platform(),
        'libc': platform.libc_ver(), 'torch': torch,
        'uv': subprocess.check_output(['uv', '--version'], text=True).strip(),
        'os': Path('/etc/os-release').read_text(),
        'build': {key: os.environ.get(key) for key in ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS')},
    }
    digest = hashlib.sha256(json.dumps(environment, sort_keys=True).encode())
    for path in (Path(lock_file), Path(__file__).with_name('install-locked.sh')):
        digest.update(path.read_bytes())
    return root / f'{digest.hexdigest()}.tar.zst'


def restore(archive, cache):
    if not archive.is_file():
        return False
    with tempfile.TemporaryDirectory(prefix='comfy-cache-', dir=cache.parent) as temporary:
        local = Path(temporary) / archive.name
        shutil.copyfile(archive, local)
        subprocess.run(['tar', '--zstd', '-xf', str(local), '-C', str(cache), '--no-same-owner'], check=True)
    return True


def save(archive, cache):
    if archive.is_file():
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    remote = archive.with_name(f'.{archive.name}.{uuid.uuid4().hex}.tmp')
    try:
        with tempfile.TemporaryDirectory(prefix='comfy-cache-', dir=cache.parent) as temporary:
            local = Path(temporary) / archive.name
            subprocess.run(['tar', '-I', 'zstd -T2 -1', '-cf', str(local), '-C', str(cache), '.'], check=True)
            shutil.copyfile(local, remote)
            os.replace(remote, archive)
    finally:
        remote.unlink(missing_ok=True)


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('key', 'restore', 'save'))
    parser.add_argument('path')
    parser.add_argument('cache', nargs='?', type=Path)
    args = parser.parse_args()
    if args.mode == 'key':
        try:
            archive = archive_path(args.path) if os.environ.get('COMFYUI_DEPENDENCY_CACHE', '1') != '0' else None
        except (OSError, subprocess.CalledProcessError) as error:
            logging.warning('Dependency cache unavailable: %s', error)
            archive = None
        print(archive or '')  # noqa: T201
        return
    if not args.path:
        return
    started = time.monotonic()
    archive = Path(args.path)
    try:
        if args.mode == 'restore':
            hit = restore(archive, args.cache)
            logging.info('Dependency cache %s (%.1fs)', 'hit' if hit else 'miss', time.monotonic() - started)
        else:
            save(archive, args.cache)
            logging.info('Dependency cache saved/reused (%.1fs)', time.monotonic() - started)
    except (OSError, subprocess.CalledProcessError) as error:
        logging.warning('Dependency cache %s failed: %s', args.mode, error)
        if args.mode == 'restore':
            # Never hand uv a partially extracted cache after an interrupted read.
            shutil.rmtree(args.cache)
            args.cache.mkdir()


if __name__ == '__main__':
    main()

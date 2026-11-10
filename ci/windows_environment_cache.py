"""Store prepared Windows environments on SeaweedFS; execute only on local NTFS."""

import argparse
import hashlib
from http.client import HTTPException
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import uuid
from urllib.request import urlopen
import zipfile


INPUTS = (
    "ci/setup-windows.sh",
    "ci/setup-windows.ps1",
    ".github/scripts/enable-graphics-apis.ps1",
    "ci/locks/pylock.windows-py314.toml",
    "ci/install-locked.sh",
    "tests/custom_nodes_stable_abi_requirements.txt",
)
DIRECTORIES = ("venv", "ci-python")


def cache_key(checkout):
    digest = hashlib.sha256("\0".join(DIRECTORIES).encode())
    for name in INPUTS:
        digest.update(name.encode() + b"\0")
        digest.update((checkout / name).read_bytes())
    return digest.hexdigest()


def restore(archive, root, cache_url=None):
    marker = root / ".comfy-environment-cache-hit"
    marker.unlink(missing_ok=True)
    try:
        if not cache_url and not archive.is_file():
            return False
        with tempfile.TemporaryDirectory(prefix="comfy-env-", dir=root) as temporary:
            staging = Path(temporary)
            local_archive = staging / "environment.zip"
            # One sequential network read; unpack and import on local NTFS.
            started = time.monotonic()
            if cache_url:
                try:
                    with urlopen(f"{cache_url.rstrip('/')}/{archive.name}", timeout=60) as source, local_archive.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                except (OSError, HTTPException) as error:
                    logging.warning("HTTP cache read failed; trying mounted cache: %s", error)
                    shutil.copyfile(archive, local_archive)
            else:
                shutil.copyfile(archive, local_archive)
            logging.info("Downloaded %.1f MiB in %.1fs", local_archive.stat().st_size / 2**20, time.monotonic() - started)
            started = time.monotonic()
            with zipfile.ZipFile(local_archive) as source:
                for member in source.namelist():
                    path = PurePosixPath(member)
                    if not path.parts or path.is_absolute() or ".." in path.parts or path.parts[0] not in DIRECTORIES:
                        raise ValueError(f"Unexpected cache member: {member}")
                source.extractall(staging / "unpacked")
            logging.info("Unpacked environment in %.1fs", time.monotonic() - started)
            local_archive.unlink()
            unpacked = staging / "unpacked"
            if not (unpacked / "venv/Scripts/python.exe").is_file() or not (unpacked / "ci-python").is_dir():
                raise ValueError("Incomplete cached Python environment")
            for name in DIRECTORIES:
                destination = root / name
                if destination.exists():
                    shutil.rmtree(destination)
                (unpacked / name).rename(destination)
        marker.touch()
        return True
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        logging.warning("Environment cache miss: %s", error)
        return False


def save(archive, root):
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary_remote = archive.with_name(f".{archive.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tempfile.TemporaryDirectory(prefix="comfy-env-", dir=root) as temporary:
            local_archive = Path(temporary) / "environment.zip"
            # Installed DLLs are uncompressed. A fast compression pass saves network
            # traffic and local staging space on every subsequent restore.
            with zipfile.ZipFile(local_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as target:
                for name in DIRECTORIES:
                    directory = root / name
                    if not directory.is_dir():
                        raise ValueError(f"Missing environment directory: {directory}")
                    for path in directory.rglob("*"):
                        if path.is_file():
                            target.write(path, path.relative_to(root).as_posix())
            shutil.copyfile(local_archive, temporary_remote)
            # Readers see either the previous complete archive or the new one.
            os.replace(temporary_remote, archive)
    finally:
        temporary_remote.unlink(missing_ok=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("restore", "save"))
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, default=Path("C:/"))
    parser.add_argument("--cache", type=Path, default=Path("C:/ci-cache/python-environments"))
    parser.add_argument("--cache-url", default=os.environ.get("COMFYUI_ENVIRONMENT_CACHE_URL"))
    args = parser.parse_args()
    archive = args.cache / f"{cache_key(args.checkout)}.zip"
    started = time.monotonic()
    if args.mode == "restore":
        hit = restore(archive, args.root, args.cache_url)
        logging.info("Environment cache %s: %s", "hit" if hit else "miss", archive)
    else:
        save(archive, args.root)
        logging.info("Saved environment cache: %s (%s bytes)", archive, f"{archive.stat().st_size:,}")
    logging.info("Cache %s took %.1fs", args.mode, time.monotonic() - started)


if __name__ == "__main__":
    main()

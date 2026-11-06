#!/usr/bin/env sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: $0 LOCK_FILE PYTHON" >&2
  exit 2
fi

lock_file=$1
python=$2

# uv requires reliable file locking and atomic directory creation. Runner-wide
# caches mounted over network filesystems do not provide those guarantees.
# Keep this install's cache on local temporary storage and remove it on exit.
UV_CACHE_DIR=$("$python" -c 'import os, tempfile; print(tempfile.mkdtemp(prefix="comfyui-uv-", dir=None if os.name == "nt" else "/tmp"))')
export UV_CACHE_DIR
cleanup_cache() {
  "$python" -c 'import os, shutil; shutil.rmtree(os.environ["UV_CACHE_DIR"])'
}
trap cleanup_cache EXIT HUP INT TERM

# The pinned VideoHelperSuite checkout uses setuptools without declaring a
# build-system requirement. Keep builds non-isolated (important for Torch
# extension packages), but provide the exact setuptools version present in all
# generated locks before uv starts building source distributions.
uv pip install --python "$python" "setuptools==84.0.0"

# Vendor images can contain any of several distributions that install the same
# cv2 package (NGC also ships one named simply "opencv"). Remove all ambient
# providers before the lock installs the one selected headless wheel; otherwise
# an extension compiled against the image's NumPy 1.x can survive beside the
# locked NumPy 2.x and fail during import.
uv pip uninstall --python "$python" \
  opencv \
  opencv-python \
  opencv-contrib-python \
  opencv-contrib-python-headless \
  opencv-python-headless

# Some NGC images leave cv2's unowned vendor binary in site-packages after its
# distribution is removed. Discover site-packages from the selected interpreter
# and clear only that now-unowned import package before installing the lock.
"$python" - <<'PY'
import shutil
import site
from pathlib import Path

for site_packages in map(Path, site.getsitepackages()):
    cv2 = site_packages / "cv2"
    if cv2.exists():
        shutil.rmtree(cv2)
PY

# Backend jobs deliberately provide their own Torch build. Remember it before
# installing the cross-platform dependency lock and fail if anything replaces
# it. CPU/macOS locks contain Torch themselves and therefore start without it.
torch_before=$(
  "$python" -c 'import importlib.metadata as metadata; print(metadata.version("torch"))' 2>/dev/null || true
)

uv pip install \
  --python "$python" \
  --no-build-isolation \
  -r "$lock_file"

# ComfyUI itself was omitted from every lock so the tested package is always
# this checkout. Its dependencies are already exact, so do not resolve again.
# The shared CI cache can be a network filesystem; a cache is useless for this
# local install and makes uv's interpreter/build probes perform random I/O on
# that filesystem.
uv pip install --no-cache --python "$python" --no-deps .

if [ -n "$torch_before" ]; then
  torch_after=$("$python" -c 'import importlib.metadata as metadata; print(metadata.version("torch"))')
  if [ "$torch_after" != "$torch_before" ]; then
    echo "Torch changed during locked install: $torch_before -> $torch_after" >&2
    exit 1
  fi
fi

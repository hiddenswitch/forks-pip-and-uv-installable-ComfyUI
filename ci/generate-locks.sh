#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

common=(
  pyproject.toml
  ci/headless-requirements.txt
  tests/custom_nodes_requirements.txt
  --extra dev
  --constraints tests/opencv_constraints.txt
  --excludes ci/lock-excludes.txt
  --extra-index-url https://nodes.appmana.com/simple
  --index-strategy unsafe-best-match
  --prerelease if-necessary-or-explicit
  "$@"
  --no-config
  --no-emit-package comfyui
  --no-emit-package opencv-python
  --no-emit-package opencv-contrib-python
  --no-emit-package opencv-contrib-python-headless
  --format pylock.toml
)

without_torch=(
  --no-emit-package torch
  --no-emit-package torchvision
  --no-emit-package torchaudio
)

uv pip compile "${common[@]}" \
  --constraints ci/torch-constraints.txt \
  "${without_torch[@]}" \
  --python-version 3.12 \
  --python-platform linux \
  --output-file ci/locks/pylock.linux-py312.toml

uv pip compile "${common[@]}" \
  --constraints ci/torch-constraints.txt \
  "${without_torch[@]}" \
  --override ci/numpy1-overrides.txt \
  --python-version 3.12 \
  --python-platform linux \
  --output-file ci/locks/pylock.linux-py312-numpy1.toml

uv pip compile "${common[@]}" \
  ci/xpu-requirements.txt \
  --python-version 3.14 \
  --python-platform linux \
  --torch-backend xpu \
  --output-file ci/locks/pylock.linux-py314-xpu.toml

uv pip compile "${common[@]}" \
  --constraints ci/torch-constraints.txt \
  "${without_torch[@]}" \
  --python-version 3.14 \
  --python-platform windows \
  --output-file ci/locks/pylock.windows-py314.toml

MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile "${common[@]}" \
  --constraints ci/torch-constraints.txt \
  --python-version 3.12 \
  --python-platform macos \
  --torch-backend auto \
  --output-file ci/locks/pylock.macos-py312.toml

uv pip compile "${common[@]}" \
  --constraints ci/arch-torch-constraints.txt \
  --python-version 3.14 \
  --python-platform linux \
  --torch-backend cpu \
  --output-file ci/locks/pylock.linux-py314-cpu.toml

uv pip compile \
  pyproject.toml \
  ci/headless-requirements.txt \
  ci/smoke-requirements.txt \
  --extra notebook-tests \
  --constraints tests/opencv_constraints.txt \
  --constraints ci/torch-constraints.txt \
  --prerelease if-necessary-or-explicit \
  "$@" \
  --no-config \
  --no-emit-package comfyui \
  --no-emit-package opencv-python \
  --format pylock.toml \
  --python-version 3.14 \
  --python-platform linux \
  --torch-backend cpu \
  --output-file ci/locks/pylock.linux-py314-core.toml

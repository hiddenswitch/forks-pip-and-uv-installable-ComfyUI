#!/usr/bin/env sh
set -eu

powershell.exe -NoProfile -NonInteractive -File ci/setup-windows.ps1

echo 'Install cached uv'
uv_zip="C:/ci-cache/installers/uv-0.12.2-x86_64-pc-windows-msvc.zip"
mkdir -p "C:/ci-cache/installers"
if [ ! -f "$uv_zip" ]; then
  tmp_zip="${uv_zip}.${GITHUB_RUN_ID:-0}.tmp"
  curl -fsSL --retry 5 -o "$tmp_zip" "https://github.com/astral-sh/uv/releases/download/0.12.2/uv-x86_64-pc-windows-msvc.zip"
  echo "01442d8ce5c7124151a73e697c836d252c6da853c18c73206d3cc4c2378a91d2  $tmp_zip" | sha256sum -c -
  mv -f "$tmp_zip" "$uv_zip"
fi
echo "01442d8ce5c7124151a73e697c836d252c6da853c18c73206d3cc4c2378a91d2  $uv_zip" | sha256sum -c -
unzip -oq "$uv_zip" -d "$(dirname "$(command -v uv)")"
uv --version

echo 'Restore prepared Windows environment from SeaweedFS'
C:/Python312/python.exe ci/windows_environment_cache.py restore


if [ ! -f C:/.comfy-environment-cache-hit ]; then
  uv venv --python 3.14.7 C:/venv
  uv pip install --python C:/venv/Scripts/python.exe \
    "torch==2.14.1+cu130" "torchvision==0.29.1+cu130" "torchaudio==2.11.0+cu130" \
    --index-url https://download.pytorch.org/whl/cu130
  ci/install-locked.sh ci/locks/pylock.windows-py314.toml C:/venv/Scripts/python.exe
  uv pip install --python C:/venv/Scripts/python.exe --no-deps "triton-windows==3.8.0.post29" --index-url https://pypi.org/simple
  uv pip install --python C:/venv/Scripts/python.exe --no-deps "sageattention==2.2.0.post1+cu130" --index-url https://appmana.github.io/forks-sageattention-stable-abi/cu130
  uv pip install --python C:/venv/Scripts/python.exe --no-deps "nunchaku==1.3.0.dev20260717+cu130" --index-url https://appmana.github.io/forks-nunchaku-stable-abi/cu130
  uv pip install --python C:/venv/Scripts/python.exe --no-deps -r tests/custom_nodes_stable_abi_requirements.txt --extra-index-url https://nodes.appmana.com/simple
else
  uv pip install --no-cache --python C:/venv/Scripts/python.exe --reinstall --no-deps .
fi

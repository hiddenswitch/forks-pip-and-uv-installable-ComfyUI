FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

ENV TZ="Etc/UTC" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    SAM2_BUILD_CUDA=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_RETRIES=10 \
    UV_HTTP_TIMEOUT=120 \
    UV_LINK_MODE=copy \
    UV_PYTHON=/opt/venv/bin/python \
    UV_OVERRIDE=/overrides.txt \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH

RUN apt-get update && apt-get install --no-install-recommends -y \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libsndfile1 libxcb1 \
    libcairo2 zip unzip ca-certificates git build-essential \
    && uv venv /opt/venv --python python3.12 \
    && touch /overrides.txt \
    && rm -rf /var/lib/apt/lists/*

# TheRock separates the Torch frontend from architecture-specific ROCm
# devices. device-all is intentional: this one image serves every architecture
# published by the selected stable release, including RX 7600/gfx1102.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --python /opt/venv/bin/python \
    --index-url https://stable.repo.amd.com/rocm/whl-next/ \
    "torch==2.13.0+rocm10.0.0" \
    "torchvision==0.28.0+rocm10.0.0" \
    "torchaudio==2.11.0.2+rocm10.0.0" \
    && for arch in gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201 gfx1250 gfx908 gfx90a gfx942 gfx950; do \
         uv pip install --python /opt/venv/bin/python --index-url https://stable.repo.amd.com/rocm/whl-next/ \
           "amd-torch-device-${arch}==2.13.0+rocm10.0.0" \
           "amd-torchvision-device-${arch}==0.28.0+rocm10.0.0"; \
       done \
    && uv pip list --python /opt/venv/bin/python --format freeze --exclude-editable > /overrides.txt \
    && printf "%s\n" \
       "opencv-python; python_version < '0'" \
       "opencv-python-headless; python_version < '0'" \
       "opencv-contrib-python; python_version < '0'" \
       >> /overrides.txt \
    && printf "%s\n" "comfyui; python_version < '0'" > /custom-node-overrides.txt \
    && cat /overrides.txt >> /custom-node-overrides.txt

COPY pyproject.toml README.md /workspace/project/
COPY tests/custom_nodes_requirements.txt /workspace/requirements/

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --extra dev \
      -r /workspace/project/pyproject.toml \
    && uv pip install --no-build-isolation \
      --override /custom-node-overrides.txt \
      -r /workspace/requirements/custom_nodes_requirements.txt \
      --extra-index-url https://nodes.appmana.com/simple \
      --index-strategy unsafe-best-match

ADD . /workspace/src
WORKDIR /workspace
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-deps "comfyui@./src"

RUN python - <<'PY'
import torch
assert torch.__version__.startswith("2.13.0")
assert torch.version.hip is not None
print("torch", torch.__version__, "hip", torch.version.hip)
PY

WORKDIR /workspace/src
EXPOSE 8188
CMD ["python", "-m", "comfy.cmd.main", "--listen", "--reserve-vram=0", "--logging-level=INFO", "--enable-cors"]

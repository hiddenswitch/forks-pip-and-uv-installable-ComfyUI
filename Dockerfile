ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

ARG STABLE_ABI_CUDA=cu130
ENV TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:True \
    SAM2_BUILD_CUDA=0 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_RETRIES=10 \
    UV_HTTP_TIMEOUT=120 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1 \
    UV_CACHE_DIR=/root/.cache/uv

# NGC's Python environment is an authored, tested CUDA ABI set. OpenCV is one
# deliberate exception: NGC's provider installs the same cv2 import as the
# headless distribution required by ComfyUI and cannot coexist with it.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
       ffmpeg libsm6 libxext6 libcairo2-dev libxcb1 zip unzip \
    && opencv_packages="$(uv pip freeze --system | sed -n -E '/^opencv/I{s/==.*//;p}' || true)" \
    && if [ -n "$opencv_packages" ]; then uv pip uninstall --system $opencv_packages; fi \
    && rm -rf /usr/local/lib/python3.12/dist-packages/cv2/ \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /workspace /root/.cache/uv

# Freeze the remaining distributions supplied by NGC. The named exceptions are
# packages for which ComfyUI deliberately selects a different compatible
# provider/version: OpenCV, the Transformers-facing hub/tokenizer pair, and
# TorchAudio (the CPU wheel avoids its exact CUDA-minor check while its tensor
# transforms continue to use the NGC Torch runtime on CUDA tensors).
RUN uv pip list --system --format freeze --exclude-editable \
      --exclude huggingface-hub \
      --exclude protobuf \
      --exclude requests \
      --exclude tokenizers \
      > /workspace/ngc-preserved.txt \
    && printf "%s\n" \
       "opencv-python; python_version < '0'" \
       "opencv-python-headless; python_version < '0'" \
       "opencv-contrib-python; python_version < '0'" \
       > /workspace/resolver-overrides.txt \
    && printf "%s\n" \
       "comfyui; python_version < '0'" \
       > /workspace/custom-node-overrides.txt \
    && cat /workspace/resolver-overrides.txt >> /workspace/custom-node-overrides.txt
ENV UV_OVERRIDE=/workspace/resolver-overrides.txt

# Dependency inputs are copied before the source tree so ordinary code changes
# reuse the complete Python/custom-node layer on ephemeral builders.
COPY pyproject.toml README.md /workspace/project/
COPY tests/custom_nodes_requirements.txt tests/custom_nodes_stable_abi_requirements.txt /workspace/requirements/

# Bake the application, test tooling, and custom-node dependency closure into
# the candidate. Hardware jobs consume this exact image and perform no installs.
# Custom-node packages declare an unbounded dependency on comfyui; the index
# would satisfy it with a published release whose pins downgrade comfy-aimdo
# and comfy-kitchen, so that requirement is overridden away (the checkout is
# installed below, and ci/lock-excludes.txt does the same for the locks).
# The NGC snapshot doubles as the constraint set so a node requirement cannot
# move a preserved distribution; the check below then only confirms it.
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-deps \
      --index-url https://download.pytorch.org/whl/cpu \
      "torchaudio==2.11.0+cpu" \
    && uv pip install --extra dev \
      -r /workspace/project/pyproject.toml \
    && uv pip install --no-build-isolation \
      --override /workspace/custom-node-overrides.txt \
      --constraints /workspace/ngc-preserved.txt \
      -r /workspace/requirements/custom_nodes_requirements.txt \
      --extra-index-url https://nodes.appmana.com/simple \
      --index-strategy unsafe-best-match \
    && if [ -n "$STABLE_ABI_CUDA" ]; then \
         uv pip install --no-deps \
           "sageattention==2.2.0.post1+${STABLE_ABI_CUDA}" \
           --index-url "https://appmana.github.io/forks-sageattention-stable-abi/${STABLE_ABI_CUDA}"; \
         uv pip install --no-deps \
           "nunchaku==1.3.0.dev20260717+${STABLE_ABI_CUDA}" \
           --index-url "https://appmana.github.io/forks-nunchaku-stable-abi/${STABLE_ABI_CUDA}"; \
         uv pip install --no-deps \
           "nccl-usb4 @ https://github.com/AppMana/forks-nccl-rdma-routing/releases/download/v2.32.3/nccl_usb4-2.32.3-py3-none-manylinux_2_28_x86_64.whl" \
           --extra-index-url https://nodes.appmana.com/simple; \
         python3 -c "import nccl_usb4; st = nccl_usb4.activate(); assert st['active'] and st['tbrxe_loaded'], st"; \
         ldconfig; \
         uv pip install --no-deps \
           -r /workspace/requirements/custom_nodes_stable_abi_requirements.txt \
           --extra-index-url https://nodes.appmana.com/simple; \
       fi

ADD . /workspace/src
WORKDIR /workspace
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-deps "comfyui@./src"

# Make the preservation rule executable: every distribution from the NGC
# snapshot must still be installed at precisely the version NVIDIA supplied.
RUN sort /workspace/ngc-preserved.txt > /workspace/ngc-preserved.sorted \
    && uv pip list --system --format freeze --exclude-editable | sort > /workspace/final-environment.sorted \
    && missing="$(comm -23 /workspace/ngc-preserved.sorted /workspace/final-environment.sorted)" \
    && if [ -n "$missing" ]; then \
         printf 'NGC packages changed during image build:\n%s\n' "$missing" >&2; \
         exit 1; \
       fi

WORKDIR /workspace/src
EXPOSE 8188
CMD ["python", "-m", "comfy.cmd.main", "--listen", "--use-sage-attention", "--reserve-vram=0", "--logging-level=INFO", "--enable-cors"]

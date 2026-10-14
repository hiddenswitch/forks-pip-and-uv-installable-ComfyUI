# CI dependency locks

The CI environments install from the platform-specific `pylock.*.toml` files in
this directory. ComfyUI is installed from the checkout. CUDA and Windows locks
omit Torch, while those jobs retain the Torch build
provided or explicitly selected by that backend. CUDA image builds freeze the
Python environment supplied by NGC, except for named compatibility exceptions,
as resolver overrides before adding ComfyUI. Hardware jobs test that immutable
candidate image without installing anything. All locks use the single
headless OpenCV distribution in `headless-requirements.txt` instead of
installing multiple distributions that provide the same `cv2` module.

Lock changes are reviewed and committed with the dependency change that caused
them. CI installs the locks without resolving dependency versions. Lock
generation intentionally uses `--no-config`: the files listed by the script are
the complete inputs, so an ambient project or user uv configuration cannot
silently change a lock. The AppMana node index is combined with the public
indexes only while generating the trusted lock; installs use the exact URLs and
hashes recorded in the resulting pylock files.

`torch-constraints.txt` holds the default Torch line locks resolve against, so
`--upgrade` cannot move Torch (or the Triton the Linux locks emit) as a side
effect of an unrelated dependency change. Bump it deliberately.

Pre-releases are selected only when a project explicitly requests one or no
stable distribution supports the target. This keeps Python 3.14 workable
without turning every dependency into a moving nightly-build target.

The macOS lock targets macOS 14 and Python 3.12. It includes the native macOS
PyTorch 2.13 wheels; Python 3.15 is not one of the CI targets.

The XPU lock includes native PyTorch 2.14.1, torchvision 0.29.1, TorchAudio
2.11.0 and their Intel runtime dependencies. The XPU job uses an isolated
Python 3.14.7 environment; the base image supplies the GPU driver libraries.
`xpu-requirements.txt` pins this backend independently of the CUDA and CPU jobs.

`pylock.linux-py312-numpy1.toml` is the compatibility lock for the CUDA 12.8
NGC image. Its Torch and bundled extensions use the NumPy 1 ABI, so the whole
NumPy/OpenCV/SciPy set is resolved together under `numpy<2`; newer backends use
the regular Python 3.12 lock. `numpy1-overrides.txt` replaces the normal exact
OpenCV pin with the last NumPy-1-compatible wheel for that resolution.

`pylock.linux-py314-core.toml` intentionally omits the development extra and
custom nodes. Master uses it for a short wheel/install/import smoke; the larger
Python 3.14 CPU lock remains the single broad develop validation environment.
The core lock includes the `notebook-tests` extra for the notebook regression
in the master smoke job; other test environments do not require a kernel.

## Accelerator image promotion

CUDA and ROCm builds first publish immutable candidate tags such as
`<full-commit>-cuda133-<run>-<attempt>.dev` and
`<full-commit>-rocm-<run>-<attempt>.dev`. Hardware jobs execute
the application and tests already baked into those images; they do not mutate
the environment. Each accelerator candidate runs the full unit suite (apart
from the browser-only Playwright conversion test), the quantization suite, and
hardware-specific smoke tests. Only after every required check succeeds is the
exact manifest retagged with the short commit and branch channel (`develop-*`
or `latest-*`).
Those promoted images also carry inline BuildKit cache metadata and are the
shared cache source for later builds. This reuses the stable dependency layer
without separately uploading the same accelerator layers to a GHA cache.

OCI defines annotations for source, revision, version, and base-image identity,
but it does not define a prerelease flag. The `.dev` tag suffix is therefore the
human-visible candidate channel; content digests provide the immutable identity.

The CUDA build has explicit, narrow exceptions to NGC's Python package set.
Its OpenCV provider is replaced by the single headless contrib provider required
by ComfyUI; `huggingface-hub` and `tokenizers` resolve with the supported
Transformers line; `protobuf` and `requests` resolve with ComfyUI's API and
object-store clients; and the official CPU TorchAudio wheel avoids TorchAudio's
exact CUDA-minor check while retaining Torch-backed tensor transforms. The
build compares every other NGC distribution before and after installation and
fails if any version changes.

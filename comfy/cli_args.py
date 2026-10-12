"""
Stub parser for upstream merge compatibility.

All ``parser.add_argument`` calls are no-ops. Real CLI parsing is handled
by Typer in ``comfy.cmd.cli``. The module-level ``args`` attribute returns
the current execution context's ``Configuration`` via a module property.

When upstream adds new ``parser.add_argument(...)`` lines, they merge
cleanly because the stub silently accepts them. You must also:
  1. Add the corresponding field to ``cli_args_types.py`` Configuration.
  2. Add a ``typer.Option`` to the appropriate command in ``comfy/cmd/cli.py``.
"""
from __future__ import annotations

import argparse
import os
import sys

from packaging import version

from .cli_args_types import Configuration, LatentPreviewMethod, PerformanceFeature
from .component_model.module_property import create_module_properties
class _StubGroup:
    def add_argument(self, *a, **kw):
        return self

    def add_mutually_exclusive_group(self, **kw):
        return _StubGroup()


class _StubParser(_StubGroup):
    def parse_args(self, args=None):
        return Configuration()

    def parse_known_args(self, args=None, **kw):
        return Configuration(), []

    def parse_known_args_with_config_files(self, args=None, **kw):
        from .cli_args_types import ParsedArgs
        return ParsedArgs(Configuration(), [], [])

parser = _StubParser()


# ===================================================================
# Upstream add_argument calls — all no-ops, kept for clean merges.
# ===================================================================

parser.add_argument('-w', "--cwd", type=str, default=None,
                    help="Specify the working directory. If not set, this is the current working directory. models/, input/, output/ and other directories will be located here by default.")
parser.add_argument("--base-paths", type=str, nargs='+', default=[],
                    help="Additional base paths for custom nodes, models and inputs.")
parser.add_argument('-H', "--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0,::",
                    help="Specify the IP address to listen on (default: 127.0.0.1). You can give a list of ip addresses by separating them with a comma like: 127.2.2.2,127.3.3.3 If --listen is provided without an argument, it defaults to 0.0.0.0,:: (listens on all ipv4 and ipv6)")
parser.add_argument("--port", type=int, default=8188, help="Set the listen port.")
parser.add_argument("--enable-cors-header", type=str, default=None, metavar="ORIGIN", nargs="?", const="*",
                    help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'.")
parser.add_argument("--max-upload-size", type=float, default=100, help="Set the maximum upload size in MB.")
parser.add_argument("--base-directory", type=str, default=None, help="Set the ComfyUI base directory for models, custom_nodes, input, output, temp, and user directories.")
parser.add_argument("--extra-model-paths-config", type=str, default=[], metavar="PATH", nargs='+',
                    help="Load one or more extra_model_paths.yaml files.")
parser.add_argument("--output-directory", type=str, default=None, help="Set the ComfyUI output directory.")
parser.add_argument("--temp-directory", type=str, default=None, help="Set the ComfyUI temp directory.")
parser.add_argument("--input-directory", type=str, default=None, help="Set the ComfyUI input directory.")
parser.add_argument("--auto-launch", action="store_true", help="Automatically launch ComfyUI in the default browser.")
parser.add_argument("--disable-auto-launch", action="store_true", help="Disable auto launching the browser.")
parser.add_argument("--cuda-device", type=str, default=None, metavar="DEVICE_ID",
                    help="Set the ids of cuda devices this instance will use, as a comma-separated list (e.g. '0' or '0,1'), or 'all' to leave all currently visible devices available. All other devices will not be visible.")
parser.add_argument("--torch-device", type=str, default=None, metavar="DEVICE",
                    help="Set the torch device by name, e.g. cuda:1, cpu, mps. Overrides --cuda-device and --cpu.")
parser.add_argument("--default-device", type=int, default=None, metavar="DEFAULT_DEVICE_ID", help="Set the id of the default device, all other devices will stay visible.")
parser.add_argument("--rank", type=int, default=None, help="Global process rank. Defaults to RANK.")
parser.add_argument("--world-size", type=int, default=None, help="Total process count. Defaults to WORLD_SIZE.")
parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help="Node-local process rank. Defaults to LOCAL_RANK.")
parser.add_argument("--local-world-size", type=int, default=None, help="Number of local processes. Defaults to LOCAL_WORLD_SIZE.")
parser.add_argument("--master-addr", type=str, default=None, help="Process-group rendezvous host. Defaults to MASTER_ADDR.")
parser.add_argument("--master-port", type=int, default=None, help="Process-group rendezvous port. Defaults to MASTER_PORT.")
parser.add_argument("--pipeline-parallel-size", type=int, default=None, help="Number of pipeline stages. Defaults to world size for external launchers and selected device count otherwise.")
parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of tensor-parallel ranks. One keeps the ordinary model path.")
parser.add_argument("--nccl-proto", choices=("auto", "simple", "ll", "ll128"), default="auto", help="Select the NCCL collective protocol. auto preserves NCCL tuning.")
parser.add_argument("--distributed-executor-backend", choices=("auto", "peer", "mp", "external_launcher"), default="auto", help="Pipeline executor backend.")

cm_group = parser.add_mutually_exclusive_group()
cm_group.add_argument("--cuda-malloc", action="store_true", help="Enable cudaMallocAsync.")
cm_group.add_argument("--disable-cuda-malloc", action="store_true", default=True, help="Disable cudaMallocAsync.")

fp_group = parser.add_mutually_exclusive_group()
fp_group.add_argument("--force-fp32", action="store_true", help="Force fp32.")
fp_group.add_argument("--force-fp16", action="store_true", help="Force fp16.")
fp_group.add_argument("--force-bf16", action="store_true", help="Force bf16.")

fpunet_group = parser.add_mutually_exclusive_group()
fpunet_group.add_argument("--fp32-unet", action="store_true", help="Run the diffusion model in fp32.")
fpunet_group.add_argument("--fp64-unet", action="store_true", help="Run the diffusion model in fp64.")
fpunet_group.add_argument("--bf16-unet", action="store_true", help="Run the diffusion model in bf16.")
fpunet_group.add_argument("--fp16-unet", action="store_true", help="Run the diffusion model in fp16.")
fpunet_group.add_argument("--fp8_e4m3fn-unet", action="store_true", help="Store unet weights in fp8_e4m3fn.")
fpunet_group.add_argument("--fp8_e5m2-unet", action="store_true", help="Store unet weights in fp8_e5m2.")
fpunet_group.add_argument("--fp8_e8m0fnu-unet", action="store_true", help="Store unet weights in fp8_e8m0fnu.")
parser.add_argument("--fp8-storage", action=argparse.BooleanOptionalAction, default=True, help="Preserve native fp8 checkpoint weights as resident fp8 storage when the device supports fp8 storage with upcasted math.")

fpvae_group = parser.add_mutually_exclusive_group()
fpvae_group.add_argument("--fp16-vae", action="store_true", help="Run the VAE in fp16.")
fpvae_group.add_argument("--fp32-vae", action="store_true", help="Run the VAE in full precision fp32.")
fpvae_group.add_argument("--bf16-vae", action="store_true", help="Run the VAE in bf16.")

parser.add_argument("--cpu-vae", action="store_true", help="Run the VAE on the CPU.")

fpte_group = parser.add_mutually_exclusive_group()
fpte_group.add_argument("--fp8_e4m3fn-text-enc", action="store_true", help="Store text encoder weights in fp8 (e4m3fn).")
fpte_group.add_argument("--fp8_e5m2-text-enc", action="store_true", help="Store text encoder weights in fp8 (e5m2).")
fpte_group.add_argument("--fp16-text-enc", action="store_true", help="Store text encoder weights in fp16.")
fpte_group.add_argument("--fp32-text-enc", action="store_true", help="Store text encoder weights in fp32.")
fpte_group.add_argument("--bf16-text-enc", action="store_true", help="Store text encoder weights in bf16.")

parser.add_argument("--fp16-intermediates", action="store_true", help="Experimental: Use fp16 for intermediate tensors between nodes instead of fp32.")

parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1, help="Use torch-directml.")
parser.add_argument("--oneapi-device-selector", type=str, default=None, metavar="SELECTOR_STRING", help="Sets the oneAPI device(s).")
parser.add_argument("--supports-fp8-compute", action="store_true", help="Act as if device supports fp8 compute.")
parser.add_argument("--fp8-materialization", choices=("auto", "torch", "comfy_kitchen"), default="auto", help="Select how FP8 weights are materialized to bf16/fp16/fp32. auto leaves room for benchmark-selected lowerings; torch uses the graph-visible torch op; comfy_kitchen uses comfy_kitchen's registered backend.")

parser.add_argument("--enable-triton-backend", action="store_true", help="Enable the Triton backend in comfy-kitchen.")
parser.add_argument("--disable-triton-backend", action="store_true", help="Force-disable the comfy-kitchen Triton backend.")
parser.add_argument("--preview-method", type=LatentPreviewMethod, default=LatentPreviewMethod.Auto, help="Default preview method.")
parser.add_argument("--preview-size", type=int, default=512, help="Sets the maximum preview size for sampler nodes.")

cache_group = parser.add_mutually_exclusive_group()
cache_group.add_argument("--cache-ram", nargs='*', type=float, default=[], metavar="GB", help="Use RAM pressure caching with the specified headroom thresholds. This is the default caching mode. The first value sets the active-cache threshold; the optional second value sets the inactive-cache/pin threshold. Defaults when no values are provided: active 10%% of system RAM (min 2GB, max 10GB), inactive 100%% of system RAM (max 96GB).")
cache_group.add_argument("--cache-classic", action="store_true", help="Use the old style (aggressive) caching.")
cache_group.add_argument("--cache-lru", type=int, default=0, help="Use LRU caching with a maximum of N node results cached. May use more RAM/VRAM.")
cache_group.add_argument("--cache-none", action="store_true", help="Reduced RAM/VRAM usage at the expense of executing every node for each run.")
cache_group.add_argument("--high-ram", action="store_true", help="Can improve performance slightly on high RAM or on systems where pagefile use is preferred over model loading.")

attn_group = parser.add_mutually_exclusive_group()
attn_group.add_argument("--use-split-cross-attention", action="store_true", help="Use split cross attention.")
attn_group.add_argument("--use-quad-cross-attention", action="store_true", help="Use sub-quadratic cross attention.")
attn_group.add_argument("--use-pytorch-cross-attention", action="store_true", help="Use PyTorch 2.0 cross attention.", default=True)
attn_group.add_argument("--use-sage-attention", action="store_true", help="Use sage attention.")
attn_group.add_argument("--use-flash-attention", action="store_true", help="Use FlashAttention.")
attn_group.add_argument("--use-ck-attention", action="store_true", help="Use Comfy Kitchen attention.")

parser.add_argument("--disable-xformers", action="store_true", help="Disable xformers.")

upcast = parser.add_mutually_exclusive_group()
upcast.add_argument("--force-upcast-attention", action="store_true", help="Force attention upcasting.")
upcast.add_argument("--dont-upcast-attention", action="store_true", help="Disable attention upcasting.")

parser.add_argument("--enable-manager", action="store_true", help="Enable the ComfyUI-Manager feature.")
manager_group = parser.add_mutually_exclusive_group()
manager_group.add_argument("--disable-manager-ui", action="store_true", help="Disables only the ComfyUI-Manager UI and endpoints. Scheduled installations and similar background tasks will still operate.")
manager_group.add_argument("--enable-manager-legacy-ui", action="store_true", help="Enables the legacy UI of ComfyUI-Manager. Implies --enable-manager.")

vram_group = parser.add_mutually_exclusive_group()
vram_group.add_argument("--gpu-only", action="store_true", help="Store and run everything (text encoders/CLIP models, etc... on the GPU).")
vram_group.add_argument("--highvram", action="store_true", help="By default models will be unloaded to CPU memory after being used. This option keeps them in GPU memory.")
vram_group.add_argument("--normalvram", action="store_true", help="Used to force normal vram use if lowvram gets automatically enabled.")
vram_group.add_argument("--lowvram", action="store_true", help="Doesn't do anything if dynamic vram is enabled. If dynamic vram isn't being used this option makes the text encoders run on the CPU.")
vram_group.add_argument("--novram", action="store_true", help="When lowvram isn't enough.")
vram_group.add_argument("--cpu", action="store_true", help="To use the CPU for everything (slow).")

parser.add_argument("--reserve-vram", type=float, default=None, help="Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reserved depending on your OS.")
parser.add_argument("--vram-headroom", type=float, default=0, help="Set the amount of vram in GB for DynamicVRAM to maintain as extra headroom above default. ComfyUI will try and keep this much VRAM completely free and unused, even counting VRAM from other apps.")
parser.add_argument("--disable-nvml-pressure", action="store_true", help="Use CUDA instead of NVML for DynamicVRAM memory pressure.")

parser.add_argument("--async-offload", nargs='?', const=2, type=int, default=None, metavar="NUM_STREAMS", help="Use async weight offloading. An optional argument controls the amount of offload streams. Default is 2. Enabled by default on Nvidia.")
parser.add_argument("--disable-async-offload", action="store_true", help="Disable async weight offloading.")
parser.add_argument("--disable-dynamic-vram", action="store_true", help="Disable dynamic VRAM and use estimate based model loading.")
parser.add_argument("--enable-dynamic-vram", action="store_true", help="Enable dynamic VRAM on systems where it's not enabled by default.")
parser.add_argument("--fast-disk", action="store_true", help="Force disk-backed dynamic loading and offload over unpinned RAM. Can be faster for users with fast NVME disks.")
parser.add_argument("--disable-fast-disk", action="store_true", help="Disable disk-backed dynamic loading and offload over unpinned RAM. Overrides --fast-disk.")
parser.add_argument("--disable-cuda-graphs", action="store_true", help="Disable CUDA graphs.")
parser.add_argument("--disable-comfy-compiler", action="store_true", help="Disable the Comfy model compiler, including its CUDA graph subfeature.")
parser.add_argument("--assert-graph-breaks", action="store_true", help="Fail on Comfy model compiler graph breaks.")
parser.add_argument("--disable-comfy-compiler", action="store_true", help="Disable the Comfy model compiler, including its CUDA graph subfeature.")
parser.add_argument("--assert-graph-breaks", action="store_true", help="Fail on Comfy model compiler graph breaks.")

parser.add_argument("--force-non-blocking", action="store_true", help="Force non-blocking operations.")
parser.add_argument("--default-hashing-function", type=str, choices=['md5', 'sha1', 'sha256', 'sha512'], default='sha256', help="Allows you to choose the hash function to use for duplicate filename / contents comparison. Default is sha256.")

parser.add_argument("--disable-smart-memory", action="store_true", help="Force ComfyUI to agressively offload to regular ram instead of keeping models in vram when it can.")
parser.add_argument("--deterministic", action="store_true", help="Make pytorch use slower deterministic algorithms when it can. Note that this might not make images deterministic in all cases.")

parser.add_argument("--fast", nargs="*", type=PerformanceFeature, help="Enable some untested and potentially quality deteriorating optimizations. This is used to test new features so using it might crash your comfyui. --fast with no arguments enables everything. You can pass a list specific optimizations if you only want to enable specific ones. Current valid optimizations: {}".format(" ".join(map(lambda c: c.value, PerformanceFeature))))

parser.add_argument("--debug-hang", action="store_true", help="Enable stack trace dumps on Ctrl-C for debugging hangs.")

parser.add_argument("--disable-pinned-memory", action="store_true", help="Disable pinned memory use.")
parser.add_argument("--pinned-memory-reserve", type=float, default=None, help="GB of the effective memory limit (host RAM, or the cgroup limit in a container) that pinned memory always leaves free, checked against memory available at each pin. Default: the larger of 4 GB and 10%% of the limit.")
parser.add_argument("--max-pinned-memory", type=float, default=None, help="Cap in GB on pinned host memory. Default and upper bound: the effective memory limit minus --pinned-memory-reserve.")

parser.add_argument("--mmap-torch-files", action="store_true", help="Use mmap for ckpt/pt files.")
parser.add_argument("--disable-mmap", action="store_true", help="Don't use mmap for safetensors.")

parser.add_argument("--dont-print-server", action="store_true", help="Don't print server output.")
parser.add_argument("--quick-test-for-ci", action="store_true", help="Quick test for CI.")
parser.add_argument("--windows-standalone-build", default=hasattr(sys, 'frozen') and getattr(sys, 'frozen'),
                    action="store_true", help="Windows standalone build.")

parser.add_argument("--disable-metadata", action="store_true", help="Disable saving prompt metadata.")
parser.add_argument("--disable-all-custom-nodes", action="store_true", help="Disable all custom nodes.")
parser.add_argument("--whitelist-custom-nodes", type=str, nargs='+', default=[], help="Custom nodes to load.")
parser.add_argument("--blacklist-custom-nodes", type=str, nargs='+', default=[], help="Custom nodes to never load.")
parser.add_argument("--disabled-nodes-config", type=str, default=None, metavar="PATH", help="Path to a YAML file listing node IDs to disable.")
parser.add_argument("--disable-partner-nodes", action="store_true", help="Disable loading partner nodes (the paid API nodes). The frontend keeps its internet access.")
parser.add_argument("--offline", action="store_true", help="Prevent the frontend from communicating with the internet by setting a restrictive Content-Security-Policy header. Also disables partner nodes.")
parser.add_argument("--disable-api-nodes", action="store_true", help="Deprecated alias for --offline. Use --offline, or --disable-partner-nodes to only disable partner nodes.")
parser.add_argument("--enable-eval", action="store_true", help="Enable eval nodes.")

parser.add_argument("--multi-user", action="store_true", help="Enable per-user storage.")
parser.add_argument("--create-directories", action="store_true", help="Create default directories then exit.")
parser.add_argument("--log-stdout", action="store_true", help="Send output to stdout.")

parser.add_argument("--plausible-analytics-base-url", required=False, help="Analytics base URL.")
parser.add_argument("--plausible-analytics-domain", required=False, help="Analytics domain.")
parser.add_argument("--analytics-use-identity-provider", action="store_true", help="Use identity for analytics.")
parser.add_argument("--distributed-queue-connection-uri", type=str, default=None, help="AMQP URL.")
parser.add_argument('--distributed-queue-worker', required=False, action="store_true", help='Run as worker.')
parser.add_argument('--distributed-queue-frontend', required=False, action="store_true", help='Run as frontend.')
parser.add_argument("--distributed-queue-name", type=str, default="comfyui", help="Queue name.")
parser.add_argument("--external-address", required=False, help="External address base URL.")
parser.add_argument("--logging-level", type=str, default='INFO', help='Logging level.')
parser.add_argument("--aimdo-logging-level", type=str, default='ERROR', help="Log level of comfy-aimdo's native library.")
parser.add_argument("--disable-known-models", action="store_true", help="Disable known model downloads.")
parser.add_argument("--max-queue-size", type=int, default=65536, help="Max queue size.")

parser.add_argument("--otel-service-name", type=str, default="comfyui", help="OTel service name.")
parser.add_argument("--otel-service-version", type=str, default="0.0.1", help="OTel service version.")
parser.add_argument("--otel-exporter-otlp-endpoint", type=str, default=None, help="OTLP endpoint.")
parser.add_argument("--force-channels-last", action="store_true", help="Force channels last format.")
parser.add_argument("--force-hf-local-dir-mode", action="store_true", help="HuggingFace local_dir mode.")
parser.add_argument("--enable-video-to-image-fallback", action="store_true", help="Enable video-to-image fallback.")

parser.add_argument("--front-end-version", type=str, default="comfyanonymous/ComfyUI@latest", help="Frontend version.")
parser.add_argument('--panic-when', nargs='+', type=str, default=[], help="Exception class names to panic on.")
parser.add_argument("--front-end-root", type=str, default=None, help="Local frontend directory.")
parser.add_argument("--executor-factory", type=str, default="ThreadPoolExecutor", help="Executor type.")
parser.add_argument("--openai-api-key", required=False, type=str, default=None, help="OpenAI API key.")
parser.add_argument("--ideogram-api-key", required=False, type=str, default=None, help="Ideogram API key.")
parser.add_argument("--anthropic-api-key", required=False, type=str, help="Anthropic API key.")
parser.add_argument("--user-directory", type=str, default=None, help="User directory.")
parser.add_argument("--models-directory", type=str, default=None, help="Models directory.")
parser.add_argument("--enable-compress-response-body", action="store_true", help="Compress response body.")
parser.add_argument("--comfy-api-base", type=str, default="https://api.comfy.org", help="ComfyUI API base URL.")
parser.add_argument("--block-runtime-package-installation", action="store_true", help="Block runtime installs.")
parser.add_argument("--workflows", type=str, nargs='+', default=[], help="Execute workflows and exit.")
parser.add_argument("--prompt", type=str, default=None, help="Override positive prompt.")
parser.add_argument("--negative-prompt", type=str, default=None, help="Override negative prompt.")
parser.add_argument("--steps", type=int, default=None, help="Override sampling steps.")
parser.add_argument("--seed", type=int, default=None, help="Override seed.")
parser.add_argument("--image", type=str, nargs='+', default=None, help="Override image inputs.")
parser.add_argument("--video", type=str, nargs='+', default=None, help="Override video inputs.")
parser.add_argument("--audio", type=str, nargs='+', default=None, help="Override audio inputs.")
parser.add_argument("-o", "--output", type=str, default=None, help="Override output directory.")
parser.add_argument("--guess-settings", action="store_true", help="Auto-detect best settings.")

parser.add_argument("--disable-requests-caching", action="store_true", help="Disable requests caching.")
parser.add_argument("--disable-manager-model-fallback", action="store_true", default=False, help="Disable manager model fallback.")
parser.add_argument("--refresh-manager-models", action="store_true", default=False, help="Fetch latest model list.")
parser.add_argument("--disable-civitai-model-fallback", action="store_true", default=False, help="Disable on-demand Civitai model lookup. Civitai fallback is auto-enabled when CIVITAI_API_TOKEN is set.")


class EnumAction:
    def __init__(self, **kw):
        pass

    def __call__(self, *a, **kw):
        pass


DEFAULT_VERSION_STRING = "comfyanonymous/ComfyUI@latest"


_module_properties = create_module_properties()


@_module_properties.getter
def _args() -> Configuration:
    from .execution_context import current_execution_context
    return current_execution_context().configuration


class _ConfigurationProxy:
    """Keep legacy ``from comfy.cli_args import args`` access context-local."""

    def __getattr__(self, name):
        return getattr(_args(), name)

    def __setattr__(self, name, value):
        setattr(_args(), name, value)

    def __getitem__(self, key):
        return _args()[key]

    def __setitem__(self, key, value):
        _args()[key] = value

    def __contains__(self, key):
        return key in _args()

    def __iter__(self):
        return iter(_args())

    def __len__(self):
        return len(_args())

    def __repr__(self):
        return repr(_args())


args = _ConfigurationProxy()

database_default_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "user", "comfyui.db")
)
parser.add_argument(
    "--database-url",
    type=str,
    default=None,
    help="Specify the database URL, e.g. 'sqlite:///:memory:'. Defaults to 'comfyui.db' in the effective user directory.",
)
parser.add_argument("--enable-assets", action="store_true", help="Enable the assets system.")
parser.add_argument("--enable-asset-hashing", action="store_true", help="Compute blake3 content hashes when scanning assets.")
parser.add_argument("--disable-assets-autoscan", action="store_true", help="Disable asset scanning on startup for database synchronization.")
parser.add_argument("--feature-flag", type=str, action="append", default=[], metavar="KEY[=VALUE]", help="Set a server feature flag.")
parser.add_argument("--list-feature-flags", action="store_true", help="Print known CLI-settable feature flags as JSON and exit.")

def default_configuration() -> Configuration:
    """Return a Configuration with all defaults (no CLI parsing)."""
    config = Configuration()
    if config.high_ram:
        config.cache_classic = True
    if config.windows_standalone_build:
        config.auto_launch = True
    if config.disable_auto_launch:
        config.auto_launch = False
    if config.force_fp16:
        config.fp16_unet = True
    if config.enable_manager_legacy_ui:
        config.enable_manager = True
    return config

def cli_args_configuration() -> Configuration:
    """Return a Configuration with all defaults.

    In the Typer-based CLI, real parsing is in ``comfy.cmd.cli``. This
    function exists for backward compat and returns defaults.
    """
    return Configuration()


def dynamic_vram_requested() -> bool:
    """Return whether the current configuration requests dynamic VRAM."""
    configuration = _args()
    return (
        (configuration.enable_dynamic_vram or dynamic_vram_supported() or PerformanceFeature.DynamicVRAM in configuration.fast)
        and not configuration.highvram
        and not configuration.gpu_only
        and not configuration.disable_dynamic_vram
    )


def dynamic_vram_supported() -> bool:
    """Return whether the current runtime supports dynamic VRAM."""
    import torch

    torch_version = version.parse(torch.__version__.split("+", maxsplit=1)[0])
    # comfy-aimdo replaces PyTorch's allocator with a CUDA pluggable allocator.
    # ROCm intentionally exposes its HIP devices through ``torch.cuda``, so a
    # device-type check alone cannot distinguish the unsupported HIP runtime.
    return (
        torch_version >= version.parse("2.8")
        and getattr(torch.version, "hip", None) is None
    )


def enables_dynamic_vram():
    if _args().disable_dynamic_vram:
        return False
    if _args().enable_dynamic_vram:
        return True
    return dynamic_vram_requested() and dynamic_vram_supported()

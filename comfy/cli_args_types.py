from __future__ import annotations

import enum
import logging
import os
from typing import Optional, List, Callable, Any, Union, Mapping, NamedTuple

import configargparse
import configargparse as argparse

ConfigurationExtender = Callable[[argparse.ArgParser], Optional[argparse.ArgParser]]


class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "latent2rgb"
    TAESD = "taesd"

    @classmethod
    def from_string(cls, value: str):
        for member in cls:
            if member.value == value:
                return member
        return None


ConfigObserver = Callable[[str, Any], None]


def db_config() -> str:
    from .vendor.appdirs import user_data_dir

    logger = logging.getLogger(__name__)
    try:
        data_dir = user_data_dir(appname="comfyui")
        os.makedirs(data_dir, exist_ok=True)
        db_path = os.path.join(data_dir, "comfy.db")
        default_db_url = f"sqlite:///{db_path}"
    except Exception as e:
        # Fallback to an in-memory database if the user directory can't be accessed
        logger.warning(f"Could not determine user data directory for database, falling back to in-memory: {e}")
        default_db_url = "sqlite:///:memory:"
    return default_db_url


def is_valid_directory(path: str) -> str:
    """Validate if the given path is a directory, and check permissions."""
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(f"The path '{path}' does not exist.")
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"'{path}' is not a directory.")
    if not os.access(path, os.R_OK):
        raise argparse.ArgumentTypeError(f"You do not have read permissions for '{path}'.")
    return path


class PerformanceFeature(enum.Enum):
    Fp16Accumulation = "fp16_accumulation"
    Fp8MatrixMultiplication = "fp8_matrix_mult"
    CublasOps = "cublas_ops"
    AutoTune = "autotune"
    DynamicVRAM = "dynamic_vram"


COMFY_KITCHEN_BACKENDS = ("eager", "cuda", "triton")
FP8_MATERIALIZATION_MODES = ("auto", "torch", "comfy_kitchen")

VRAM_MODES = ("gpu_only", "highvram", "normalvram", "lowvram", "novram", "cpu")
PRECISION_MODES = ("force_fp32", "force_fp16", "force_bf16")
UNET_MODES = ("fp32_unet", "fp64_unet", "bf16_unet", "fp16_unet", "fp8_e4m3fn_unet", "fp8_e5m2_unet", "fp8_e8m0fnu_unet")
VAE_MODES = ("fp16_vae", "fp32_vae", "bf16_vae")
TEXT_ENC_MODES = ("fp8_e4m3fn_text_enc", "fp8_e5m2_text_enc", "fp16_text_enc", "fp32_text_enc", "bf16_text_enc")
ATTENTION_MODES = ("use_split_cross_attention", "use_quad_cross_attention", "use_pytorch_cross_attention", "use_sage_attention", "use_flash_attention")
CACHE_MODES = ("cache_classic", "cache_lru", "cache_none", "cache_ram")
UPCAST_MODES = ("force_upcast_attention", "dont_upcast_attention")


class Configuration(dict):
    """
    Configuration options parsed from command-line arguments or config files.

    Attributes:
        config_files (Optional[List[str]]): Path to the configuration file(s) that were set in the arguments.
        cwd (Optional[str]): Working directory. Defaults to the current directory. This is always treated as a base path for model files, and it will be the place where model files are downloaded.
        base_paths (Optional[list[str]]): Additional base paths for custom nodes, models and inputs.
        base_directory (Optional[str]): Set the ComfyUI base directory for models, custom_nodes, input, output, temp, and user directories.
        models_directory (Optional[str]): Set the ComfyUI models directory independently of the base directory.
        listen (str): IP address to listen on. Defaults to "127.0.0.1".
        port (int): Port number for the server to listen on. Defaults to 8188.
        tls_keyfile (Optional[str]): Path to TLS key file.
        tls_certfile (Optional[str]): Path to TLS certificate file.
        enable_cors_header (Optional[str]): Enables CORS with the specified origin.
        max_upload_size (float): Maximum upload size in MB. Defaults to 100.
        extra_model_paths_config (Optional[List[str]]): Extra model paths configuration files.
        add_model_folder_path (list[str]): Additional folder paths registered under a folder_paths kind. Each entry is a string of the form "KIND=PATH" (e.g. "checkpoints=/mnt/nas/sdxl"). Applied at startup before any node executes.
        output_directory (Optional[str]): Directory for output files. This can also be a relative path to the cwd or current working directory.
        temp_directory (Optional[str]): Temporary directory for processing.
        input_directory (Optional[str]): Directory for input files. When this is a relative path, it will be looked up relative to the cwd (current working directory) and all of the base_paths.
        auto_launch (bool): Auto-launch UI in the default browser. Defaults to False.
        disable_auto_launch (bool): Disable auto launching the browser.
        cuda_device (Optional[str]): CUDA device ID or comma-separated IDs. None means all visible devices.
        torch_device (Optional[str]): Torch device by name, e.g. "cuda:1", "cpu", "mps". Overrides cuda_device and cpu flags.
        rank (Optional[int]): Global process rank. Defaults to the canonical RANK environment variable.
        world_size (Optional[int]): Total process count. Defaults to the canonical WORLD_SIZE environment variable.
        local_rank (Optional[int]): Node-local process rank. Defaults to LOCAL_RANK.
        local_world_size (Optional[int]): Number of local processes. Defaults to LOCAL_WORLD_SIZE.
        master_addr (Optional[str]): Process-group rendezvous host. Defaults to MASTER_ADDR.
        master_port (Optional[int]): Process-group rendezvous port. Defaults to MASTER_PORT.
        pipeline_parallel_size (Optional[int]): Number of pipeline stages. Defaults to world size for external launchers and selected device count otherwise.
        tensor_parallel_size (Optional[int]): Number of tensor-parallel ranks. Defaults to COMFYUI_TENSOR_PARALLEL_SIZE, else the largest power of two of identical NVIDIA GPUs when guess_settings is on (Linux only), else one.
        ulysses_degree (Optional[int]): xDiT Ulysses sequence-parallel degree. Defaults to COMFYUI_ULYSSES_DEGREE or one.
        ring_degree (Optional[int]): xDiT ring-attention degree. Defaults to COMFYUI_RING_DEGREE or one.
        nccl_proto (str): NCCL protocol selection: auto, simple, ll, or ll128.
        distributed_executor_backend (str): Pipeline executor selection: auto, peer, mp, or external_launcher.
        model_management_device_scope (str): Internal process ownership scope for model memory. ``all`` permits shared-process multi-device management; ``local`` restricts a process peer to its selected device.
        cuda_malloc (bool): Enable cudaMallocAsync. Defaults to True in applicable setups.
        disable_cuda_malloc (bool): Disable cudaMallocAsync.
        dont_upcast_attention (bool): Disable upcasting of attention.
        force_upcast_attention (bool): Force upcasting of attention.
        force_fp32 (bool): Force using FP32 precision.
        force_fp16 (bool): Force using FP16 precision.
        force_bf16 (bool): Force using BF16 precision.
        bf16_unet (bool): Use BF16 precision for UNet.
        fp16_unet (bool): Use FP16 precision for UNet.
        fp32_unet (bool): Run the diffusion model in fp32.
        fp64_unet (bool): Run the diffusion model in fp64.
        fp8_e4m3fn_unet (bool): Store unet weights in fp8_e4m3fn
        fp8_e5m2_unet (bool): Store unet weights in fp8_e5m2.
        fp8_storage (bool): Preserve native fp8 checkpoint weights as resident fp8 storage when supported.
        fp16_vae (bool): Run the VAE in FP16 precision.
        fp32_vae (bool): Run the VAE in FP32 precision.
        bf16_vae (bool): Run the VAE in BF16 precision.
        cpu_vae (bool): Run the VAE on the CPU.
        fp8_e4m3fn_text_enc (bool): Use FP8 precision for the text encoder (e4m3fn variant).
        fp8_e5m2_text_enc (bool): Use FP8 precision for the text encoder (e5m2 variant).
        fp16_text_enc (bool): Use FP16 precision for the text encoder.
        fp32_text_enc (bool): Use FP32 precision for the text encoder.
        fp16_intermediates (bool): Experimental: Use fp16 for intermediate tensors between nodes instead of fp32.
        oneapi_device_selector (Optional[str]): Sets the oneAPI device(s) this instance will use.
        directml (Optional[int]): Use DirectML. -1 for auto-selection.
        preview_method (LatentPreviewMethod): Method for generating previews. Defaults to "auto".
        cache_lru (int): Use LRU caching with a maximum of N node results cached. May use more RAM/VRAM.
        cache_ram (float): Use RAM pressure caching with the specified headroom threshold.
        high_ram (bool): Prefer pinning/offloading behavior for systems with abundant RAM.
        use_split_cross_attention (bool): Use split cross-attention optimization.
        use_quad_cross_attention (bool): Use sub-quadratic cross-attention optimization.
        use_pytorch_cross_attention (bool): Use PyTorch's cross-attention function.
        use_sage_attention (bool): Use Sage Attention
        use_flash_attention (bool): Use FlashAttention
        use_ck_attention (bool): Use Comfy Kitchen attention.
        disable_xformers (bool): Disable xformers.
        gpu_only (bool): Run everything on the GPU.
        highvram (bool): Keep models in GPU memory.
        normalvram (bool): Default VRAM usage setting.
        lowvram (bool): Reduce UNet's VRAM usage.
        novram (bool): Minimize VRAM usage.
        cpu (bool): Use CPU for processing.
        fast (set[PerformanceFeature]): Enable some untested and potentially quality deteriorating optimizations. Pass specific optimizations if you only want to enable some (e.g. --fast fp16_accumulation fp8_matrix_mult or --fast fp16_accumulation,fp8_matrix_mult). Valid optimizations: fp16_accumulation, fp8_matrix_mult, cublas_ops, autotune, dynamic_vram
        reserve_vram (Optional[float]): Set the amount of vram in GB you want to reserve for use by your OS/other software. By default some amount is reserved depending on your OS
        vram_headroom (float): Extra DynamicVRAM headroom in GB above the default reservation.
        disable_nvml_pressure (bool): Use CUDA instead of NVML for DynamicVRAM memory pressure.
        disable_dynamic_vram (bool): Disable dynamic VRAM and use estimate-based model loading.
        disable_cuda_graphs (bool): Disable CUDA graph capture and replay.
        disable_comfy_compiler (bool): Disable the Comfy model compiler, including its CUDA graph subfeature.
        assert_graph_breaks (bool): Fail on Comfy model compiler graph breaks.
        fast_disk (bool): Force disk-backed dynamic loading and offload over unpinned RAM.
        disable_fast_disk (bool): Disable disk-backed loading, overriding fast_disk.
        disable_smart_memory (bool): Disable smart memory management.
        deterministic (bool): Use deterministic algorithms where possible.
        quick_test_for_ci (bool): Enable quick testing mode for CI.
        windows_standalone_build (bool): Enable features for standalone Windows build.
        disable_metadata (bool): Disable saving metadata with outputs.
        disable_all_custom_nodes (bool): Disable loading all custom nodes.
        multi_user (bool): Enable multi-user mode.
        plausible_analytics_base_url (Optional[str]): Base URL for server-side analytics.
        plausible_analytics_domain (Optional[str]): Domain for analytics events.
        analytics_use_identity_provider (bool): Use platform identifiers for analytics.
        write_out_config_file (bool): Enable writing out the configuration file.
        create_directories (bool): Creates the default models/, input/, output/ and temp/ directories, then exits.
        distributed_queue_connection_uri (Optional[str]): Servers and clients will connect to this AMQP URL to form a distributed queue and exchange prompt execution requests and progress updates.
        distributed_queue_frontend (bool): Frontends will start the web UI and connect to the provided AMQP URL to submit prompts.
        distributed_queue_worker (bool): Workers will pull requests off the AMQP URL.
        distributed_queue_name (str): This name will be used by the frontends and workers to exchange prompt requests and replies. Progress updates will be prefixed by the queue name, followed by a '.', then the user ID.
        external_address (str): Specifies a base URL for external addresses reported by the API, such as for image paths.
        logging_level (str): Specifies a log level
        debug_hang (bool): Enable stack trace dumps on Ctrl-C for debugging hangs.
        disable_known_models (bool): Disables automatic downloads of known models and prevents them from appearing in the UI.
        max_queue_size (int): The API will reject prompt requests if the queue's size exceeds this value.
        otel_service_name (str): The name of the service or application that is generating telemetry data. Default: "comfyui".
        otel_service_version (str): The version of the service or application that is generating telemetry data. Default: "0.0.1".
        otel_exporter_otlp_endpoint (Optional[str]): A base endpoint URL for any signal type, with an optionally-specified port number. Helpful for when you're sending more than one signal to the same endpoint and want one environment variable to control the endpoint.
        force_channels_last (bool): Force channels last format when inferencing the models.
        force_hf_local_dir_mode (bool): Download repos from huggingface.co to the models/huggingface directory with the "local_dir" argument instead of models/huggingface_cache with the "cache_dir" argument, recreating the traditional file structure.
        executor_factory (str): Either ThreadPoolExecutor or ProcessPoolExecutor, defaulting to ThreadPoolExecutor
        preview_size (int): Sets the maximum preview size for sampler nodes. Defaults to 512.
        openai_api_key (str): Configures the OpenAI API Key for the OpenAI nodes. Visit https://platform.openai.com/api-keys to create this key.
        ideogram_api_key (str): Configures the Ideogram API Key for the Ideogram nodes. Visit https://ideogram.ai/manage-api to create this key.
        anthropic_api_key (str): Configures the Anthropic API key for its nodes related to Claude functionality. Visit https://console.anthropic.com/settings/keys to create this key.
        user_directory (Optional[str]): Set the ComfyUI user directory with an absolute path.
        log_stdout (bool): Send normal process output to stdout instead of stderr (default)
        panic_when (list[str]): List of fully qualified exception class names to panic (sys.exit(1)) when a workflow raises it.
        enable_manager (bool): Enable the ComfyUI-Manager feature.
        disable_manager_ui (bool): Disables only the ComfyUI-Manager UI.
        enable_manager_legacy_ui (bool): Enables the legacy UI of ComfyUI-Manager.
        enable_compress_response_body (bool): Enable compressing response body.
        workflows (list[str]): Execute the API workflow(s) and exit. Each value can be a file path, a literal JSON string starting with '{', a URI (https://, s3://, hf://, etc.), or '-' for stdin. Outputs are printed to stdout; logs go to stderr.
        prompt (Optional[str]): Override the positive prompt text in workflows executed via --workflows.
        negative_prompt (Optional[str]): Override the negative prompt text in workflows executed via --workflows.
        steps (Optional[int]): Override the number of sampling steps in workflows run via --workflows.
        seed (Optional[int]): Override the seed in sampler and noise nodes in workflows run via --workflows.
        quantity (int): Number of workflow jobs to run/submit/convert. Defaults to 1.
        image (Optional[list[str]]): Override image inputs in workflows run via --workflows. Accepts file paths or URIs (space-separated or comma-separated).
        video (Optional[list[str]]): Override video inputs in workflows run via --workflows. Accepts file paths or URIs (space-separated or comma-separated).
        audio (Optional[list[str]]): Override audio inputs in workflows run via --workflows. Accepts file paths or URIs (space-separated or comma-separated).
        output (Optional[str]): Override the output directory for workflows run via --workflows.
        guess_settings (bool): Auto-detect best settings for this machine (GPU type, RAM, attention backend, etc.). Explicit flags override guessed values.
        disable_pinned_memory (bool): Disable pinned memory use.
        pinned_memory_reserve (Optional[float]): GB of the effective memory limit (host RAM, or the cgroup limit in a container) that pinned memory always leaves free, checked against memory available at each pin. Defaults to the larger of 4 GB and 10% of the limit.
        max_pinned_memory (Optional[float]): Cap in GB on pinned host memory. Defaults to, and never exceeds, the effective memory limit minus pinned_memory_reserve.
        fp8_e8m0fnu_unet (bool): Store unet weights in fp8_e8m0fnu.
        fp8_storage (bool): Preserve native fp8 checkpoint weights as resident fp8 storage when supported.
        bf16_text_enc (bool): Store text encoder weights in bf16.
        supports_fp8_compute (bool): ComfyUI will act like if the device supports fp8 compute.
        fp8_materialization (str): Select how FP8 weights are materialized to bf16/fp16/fp32. auto leaves room for benchmark-selected lowerings; torch uses the graph-visible torch op; comfy_kitchen uses comfy_kitchen's registered backend.
        cache_classic (bool): WARNING: Unused. Use the old style (aggressive) caching.
        cache_none (bool): Reduced RAM/VRAM usage at the expense of executing every node for each run.
        async_offload (Optional[int]): Use async weight offloading. An optional argument controls the amount of offload streams.
        disable_async_offload (bool): Disable async weight offloading.
        force_non_blocking (bool): Force ComfyUI to use non-blocking operations for all applicable tensors. This may improve performance on some non-Nvidia systems but can cause issues with some workflows.
        default_hashing_function (str): Allows you to choose the hash function to use for duplicate filename / contents comparison. Default is sha256.
        mmap_torch_files (bool): Use mmap when loading ckpt/pt files.
        disable_mmap (bool): Don't use mmap when loading safetensors.
        dont_print_server (bool): Don't print server output.
        disable_partner_nodes (bool): Disable loading partner nodes (the paid API nodes). The frontend keeps its internet access.
        offline (bool): Prevent the frontend from communicating with the internet by setting a restrictive Content-Security-Policy header. Also disables partner nodes.
        disable_api_nodes (bool): Deprecated alias for offline.
        front_end_version (str): Specifies the version of the frontend to be used.
        front_end_root (Optional[str]): The local filesystem path to the directory where the frontend is located. Overrides --front-end-version.
        comfy_api_base (str): Set the base URL for the ComfyUI API. (default: https://api.comfy.org)
        database_url (Optional[str]): Specify the database URL, e.g. 'sqlite:///:memory:'. By default the database is stored in the effective user directory.
        enable_assets (bool): Enable the assets API and database-backed asset management features.
        enable_asset_hashing (bool): Compute blake3 hashes while scanning assets.
        disable_assets_autoscan (bool): Disable asset scanning on startup for database synchronization.
        feature_flag (list[str]): Server feature flags in KEY[=VALUE] form.
        list_feature_flags (bool): Print known CLI-settable feature flags as JSON and exit.
        blacklist_custom_nodes (list[str]): Specify custom node folders to never load. Accepts shell-style globs.
        whitelist_custom_nodes (list[str]): Specify custom node folders to load even when --disable-all-custom-nodes is enabled.
        default_device (Optional[int]): Set the id of the default device, all other devices will stay visible.
        block_runtime_package_installation (Optional[bool]): When set, custom nodes like ComfyUI Manager, Easy Use, Nunchaku and others will not be able to use pip or uv to install packages at runtime (experimental).
        enable_eval (Optional[bool]): Enable nodes that can evaluate Python code in workflows.
        pip_facade_registry_base_url (str): Base URL for the Comfy registry API used by the `serve-pip` facade when no snapshot URI is provided.
        pip_facade_cache_prefix (Optional[str]): Writable fsspec URI prefix where `serve-pip` caches generated wheels.
        pip_facade_cache_s3_endpoint_url (Optional[str]): Custom S3-compatible endpoint URL for an s3:// pip-facade cache prefix.
        pip_facade_only_known_nodes (bool): Only expose nodes present in this repository's local compatibility registry.
        pip_facade_snapshot_uri (Optional[str]): Read a pip-facade registry snapshot from this fsspec URI instead of querying the live registry API.
        pip_facade_snapshot_max_age_seconds (int): Maximum snapshot age before metadata and readiness return 503. Zero disables age enforcement.
        pip_facade_snapshot_output (Optional[str]): Output path for `snapshot-pip-registry`. Use a `.xz` suffix or `--pip-facade-snapshot-compression=xz` for a compressed archive.
        pip_facade_snapshot_compression (str): Snapshot compression mode for `snapshot-pip-registry`. One of `auto`, `none`, or `xz`.
        pip_facade_snapshot_overwrite (bool): Overwrite an existing snapshot output file.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._observers: List[ConfigObserver] = []
        self.config_files = []
        self.cwd: Optional[str] = None
        self.base_paths: list[str] = []
        self.base_directory: Optional[str] = None
        self.models_directory: Optional[str] = None
        self.listen: str = "127.0.0.1"
        self.port: int = 8188
        self.tls_keyfile: Optional[str] = None
        self.tls_certfile: Optional[str] = None
        self.enable_cors_header: Optional[str] = None
        self.enable_compress_response_body: bool = False
        self.max_upload_size: float = 100.0
        self.extra_model_paths_config: Optional[List[str]] = []
        self.output_directory: Optional[str] = None
        self.temp_directory: Optional[str] = None
        self.input_directory: Optional[str] = None
        self.auto_launch: bool = False
        self.disable_auto_launch: bool = False
        self.cuda_device: Optional[str] = None
        self.torch_device: Optional[str] = None
        self.rank: Optional[int] = None
        self.world_size: Optional[int] = None
        self.local_rank: Optional[int] = None
        self.local_world_size: Optional[int] = None
        self.master_addr: Optional[str] = None
        self.master_port: Optional[int] = None
        self.pipeline_parallel_size: Optional[int] = None
        self.tensor_parallel_size: Optional[int] = None
        self.ulysses_degree: Optional[int] = None
        self.ring_degree: Optional[int] = None
        self.nccl_proto: str = "auto"
        self.distributed_executor_backend: str = "auto"
        self.model_management_device_scope: str = "all"
        self.cuda_malloc: bool = True
        self.disable_cuda_malloc: bool = True
        self.dont_upcast_attention: bool = False
        self.force_upcast_attention: bool = False
        self.force_fp32: bool = False
        self.force_fp16: bool = False
        self.force_bf16: bool = False
        self.bf16_unet: bool = False
        self.fp16_unet: bool = False
        self.fp32_unet: bool = False
        self.fp64_unet: bool = False
        self.fp8_e4m3fn_unet: bool = False
        self.fp8_e5m2_unet: bool = False
        self.fp8_storage: bool = True
        self.fp16_vae: bool = False
        self.fp32_vae: bool = False
        self.bf16_vae: bool = False
        self.cpu_vae: bool = False
        self.fp8_e4m3fn_text_enc: bool = False
        self.fp8_e5m2_text_enc: bool = False
        self.fp16_text_enc: bool = False
        self.fp32_text_enc: bool = False
        self.fp16_intermediates: bool = False
        self.directml: Optional[int] = None
        self.preview_method: LatentPreviewMethod = LatentPreviewMethod.Auto
        self.use_split_cross_attention: bool = False
        self.use_quad_cross_attention: bool = False
        self.use_pytorch_cross_attention: bool = False
        self.use_sage_attention: bool = False
        self.use_flash_attention: bool = False
        self.use_ck_attention: bool = False
        self.disable_xformers: bool = False
        self.gpu_only: bool = False
        self.highvram: bool = False
        self.normalvram: bool = False
        self.lowvram: bool = False
        self.novram: bool = False
        self.cpu: bool = False
        self.fast: set[PerformanceFeature] = set()
        self.enable_comfy_kitchen_backends: list[str] = []
        self.disable_comfy_kitchen_backends: list[str] = []
        self.enable_triton_backend: bool = False
        self.disable_triton_backend: bool = False
        self.fp8_materialization: str = "auto"
        # reserve 0, because this has been exceptionally buggy
        self.reserve_vram: Optional[float] = None
        self.vram_headroom: float = 0.0
        self.disable_nvml_pressure: bool = False
        self.disable_dynamic_vram: bool = False
        self.enable_dynamic_vram: bool = False
        self.fast_disk: bool = False
        self.disable_fast_disk: bool = False
        self.disable_cuda_graphs: bool = False
        self.disable_comfy_compiler: bool = False
        self.assert_graph_breaks: bool = False
        self.disable_smart_memory: bool = False
        self.deterministic: bool = False
        self.dont_print_server: bool = False
        self.quick_test_for_ci: bool = False
        self.windows_standalone_build: bool = False
        self.disable_metadata: bool = False
        self.disable_all_custom_nodes: bool = False
        self.blacklist_custom_nodes: list[str] = []
        self.whitelist_custom_nodes: list[str] = []
        self.multi_user: bool = False
        self.plausible_analytics_base_url: Optional[str] = None
        self.plausible_analytics_domain: Optional[str] = None
        self.analytics_use_identity_provider: bool = False
        self.write_out_config_file: bool = False
        self.create_directories: bool = False
        self.distributed_queue_connection_uri: Optional[str] = None
        self.distributed_queue_worker: bool = False
        self.distributed_queue_frontend: bool = False
        self.distributed_queue_name: str = "comfyui"
        self.external_address: Optional[str] = None
        self.disable_known_models: bool = False
        self.max_queue_size: int = 65536
        self.disable_requests_caching: bool = False
        self.force_channels_last: bool = False
        self.force_hf_local_dir_mode: bool = False
        self.preview_size: int = 512
        self.logging_level: str = "INFO"
        self.debug_hang: bool = False
        self.oneapi_device_selector: Optional[str] = None
        self.log_stdout: bool = False

        # from guill
        self.cache_lru: int = 0

        # from opentracing docs
        self.otel_service_name: str = "comfyui"
        self.otel_service_version: str = "0.0.1"
        self.otel_exporter_otlp_endpoint: Optional[str] = None
        self.executor_factory: str = "ThreadPoolExecutor"
        self.openai_api_key: Optional[str] = None
        self.ideogram_api_key: Optional[str] = None
        self.anthropic_api_key: Optional[str] = None
        self.google_api_key: Optional[str] = None
        self.user_directory: Optional[str] = None
        self.panic_when: list[str] = []
        self.workflows: list[str] = []
        self.prompt: Optional[str] = None
        self.negative_prompt: Optional[str] = None
        self.steps: Optional[int] = None
        self.seed: Optional[int] = None
        self.quantity: int = 1
        self.image: Optional[list[str]] = None
        self.video: Optional[list[str]] = None
        self.audio: Optional[list[str]] = None
        self.output: Optional[str] = None
        self.guess_settings: bool = True
        self.enable_manager: bool = False
        self.disable_manager_ui: bool = False
        self.enable_manager_legacy_ui: bool = False
        self.disable_pinned_memory: bool = False
        self.pinned_memory_reserve: Optional[float] = None
        self.max_pinned_memory: Optional[float] = None

        self.fp8_e8m0fnu_unet: bool = False
        self.bf16_text_enc: bool = False
        self.supports_fp8_compute: bool = False
        self.cache_classic: bool = False
        self.cache_none: bool = False
        self.cache_ram: float = 0.0
        self.high_ram: bool = False
        self.async_offload: Optional[int] = None
        self.disable_async_offload: bool = False
        self.force_non_blocking: bool = False
        self.default_hashing_function: str = 'sha256'
        self.mmap_torch_files: bool = False
        self.disable_mmap: bool = False
        self.disable_partner_nodes: bool = False
        self.offline: bool = False
        self.disable_api_nodes: bool = False
        self.front_end_version: str = "comfyanonymous/ComfyUI@latest"
        self.front_end_root: Optional[str] = None
        self.comfy_api_base: str = "https://api.comfy.org"
        self.database_url: Optional[str] = None
        self._database_url_explicit: bool = False
        self.enable_assets: bool = False
        self.enable_asset_hashing: bool = False
        self.disable_assets_autoscan: bool = False
        self.feature_flag: list[str] = []
        self.list_feature_flags: bool = False
        self.default_device: Optional[int] = None
        self.block_runtime_package_installation: bool = True
        self.enable_eval: Optional[bool] = False
        self.disable_progress: bool = False
        self.enable_video_to_image_fallback: bool = False
        self.disable_manager_model_fallback: bool = False
        self.refresh_manager_models: bool = False
        self.disable_civitai_model_fallback: bool = False
        self.pip_facade_registry_base_url: str = "https://api.comfy.org"
        self.pip_facade_cache_prefix: Optional[str] = None
        self.pip_facade_cache_s3_endpoint_url: Optional[str] = None
        self.pip_facade_cache_revision: Optional[int] = None
        self.pip_facade_only_known_nodes: bool = False
        self.pip_facade_snapshot_uri: Optional[str] = None
        self.pip_facade_snapshot_max_age_seconds: int = 0
        self.pip_facade_snapshot_output: Optional[str] = None
        self.pip_facade_snapshot_compression: str = "auto"
        self.pip_facade_snapshot_overwrite: bool = False

        self.cfg: Optional[float] = None
        self.sampler: Optional[str] = None
        self.scheduler: Optional[str] = None
        self.denoise: Optional[float] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.batch_size: Optional[int] = None
        self.checkpoint: Optional[str] = None
        self.diffusion_model: Optional[str] = None
        self.set: list[str] = []
        self.add_lora: Optional[list[str]] = None
        self.add_model_folder_path: list[str] = []
        self.compile: bool = False

        for key, value in kwargs.items():
            self[key] = value
        # this must always be last

    def __getattr__(self, item):
        if item not in self:
            return None
        return self[item]

    def __setattr__(self, key, value):
        if key != "_observers":
            old_value = self.get(key)
            self[key] = value
            if old_value != value:
                self._notify_observers(key, value)
        else:
            super().__setattr__(key, value)

    def update(self, __m: Union[Mapping[str, Any], None] = None, **kwargs):
        if __m is None:
            __m = {}
        changes = {}
        for k, v in dict(__m, **kwargs).items():
            if k not in self or self[k] != v:
                changes[k] = v
        super().update(__m, **kwargs)
        for k, v in changes.items():
            self._notify_observers(k, v)
        # make this more pythonic
        return self

    def register_observer(self, observer: ConfigObserver):
        self._observers.append(observer)

    def unregister_observer(self, observer: ConfigObserver):
        self._observers.remove(observer)

    def _notify_observers(self, key, value):
        for observer in self._observers:
            observer(key, value)

    def __getstate__(self):
        state = self.copy()
        if "_observers" in state:
            state.pop("_observers")
        return state

    def __setstate__(self, state):
        self._observers = []
        self.update(state)

    @property
    def verbose(self) -> str:
        return self.logging_level

    @verbose.setter
    def verbose(self, value):
        if isinstance(value, bool):
            self.logging_level = "DEBUG"
        else:
            self.logging_level = value


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums in a case-insensitive manner.
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        self._enum = enum_type

        # Generate choices from the Enum for the help message
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        # We handle choices ourselves for case-insensitivity, so remove it before calling super.
        if "choices" in kwargs:
            del kwargs["choices"]

        super(EnumAction, self).__init__(**kwargs)
        self._choices = choices

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum, case-insensitively
        value_lower = values.lower()
        for member in self._enum:
            if member.value.lower() == value_lower:
                setattr(namespace, self.dest, member)
                return

        # If no match found, raise an error
        msg = f"invalid choice: {values!r} (choose from {', '.join(self._choices)})"
        raise argparse.ArgumentError(self, msg)


class ParsedArgs(NamedTuple):
    namespace: configargparse.Namespace
    unknown_args: list[str]
    config_file_paths: list[str]


class EnhancedConfigArgParser(configargparse.ArgParser):
    def parse_known_args_with_config_files(self, args=None, namespace=None, **kwargs) -> ParsedArgs:
        # usually the single method open
        prev_open_func = self._config_file_open_func
        config_files: List[str] = []

        try:
            self._config_file_open_func = lambda path: config_files.append(path)
            self._open_config_files(args)
        finally:
            self._config_file_open_func = prev_open_func

        namespace, unknown_args = super().parse_known_args(args, namespace, **kwargs)
        return ParsedArgs(namespace, unknown_args, config_files)


class FlattenAndAppendAction(argparse.Action):
    """
    Custom action to handle comma-separated values and multiple invocations
    of the same argument, flattening them into a single list.

    Type conversion (if specified via ``type=``) is applied *after* comma
    splitting so that ``--fast fp16_accumulation,fp8_matrix_mult`` works
    correctly with enum or other non-string types.
    """

    def __init__(self, **kwargs):
        self._item_type = kwargs.pop("type", None)
        super().__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None)
        if items is None:
            items = []
        else:
            items = list(items)

        if values is None:
            values = []
        elif isinstance(values, str):
            values = [values]

        for value in values:
            for item in (v.strip() for v in str(value).split(',')):
                if not item:
                    continue
                if self._item_type is not None:
                    try:
                        items.append(self._item_type(item))
                    except (ValueError, KeyError) as exc:
                        raise argparse.ArgumentError(self, str(exc)) from exc
                else:
                    items.append(item)

        setattr(namespace, self.dest, items)

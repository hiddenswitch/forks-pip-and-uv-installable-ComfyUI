from __future__ import annotations

from ..cli_args_types import Configuration
from ..cli_args import default_configuration

# Fields that affect folder paths - when changed, folder_names_and_paths needs reinitialization
AFFECTS_PATHS: frozenset[str] = frozenset({
    'cwd',
    'base_directory',
    'base_paths',
    'output_directory',
    'input_directory',
    'previews_directory',
    'temp_directory',
    'user_directory',
    'extra_model_paths_config',
})

# Fields that affect model management behavior - when changed, requires ProcessPoolExecutor
MODEL_MANAGEMENT_ARGS: frozenset[str] = frozenset({
    "deterministic",
    "directml",
    "cpu",
    "torch_device",
    "pipeline_parallel_size",
    "tensor_parallel_size",
    "hunyuan_expert_parallel_only",
    "ulysses_degree",
    "ring_degree",
    "nccl_proto",
    "distributed_executor_backend",
    "model_management_device_scope",
    "disable_xformers",
    # todo: this is the default, so it will be omitted
    # "use_pytorch_cross_attention",
    "use_split_cross_attention",
    "use_quad_cross_attention",
    "use_pytorch_cross_attention",
    "supports_fp8_compute",
    "fast",
    "lowvram",
    "novram",
    "highvram",
    "gpu_only",
    "disable_dynamic_vram",
    "fast_disk",
    "force_fp32",
    "force_fp16",
    "force_bf16",
    "reserve_vram",
    "vram_headroom",
    "disable_nvml_pressure",
    "high_ram",
    "disable_smart_memory",
    "disable_pinned_memory",
    "pinned_memory_reserve",
    "max_pinned_memory",
    "async_offload",
    "disable_async_offload",
    "force_non_blocking",
    "force_channels_last",
    "fp32_unet",
    "fp64_unet",
    "bf16_unet",
    "fp16_unet",
    "fp8_e4m3fn_unet",
    "fp8_e5m2_unet",
    "fp8_e8m0fnu_unet",
    "fp8_storage",
    "fp8_e4m3fn_text_enc",
    "fp8_e5m2_text_enc",
    "fp16_text_enc",
    "bf16_text_enc",
    "fp32_text_enc",
    "cpu_vae",
    "fp16_vae",
    "bf16_vae",
    "fp16_intermediates",
    "fp32_vae",
    "force_upcast_attention",
    "use_sage_attention",
    "use_flash_attention",
    "use_ck_attention",
    "disable_cuda_graphs",
    "disable_comfy_compiler",
    "assert_graph_breaks",
})


def requires_process_pool_executor(configuration: Configuration | None) -> bool:
    if configuration is None:
        return False

    from ..execution_context import current_execution_context

    baseline = current_execution_context().configuration
    default = baseline or default_configuration()
    for key in MODEL_MANAGEMENT_ARGS:
        # Check if key is in configuration and differs from default
        if key in configuration:
            val = configuration[key]
            # Use equality check, handling potential missing keys in default (though default should have them)
            if key not in default or val != default[key]:
                return True
    return False


def model_management_fingerprint(configuration: Configuration | dict | None) -> tuple:
    """Return a hashable fingerprint of the ``MODEL_MANAGEMENT_ARGS`` subset of *configuration*.

    Two configurations with the same fingerprint produce identical model
    management state and can safely share a subprocess worker. When the
    fingerprints differ, a fresh ``ProcessPoolExecutor`` is required
    because model_management latches VRAM/precision/attention settings at
    import time, which happens once per subprocess.

    ``None`` is treated as "all defaults" — i.e. ``default_configuration()``
    — so ``Comfy()`` with no config and ``Comfy(configuration=Configuration())``
    share the same fingerprint.
    """
    if configuration is None or (isinstance(configuration, dict) and not configuration):
        configuration = default_configuration()
    # Sort the keys so set-like iteration order doesn't break equality
    # (some values can be sets — see ``fast``).
    entries: list[tuple[str, object]] = []
    for key in sorted(MODEL_MANAGEMENT_ARGS):
        val = configuration.get(key) if hasattr(configuration, "get") else None
        # Sets aren't hashable inside tuples; normalize to a sorted tuple.
        if isinstance(val, (set, frozenset)):
            val = tuple(sorted(str(x) for x in val))
        elif isinstance(val, list):
            val = tuple(val)
        entries.append((key, val))
    return tuple(entries)

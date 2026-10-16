"""
Typer CLI application for ComfyUI.

Single entry point for all commands: serve, worker, stop, logs,
and sub-apps: models, workflows, nodes, jobs, env.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import io
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

# must be set before any transitive import of requests (e.g. via typer/click)
warnings.filterwarnings("ignore", message=".*doesn't match a supported version")

import click
import typer

from ..cli_args_types import (
    Configuration, LatentPreviewMethod, PerformanceFeature,
    VRAM_MODES, PRECISION_MODES, UNET_MODES, VAE_MODES, TEXT_ENC_MODES,
    ATTENTION_MODES, UPCAST_MODES, FP8_MATERIALIZATION_MODES,
)
from ..distributed.config import DISTRIBUTED_EXECUTOR_BACKENDS

logger = logging.getLogger(__name__)


class _ComfyGroup(typer.core.TyperGroup):
    """Custom group that renders compact help for ``-h`` and full help for ``--help``."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if self._maybe_render_workflow_help(ctx, args):
            ctx.exit()
        return super().parse_args(ctx, args)

    def _maybe_render_workflow_help(self, ctx: click.Context, args: list[str]) -> bool:
        if "--help" not in args and "-h" not in args:
            return False

        command, command_ctx, workflow_args = self._workflow_help_command_context(ctx, args)
        if command is None or command_ctx is None or workflow_args is None:
            return False

        ref = command._extract_workflow_ref(workflow_args, ctx=command_ctx)
        if not ref:
            return False

        try:
            params = _discover_from_ref(ref)
            raw_workflow = _DISCOVER_RAW_CACHE.get(ref)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"(Could not discover workflow parameters: {exc})", err=True)
            typer.echo()
            typer.echo(command.get_help(command_ctx))
            return True

        command._render_workflow_help(command_ctx, ref, params, raw_workflow)
        return True

    def _workflow_help_command_context(
        self,
        ctx: click.Context,
        args: list[str],
    ) -> tuple["_RunWorkflowCommand | None", click.Context | None, list[str] | None]:
        command = self.get_command(ctx, "run-workflow")
        if args and args[0] == "run-workflow" and isinstance(command, _RunWorkflowCommand):
            command_ctx = click.Context(command, info_name="run-workflow", parent=ctx)
            return command, command_ctx, args[1:]

        if len(args) >= 2 and args[0] == "workflows":
            workflows = self.get_command(ctx, "workflows")
            if not isinstance(workflows, click.Group):
                return None, None, None
            command = workflows.get_command(ctx, args[1])
            if not isinstance(command, _RunWorkflowCommand):
                return None, None, None
            workflows_ctx = click.Context(workflows, info_name="workflows", parent=ctx)
            command_ctx = click.Context(command, info_name=args[1], parent=workflows_ctx)
            return command, command_ctx, args[2:]

        return None, None, None

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        if _short_help_requested():
            self._format_short_help(ctx, formatter)
        else:
            super().format_help(ctx, formatter)

    def _format_short_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        from .. import __version__
        formatter.write(f"comfyui {__version__}\n\n")
        formatter.write("usage: comfyui <command> [options]\n\n")

        commands = []
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            commands.append((name, cmd.get_short_help_str(limit=60)))

        if commands:
            max_name = max(len(n) for n, _ in commands)
            for name, short_help in commands:
                formatter.write(f"  {name:<{max_name}}  {short_help}\n")

        formatter.write(f"\nRun 'comfyui <command> --help' for detailed usage.\n")


def _short_help_requested() -> bool:
    """True when the user passed ``-h`` (not ``--help``)."""
    return "-h" in sys.argv and "--help" not in sys.argv


app = typer.Typer(
    name="comfyui",
    no_args_is_help=False,
    add_completion=False,
    cls=_ComfyGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)

_COMFYUI_ENV = {"auto_envvar_prefix": "COMFYUI"}


DEFAULT_VERSION_STRING = "comfyanonymous/ComfyUI@latest"



_DIRECTORY_OPTS: list[tuple] = [
    ("cwd", Optional[str], typer.Option(None, "-w", "--cwd", help="Specify the working directory. If not set, this is the current working directory. models/, input/, output/ and other directories will be located here by default.")),
    ("base_paths", Optional[list[str]], typer.Option(None, "--base-paths", help="Additional base paths for custom nodes, models and inputs.")),
    ("base_directory", Optional[str], typer.Option(None, "--base-directory", help="Set the ComfyUI base directory for models, custom_nodes, input, output, temp, and user directories.")),
    ("models_directory", Optional[str], typer.Option(None, "--models-directory", help="Set the ComfyUI models directory independently of the base directory.")),
    ("extra_model_paths_config", Optional[list[str]], typer.Option(None, "--extra-model-paths-config", help="Load one or more extra_model_paths.yaml files.")),
    ("output_directory", Optional[str], typer.Option(None, "--output-directory", help="Set the ComfyUI output directory.")),
    ("temp_directory", Optional[str], typer.Option(None, "--temp-directory", help="Set the ComfyUI temp directory.")),
    ("input_directory", Optional[str], typer.Option(None, "--input-directory", help="Set the ComfyUI input directory.")),
    ("user_directory", Optional[str], typer.Option(None, "--user-directory", help="Set the ComfyUI user directory with an absolute path.")),
    ("add_model_folder_path", Optional[list[str]], typer.Option(None, "--add-model-folder-path", envvar="COMFYUI_ADD_MODEL_FOLDER_PATH", help="Register an additional directory under a folder_paths kind. Format: KIND=PATH (e.g. checkpoints=/mnt/nas/sdxl). Repeatable. Comma-separated entries also accepted.")),
]

_DEVICE_OPTS: list[tuple] = [
    ("cuda_device", Optional[str], typer.Option(None, "--cuda-device", help="Select one CUDA device, an ordered comma-separated device list (for example 0,1), or all to preserve every currently visible device.")),
    ("torch_device", Optional[str], typer.Option(None, "--torch-device", help="Set the torch device by name, e.g. cuda:1, cpu, mps. Overrides --cuda-device and --cpu.")),
    ("default_device", Optional[int], typer.Option(None, "--default-device", help="Set the id of the default device, all other devices will stay visible.")),
    ("cuda_malloc", bool, typer.Option(False, "--cuda-malloc/--no-cuda-malloc", help="Enable cudaMallocAsync.")),
    ("disable_cuda_malloc", bool, typer.Option(True, "--disable-cuda-malloc", help="Disable cudaMallocAsync.")),
    ("directml", Optional[int], typer.Option(None, "--directml", help="Use torch-directml. -1 for auto-selection.")),
    ("oneapi_device_selector", Optional[str], typer.Option(None, "--oneapi-device-selector", help="Sets the oneAPI device(s) this instance will use.")),
    ("supports_fp8_compute", bool, typer.Option(False, "--supports-fp8-compute/--no-supports-fp8-compute", help="ComfyUI will act like if the device supports fp8 compute.")),
    ("enable_comfy_kitchen_backends", Optional[list[str]], typer.Option(None, "--enable-comfy-kitchen-backends", help="Re-enable comfy_kitchen quantization backends previously disabled by guess-settings or another flag. Comma-separated. Valid: eager, cuda, triton.")),
    ("disable_comfy_kitchen_backends", Optional[list[str]], typer.Option(None, "--disable-comfy-kitchen-backends", help="Disable comfy_kitchen quantization backends. Use this to skip an op backend that crashes on your hardware (e.g. triton fp8e4nv on Ampere). Comma-separated. Valid: eager, cuda, triton.")),
    ("enable_triton_backend", bool, typer.Option(False, "--enable-triton-backend/--no-enable-triton-backend", help="Enable the Triton backend in comfy-kitchen.")),
    ("disable_triton_backend", bool, typer.Option(False, "--disable-triton-backend", help="Force-disable the comfy-kitchen Triton backend.")),
    ("fp8_materialization", str, typer.Option("auto", "--fp8-materialization", envvar="COMFYUI_FP8_MATERIALIZATION", help=f"Select how FP8 weights are materialized to bf16/fp16/fp32. Choices: {', '.join(FP8_MATERIALIZATION_MODES)}. auto is reserved for benchmark-selected lowerings; torch uses the graph-visible torch op; comfy_kitchen uses comfy_kitchen's registered backend.")),
]

_DISTRIBUTED_OPTS: list[tuple] = [
    ("rank", Optional[int], typer.Option(None, "--rank", envvar="RANK", help="Global process rank. Defaults to the torchrun RANK environment variable.")),
    ("world_size", Optional[int], typer.Option(None, "--world-size", envvar="WORLD_SIZE", help="Total process count. Defaults to the torchrun WORLD_SIZE environment variable.")),
    ("local_rank", Optional[int], typer.Option(None, "--local-rank", "--local_rank", envvar="LOCAL_RANK", help="Node-local process rank. The dashed spelling is canonical; the underscored spelling is accepted for older launchers.")),
    ("local_world_size", Optional[int], typer.Option(None, "--local-world-size", envvar="LOCAL_WORLD_SIZE", help="Number of processes on this node. Defaults to LOCAL_WORLD_SIZE.")),
    ("master_addr", Optional[str], typer.Option(None, "--master-addr", envvar="MASTER_ADDR", help="Process-group rendezvous host. Defaults to MASTER_ADDR.")),
    ("master_port", Optional[int], typer.Option(None, "--master-port", envvar="MASTER_PORT", help="Process-group rendezvous port. Defaults to MASTER_PORT.")),
    ("pipeline_parallel_size", Optional[int], typer.Option(None, "--pipeline-parallel-size", "-pp", envvar="COMFYUI_PIPELINE_PARALLEL_SIZE", help="Number of pipeline stages. Defaults to world size for external launchers and selected device count otherwise.")),
    ("tensor_parallel_size", Optional[int], typer.Option(None, "--tensor-parallel-size", "-tp", envvar="COMFYUI_TENSOR_PARALLEL_SIZE", help="Number of tensor-parallel ranks. With --guess-settings, identical NVIDIA GPUs on Linux default to the largest power of two of them; one keeps the ordinary model path.")),
    ("ulysses_degree", Optional[int], typer.Option(None, "--ulysses-degree", envvar="COMFYUI_ULYSSES_DEGREE", help="xDiT Ulysses sequence-parallel degree. Defaults to one; one keeps the ordinary attention path.")),
    ("ring_degree", Optional[int], typer.Option(None, "--ring-degree", envvar="COMFYUI_RING_DEGREE", help="xDiT ring-attention degree. Defaults to one; one disables ring attention.")),
    ("nccl_proto", str, typer.Option("auto", "--nccl-proto", click_type=click.Choice(("auto", "simple", "ll", "ll128")), help="Select the NCCL collective protocol. auto preserves NCCL topology tuning.")),
    ("distributed_executor_backend", str, typer.Option("auto", "--distributed-executor-backend", envvar="COMFYUI_DISTRIBUTED_EXECUTOR_BACKEND", click_type=click.Choice(DISTRIBUTED_EXECUTOR_BACKENDS), help="Pipeline executor backend: auto, peer, mp, or external_launcher.")),
]

_VRAM_OPTS: list[tuple] = [
    ("gpu_only", bool, typer.Option(False, "--gpu-only/--no-gpu-only", help="Store and run everything on the GPU.")),
    ("highvram", bool, typer.Option(False, "--highvram/--no-highvram", help="Keep models in GPU memory.")),
    ("normalvram", bool, typer.Option(False, "--normalvram/--no-normalvram", help="Default VRAM usage setting.")),
    ("lowvram", bool, typer.Option(False, "--lowvram/--no-lowvram", help="Reduce UNet's VRAM usage.")),
    ("novram", bool, typer.Option(False, "--novram/--no-novram", help="Minimize VRAM usage.")),
    ("cpu", bool, typer.Option(False, "--cpu/--no-cpu", help="Use CPU for processing.")),
    ("reserve_vram", Optional[float], typer.Option(None, "--reserve-vram", help="Set the amount of vram in GB you want to reserve for use by your OS/other software.")),
    ("vram_headroom", float, typer.Option(0, "--vram-headroom", help="Set extra DynamicVRAM headroom in GB above the default reservation.")),
    ("disable_nvml_pressure", bool, typer.Option(False, "--disable-nvml-pressure", help="Use CUDA instead of NVML for DynamicVRAM memory pressure.")),
    ("disable_dynamic_vram", bool, typer.Option(False, "--disable-dynamic-vram", help="Disable dynamic VRAM and use estimate based model loading.")),
    ("fast_disk", bool, typer.Option(False, "--fast-disk/--no-fast-disk", help="Force disk-backed dynamic loading and offload over unpinned RAM. Can be faster for users with fast NVME disks.")),
    ("disable_fast_disk", bool, typer.Option(False, "--disable-fast-disk", help="Disable disk-backed dynamic loading. Overrides --fast-disk.")),
    ("disable_cuda_graphs", bool, typer.Option(False, "--disable-cuda-graphs", help="Disable CUDA graph capture and replay.")),
    ("disable_comfy_compiler", bool, typer.Option(False, "--disable-comfy-compiler", help="Disable the Comfy model compiler, including its CUDA graph subfeature.")),
    ("assert_graph_breaks", bool, typer.Option(False, "--assert-graph-breaks/--no-assert-graph-breaks", help="Fail on Comfy model compiler graph breaks.")),
]

_PRECISION_OPTS: list[tuple] = [

    ("force_fp32", bool, typer.Option(False, "--force-fp32/--no-force-fp32", help="Force using FP32 precision.")),
    ("force_fp16", bool, typer.Option(False, "--force-fp16/--no-force-fp16", help="Force using FP16 precision.")),
    ("force_bf16", bool, typer.Option(False, "--force-bf16/--no-force-bf16", help="Force using BF16 precision.")),

    ("fp32_unet", bool, typer.Option(False, "--fp32-unet/--no-fp32-unet", help="Run the diffusion model in fp32.")),
    ("fp64_unet", bool, typer.Option(False, "--fp64-unet/--no-fp64-unet", help="Run the diffusion model in fp64.")),
    ("bf16_unet", bool, typer.Option(False, "--bf16-unet/--no-bf16-unet", help="Run the diffusion model in bf16.")),
    ("fp16_unet", bool, typer.Option(False, "--fp16-unet/--no-fp16-unet", help="Run the diffusion model in fp16.")),
    ("fp8_e4m3fn_unet", bool, typer.Option(False, "--fp8_e4m3fn-unet/--no-fp8_e4m3fn-unet", help="Store unet weights in fp8_e4m3fn.")),
    ("fp8_e5m2_unet", bool, typer.Option(False, "--fp8_e5m2-unet/--no-fp8_e5m2-unet", help="Store unet weights in fp8_e5m2.")),
    ("fp8_e8m0fnu_unet", bool, typer.Option(False, "--fp8_e8m0fnu-unet/--no-fp8_e8m0fnu-unet", help="Store unet weights in fp8_e8m0fnu.")),
    ("fp8_storage", bool, typer.Option(True, "--fp8-storage/--no-fp8-storage", help="Preserve native fp8 checkpoint weights as resident fp8 storage when the device supports fp8 storage with upcasted math.")),

    ("fp16_vae", bool, typer.Option(False, "--fp16-vae/--no-fp16-vae", help="Run the VAE in FP16 precision.")),
    ("fp32_vae", bool, typer.Option(False, "--fp32-vae/--no-fp32-vae", help="Run the VAE in full precision fp32.")),
    ("bf16_vae", bool, typer.Option(False, "--bf16-vae/--no-bf16-vae", help="Run the VAE in BF16 precision.")),
    ("cpu_vae", bool, typer.Option(False, "--cpu-vae/--no-cpu-vae", help="Run the VAE on the CPU.")),

    ("fp8_e4m3fn_text_enc", bool, typer.Option(False, "--fp8_e4m3fn-text-enc/--no-fp8_e4m3fn-text-enc", help="Store text encoder weights in fp8 (e4m3fn).")),
    ("fp8_e5m2_text_enc", bool, typer.Option(False, "--fp8_e5m2-text-enc/--no-fp8_e5m2-text-enc", help="Store text encoder weights in fp8 (e5m2).")),
    ("fp16_text_enc", bool, typer.Option(False, "--fp16-text-enc/--no-fp16-text-enc", help="Store text encoder weights in fp16.")),
    ("fp32_text_enc", bool, typer.Option(False, "--fp32-text-enc/--no-fp32-text-enc", help="Store text encoder weights in fp32.")),
    ("bf16_text_enc", bool, typer.Option(False, "--bf16-text-enc/--no-bf16-text-enc", help="Store text encoder weights in bf16.")),
    ("fp16_intermediates", bool, typer.Option(False, "--fp16-intermediates/--no-fp16-intermediates", help="Experimental: Use fp16 for intermediate tensors between nodes instead of fp32.")),
]

_ATTENTION_OPTS: list[tuple] = [
    ("use_split_cross_attention", bool, typer.Option(False, "--use-split-cross-attention/--no-use-split-cross-attention", help="Use split cross-attention optimization.")),
    ("use_quad_cross_attention", bool, typer.Option(False, "--use-quad-cross-attention/--no-use-quad-cross-attention", help="Use sub-quadratic cross-attention optimization.")),
    ("use_pytorch_cross_attention", bool, typer.Option(False, "--use-pytorch-cross-attention/--no-use-pytorch-cross-attention", help="Use PyTorch's cross-attention function.")),
    ("use_sage_attention", bool, typer.Option(False, "--use-sage-attention/--no-use-sage-attention", help="Use sage attention.")),
    ("use_flash_attention", bool, typer.Option(False, "--use-flash-attention/--no-use-flash-attention", help="Use FlashAttention.")),
    ("use_ck_attention", bool, typer.Option(False, "--use-ck-attention/--no-use-ck-attention", help="Use Comfy Kitchen attention.")),
    ("disable_xformers", bool, typer.Option(False, "--disable-xformers", help="Disable xformers.")),
    ("force_upcast_attention", bool, typer.Option(False, "--force-upcast-attention/--no-force-upcast-attention", help="Force upcasting of attention.")),
    ("dont_upcast_attention", bool, typer.Option(False, "--dont-upcast-attention", help="Disable upcasting of attention.")),
]

_MEMORY_OPTS: list[tuple] = [
    ("async_offload", Optional[int], typer.Option(None, "--async-offload", help="Use async weight offloading. An optional argument controls the amount of offload streams.")),
    ("disable_async_offload", bool, typer.Option(False, "--disable-async-offload", help="Disable async weight offloading.")),
    ("force_non_blocking", bool, typer.Option(False, "--force-non-blocking/--no-force-non-blocking", help="Force non-blocking operations for all applicable tensors.")),
    ("disable_smart_memory", bool, typer.Option(False, "--disable-smart-memory", help="Disable smart memory management.")),
    ("disable_pinned_memory", bool, typer.Option(False, "--disable-pinned-memory", help="Disable pinned memory use.")),
    ("pinned_memory_reserve", Optional[float], typer.Option(None, "--pinned-memory-reserve", help="GB of the effective memory limit (host RAM, or the cgroup limit in a container) that pinned memory always leaves free, checked against memory available at each pin. Default: the larger of 4 GB and 10% of the limit.")),
    ("max_pinned_memory", Optional[float], typer.Option(None, "--max-pinned-memory", help="Cap in GB on pinned host memory. Default and upper bound: the effective memory limit minus --pinned-memory-reserve.")),
]

_CACHE_OPTS: list[tuple] = [
    ("cache_classic", bool, typer.Option(False, "--cache-classic/--no-cache-classic", help="Use the old style (aggressive) caching.")),
    ("cache_lru", int, typer.Option(0, "--cache-lru", help="Use LRU caching with a maximum of N node results cached. May use more RAM/VRAM.")),
    ("cache_none", bool, typer.Option(False, "--cache-none/--no-cache-none", help="Reduced RAM/VRAM usage at the expense of executing every node for each run.")),
    ("cache_ram", float, typer.Option(0, "--cache-ram", help="Use RAM pressure caching with the specified headroom threshold. Default (when no value is provided): 25%% of system RAM (min 4GB, max 32GB).")),
    ("high_ram", bool, typer.Option(False, "--high-ram/--no-high-ram", help="Prefer high-RAM model caching and pinning behavior.")),
]

_PREVIEW_OPTS: list[tuple] = [
    ("preview_method", str, typer.Option("auto", "--preview-method", click_type=click.Choice(["none", "auto", "latent2rgb", "taesd"]), help="Method for generating previews.")),
    ("preview_size", int, typer.Option(512, "--preview-size", help="Sets the maximum preview size for sampler nodes.")),
]

_PERF_OPTS: list[tuple] = [
    ("fast", Optional[list[str]], typer.Option(None, "--fast", help="Enable some untested and potentially quality deteriorating optimizations. Valid optimizations: fp16_accumulation, fp8_matrix_mult, cublas_ops, autotune, dynamic_vram.")),
    ("deterministic", bool, typer.Option(False, "--deterministic/--no-deterministic", help="Use deterministic algorithms where possible.")),
    ("default_hashing_function", str, typer.Option("sha256", "--default-hashing-function", click_type=click.Choice(["md5", "sha1", "sha256", "sha512"]), help="Hash function for duplicate filename / contents comparison.")),
    ("force_channels_last", bool, typer.Option(False, "--force-channels-last/--no-force-channels-last", help="Force channels last format when inferencing the models.")),
    ("force_hf_local_dir_mode", bool, typer.Option(False, "--force-hf-local-dir-mode/--no-force-hf-local-dir-mode", help="Download HF repos with local_dir instead of cache_dir.")),
    ("mmap_torch_files", bool, typer.Option(False, "--mmap-torch-files/--no-mmap-torch-files", help="Use mmap when loading ckpt/pt files.")),
    ("disable_mmap", bool, typer.Option(False, "--disable-mmap", help="Don't use mmap when loading safetensors.")),
]

_NODE_OPTS: list[tuple] = [
    ("disable_metadata", bool, typer.Option(False, "--disable-metadata", help="Disable saving metadata with outputs.")),
    ("disable_all_custom_nodes", bool, typer.Option(False, "--disable-all-custom-nodes", help="Disable loading all custom nodes.")),
    ("whitelist_custom_nodes", Optional[list[str]], typer.Option(None, "--whitelist-custom-nodes", help="Specify custom node folders to load even when --disable-all-custom-nodes is enabled.")),
    ("blacklist_custom_nodes", Optional[list[str]], typer.Option(None, "--blacklist-custom-nodes", help="Specify custom node folders to never load. Accepts shell-style globs.")),
    ("disable_partner_nodes", bool, typer.Option(False, "--disable-partner-nodes", help="Disable loading partner nodes (the paid API nodes). The frontend keeps its internet access.")),
    ("offline", bool, typer.Option(False, "--offline/--no-offline", help="Prevent the frontend from communicating with the internet by setting a restrictive Content-Security-Policy header. Also disables partner nodes.")),
    ("disable_api_nodes", bool, typer.Option(False, "--disable-api-nodes", help="Deprecated alias for --offline. Use --offline, or --disable-partner-nodes to only disable partner nodes.")),
    ("enable_eval", bool, typer.Option(False, "--enable-eval/--no-enable-eval", help="Enable nodes that can evaluate Python code in workflows.")),
    ("enable_video_to_image_fallback", bool, typer.Option(False, "--enable-video-to-image-fallback/--no-enable-video-to-image-fallback", help="Enable video-to-image fallback.")),
    ("disable_known_models", bool, typer.Option(False, "--disable-known-models", help="Disables automatic downloads of known models.")),
    ("enable_assets", bool, typer.Option(False, "--enable-assets/--no-enable-assets", help="Enable the assets API and asset management features.")),
    ("enable_asset_hashing", bool, typer.Option(False, "--enable-asset-hashing/--no-enable-asset-hashing", help="Compute blake3 content hashes when scanning assets.")),
    ("disable_assets_autoscan", bool, typer.Option(False, "--disable-assets-autoscan", help="Disable asset scanning on startup for database synchronization.")),
]

_TELEMETRY_OPTS: list[tuple] = [
    ("otel_service_name", str, typer.Option("comfyui", "--otel-service-name", envvar="OTEL_SERVICE_NAME", help="OpenTelemetry service.name for emitted spans. Use a distinct value per benchmark run, e.g. comfyui-torch-2.12.0, so file-collected traces are easy to filter.")),
    ("otel_service_version", Optional[str], typer.Option(None, "--otel-service-version", envvar="OTEL_SERVICE_VERSION", help="OpenTelemetry service.version for emitted spans. Use this to record the tested ComfyUI build, torch version, or package revision.")),
    ("otel_exporter_otlp_endpoint", Optional[str], typer.Option(None, "--otel-exporter-otlp-endpoint", envvar="OTEL_EXPORTER_OTLP_ENDPOINT", help="Enable trace export. Use file:///tmp/comfyui-traces.jsonl for direct local JSONL spans, or an OTLP/HTTP collector endpoint such as http://127.0.0.1:4318.")),
]

_LOGGING_OPTS: list[tuple] = [
    ("logging_level", str, typer.Option("INFO", "--logging-level", click_type=click.Choice(["DEBUG", "DETAIL", "INFO", "WARNING", "ERROR", "CRITICAL"]), help="Specifies the logging level.")),
    ("debug_hang", bool, typer.Option(False, "--debug-hang/--no-debug-hang", help="Enable stack trace dumps on Ctrl-C for debugging hangs.")),
]

_PIP_FACADE_OPTS: list[tuple] = [
    ("pip_facade_snapshot_max_age_seconds", int, typer.Option(0, "--pip-facade-snapshot-max-age-seconds", min=0, help="Maximum registry snapshot age in seconds; 0 disables age enforcement.")),
    ("pip_facade_registry_base_url", str, typer.Option("https://api.comfy.org", "--pip-facade-registry-base-url", help="Base URL for the Comfy registry API used to resolve custom node versions.")),
    ("pip_facade_cache_prefix", Optional[str], typer.Option(None, "--pip-facade-cache-prefix", help="Writable fsspec URI prefix where generated facade wheels are cached, e.g. /var/cache/comfyui, file:///var/cache/comfyui, or s3://bucket/prefix.")),
    ("pip_facade_cache_s3_endpoint_url", Optional[str], typer.Option(None, "--pip-facade-cache-s3-endpoint-url", envvar="PIP_FACADE_CACHE_S3_ENDPOINT_URL", help="Custom S3-compatible endpoint URL for an s3:// facade cache prefix, e.g. a SeaweedFS S3 gateway.")),
    ("pip_facade_cache_revision", Optional[int], typer.Option(None, "--pip-facade-cache-revision", help="Override the wheel cache revision. Changing this invalidates all cached wheels.")),
    ("pip_facade_only_known_nodes", bool, typer.Option(False, "--pip-facade-only-known-nodes/--no-pip-facade-only-known-nodes", help="Only expose nodes covered by the local custom node compatibility registry.")),
    ("pip_facade_snapshot_uri", Optional[str], typer.Option(None, "--pip-facade-snapshot-uri", help="Read facade registry metadata from this fsspec URI instead of querying the live registry API, e.g. file:///data/registry.sqlite.xz, s3://bucket/registry.sqlite.xz, or pkg://comfy.custom_nodes/pip_facade_registry_snapshot.sqlite.xz.")),
]

_PIP_FACADE_SNAPSHOT_OPTS: list[tuple] = [
    ("pip_facade_snapshot_output", str, typer.Argument(..., help="Output snapshot path. Use a .xz suffix or --pip-facade-snapshot-compression=xz for a compressed archive.")),
    ("pip_facade_snapshot_compression", str, typer.Option("auto", "--pip-facade-snapshot-compression", click_type=click.Choice(["auto", "none", "xz"]), help="Snapshot compression mode. 'auto' uses the output suffix.")),
    ("pip_facade_snapshot_overwrite", bool, typer.Option(False, "--pip-facade-snapshot-overwrite/--no-pip-facade-snapshot-overwrite", help="Overwrite an existing snapshot output file.")),
]

_MISC_OPTS: list[tuple] = [
    ("guess_settings", bool, typer.Option(True, "-g", "--guess-settings/--no-guess-settings", envvar="COMFYUI_GUESS_SETTINGS", help="Auto-detect best settings for this machine (GPU type, RAM, attention backend, etc.). Enabled by default; pass --no-guess-settings or COMFYUI_GUESS_SETTINGS=0 to disable.")),
    ("database_url", Optional[str], typer.Option(None, "--database-url", help="Specify the database URL, e.g. 'sqlite:///:memory:'.")),
    ("feature_flag", Optional[list[str]], typer.Option(None, "--feature-flag", help="Set a server feature flag in KEY[=VALUE] form.")),
    ("list_feature_flags", bool, typer.Option(False, "--list-feature-flags/--no-list-feature-flags", help="Print known CLI-settable feature flags as JSON and exit.")),
    ("panic_when", Optional[list[str]], typer.Option(None, "--panic-when", help="List of fully qualified exception class names to panic (sys.exit(1)) when a workflow raises it.")),
    ("executor_factory", str, typer.Option("ThreadPoolExecutor", "--executor-factory", help="Either ThreadPoolExecutor or ProcessPoolExecutor.")),
]

_WORKFLOW_OVERRIDE_OPTS: list[tuple] = [
    ("prompt", Optional[str], typer.Option(None, "--prompt", help="Override the positive prompt text in workflows.")),
    ("negative_prompt", Optional[str], typer.Option(None, "--negative-prompt", help="Override the negative prompt text in workflows.")),
    ("steps", Optional[int], typer.Option(None, "--steps", help="Override the number of sampling steps in workflows.")),
    ("seed", Optional[int], typer.Option(None, "--seed", help="Override the seed in sampler and noise nodes in workflows.")),
    ("quantity", int, typer.Option(1, "--quantity", min=1, help="Run, submit, or convert this many jobs. API workflows use a random base seed unless --seed is set, then increment seed values by 1 per job; UI workflows honor each seed widget's control_after_generate mode.")),
    ("cfg", Optional[float], typer.Option(None, "--cfg", help="Override the CFG scale in sampler nodes.")),
    ("sampler", Optional[str], typer.Option(None, "--sampler", help="Override the sampler name in sampler nodes.")),
    ("scheduler", Optional[str], typer.Option(None, "--scheduler", help="Override the scheduler in sampler/scheduler nodes.")),
    ("denoise", Optional[float], typer.Option(None, "--denoise", help="Override the denoise strength in sampler nodes.")),
    ("width", Optional[int], typer.Option(None, "--width", help="Override the width in latent image nodes.")),
    ("height", Optional[int], typer.Option(None, "--height", help="Override the height in latent image nodes.")),
    ("batch_size", Optional[int], typer.Option(None, "--batch-size", help="Override the batch size in latent image nodes.")),
    ("checkpoint", Optional[str], typer.Option(None, "--checkpoint", help="Override the checkpoint model name.")),
    ("diffusion_model", Optional[str], typer.Option(None, "--diffusion-model", help="Override the diffusion model name (UNETLoader / DiffusionModelLoader / UnetLoaderGGUF unet_name). Accepts a bare filename (looked up in models/diffusion_models/ or models/unet/) or an https:// / hf:// URI.")),
    ("image", Optional[list[str]], typer.Option(None, "--image", help="Override image inputs in workflows. Accepts file paths or URIs.")),
    ("video", Optional[list[str]], typer.Option(None, "--video", help="Override video inputs in workflows. Accepts file paths or URIs.")),
    ("audio", Optional[list[str]], typer.Option(None, "--audio", help="Override audio inputs in workflows. Accepts file paths or URIs.")),
    ("output", Optional[str], typer.Option(None, "-o", "--output", help="Override the output directory for workflows.")),
    ("set", Optional[list[str]], typer.Option(None, "--set", help="Override arbitrary node inputs. Format: node_id.inputs.field=value")),
    ("add_lora", Optional[list[str]], typer.Option(None, "--add-lora", help="Inject a LoRA into the workflow. Format: name[:strength_model[:strength_clip]]. Bare name looks up in models/loras/; paths, hf:// URIs, and https:// URLs (including civitai.com) are resolved through the model downloader. Repeat to stack multiple LoRAs.")),
    ("compile", bool, typer.Option(False, "--compile/--no-compile", help="Wrap the workflow's MODEL chain tail with TorchCompileModel for a 2-4x step-time speedup after a first-run compile cost.")),
]

# Subset of override options for commands that don't execute the workflow
# in-process (workflows submit, workflows convert). They should not accept
# --output as a runtime output-directory override, which would also collide
# with the local -o/--output that writes a JSON artifact to disk.
_WORKFLOW_OVERRIDE_OPTS_NO_OUTPUT: list[tuple] = [
    opt for opt in _WORKFLOW_OVERRIDE_OPTS if opt[0] != "output"
]

_RUN_WORKFLOWS_ARGUMENT = typer.Argument(
    ...,
    help="Workflow files, URIs, template names, '-' for stdin, or literal JSON.",
)
_RUN_ALL_OPTION = typer.Option(
    False,
    "--all/--no-all",
    "-a",
    help="Install missing custom nodes and download missing models before running.",
)
_RUN_DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run/--no-dry-run",
    help="Print the execution plan (discovered models, folder-path registrations, missing downloads, disk free) and exit. Implies --all.",
)
_RUN_DISABLE_PROGRESS_OPTION = typer.Option(
    False,
    "--disable-progress",
    help="Disable CLI progress bars.",
)
_RUN_BLOCK_RUNTIME_PACKAGE_INSTALLATION_OPTION = typer.Option(
    False,
    "--block-runtime-package-installation/--no-block-runtime-package-installation",
    help="Block runtime package installations.",
)

_COMPUTE_OPTS = (
    _DEVICE_OPTS + _DISTRIBUTED_OPTS + _VRAM_OPTS + _PRECISION_OPTS + _ATTENTION_OPTS +
    _MEMORY_OPTS + _CACHE_OPTS + _PREVIEW_OPTS + _PERF_OPTS
)

_ALL_SHARED_OPTS = (
    _DIRECTORY_OPTS + _COMPUTE_OPTS + _NODE_OPTS +
    _TELEMETRY_OPTS + _LOGGING_OPTS + _MISC_OPTS
)

_NULLABLE_LIST_FIELDS = frozenset({
    "fast", "base_paths", "extra_model_paths_config", "panic_when",
    "whitelist_custom_nodes", "blacklist_custom_nodes", "workflows",
    "image", "video", "audio", "set", "add_lora",
    "add_model_folder_path",
    "enable_comfy_kitchen_backends", "disable_comfy_kitchen_backends",
})


def _with_options(*option_groups):
    """Add shared option groups to a Typer command function.

    The decorated function should accept **kwargs for the injected options.
    Options are appended to the function's signature so Typer discovers them.
    """
    combined = []
    for group in option_groups:
        combined.extend(group)

    def decorator(func):
        sig = inspect.signature(func)
        params = [p for p in sig.parameters.values()
                  if p.kind != inspect.Parameter.VAR_KEYWORD]
        for name, annotation, default in combined:
            params.append(inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default, annotation=annotation,
            ))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.__signature__ = sig.replace(parameters=params)
        return wrapper
    return decorator


def _set_config_context(config: Configuration):
    """Set config into the execution context so main_pre reads the correct values.

    Must be called before any import that transitively triggers main_pre.py's
    import-time side effects (e.g. importing main.py, model_management, nodes.package).
    """
    from dataclasses import replace
    from ..execution_context import comfyui_execution_context, current_execution_context
    ctx = replace(current_execution_context(), configuration=config)
    comfyui_execution_context.set(ctx)


@contextlib.contextmanager
def _application_runtime(config: Configuration):
    """Initialize one shared application/distributed lifecycle for every runner."""
    from ..component_model.setup import (
        prepare_distributed_environment,
        shutdown_tracing,
        setup_distributed_device,
        setup_distributed_runtime,
        setup_post_torch,
        setup_pre_torch,
    )

    distributed = prepare_distributed_environment(config)
    setup_pre_torch(config)
    setup_distributed_device(distributed)
    _set_config_context(config)
    runtime = setup_distributed_runtime(distributed)
    setup_post_torch(config)
    try:
        if runtime is not None and not runtime.is_driver:
            runtime.run_worker_service()
            yield False
        else:
            yield True
    finally:
        if runtime is not None:
            runtime.close()
        shutdown_tracing()


def _validate_mutex(params: dict, group_name: str, fields: tuple[str, ...]):
    """Raise if more than one field in a mutually exclusive group is set to a truthy non-default value."""
    set_modes = [f for f in fields if params.get(f)]
    if len(set_modes) > 1:
        flags = ", ".join(f"--{f.replace('_', '-')}" for f in set_modes)
        raise typer.BadParameter(f"Only one of {flags} can be set ({group_name})")


def _parse_listen_address(value: str) -> tuple[str, int | None]:
    """Parse an optional port from a listen address.

    Supports ``host:port``, ``[ipv6]:port``, and bare addresses.
    Returns ``(host, port)`` where port is None if not specified.
    """
    # [ipv6]:port
    if value.startswith("[") and "]:" in value:
        bracket_end = value.index("]:")
        return value[1:bracket_end], int(value[bracket_end + 2:])
    # host:port (but not bare IPv6 like :: or ::1)
    if ":" in value and value.count(":") == 1:
        host, port_str = value.rsplit(":", 1)
        if port_str.isdigit():
            return host, int(port_str)
    return value, None


def _build_config(params: dict) -> Configuration:
    """Build Configuration from Typer-parsed parameters."""
    _validate_mutex(params, "VRAM mode", VRAM_MODES)
    _validate_mutex(params, "precision", PRECISION_MODES)
    _validate_mutex(params, "UNet precision", UNET_MODES)
    _validate_mutex(params, "VAE precision", VAE_MODES)
    _validate_mutex(params, "text encoder precision", TEXT_ENC_MODES)
    _validate_mutex(params, "attention", ATTENTION_MODES)
    _validate_mutex(params, "upcast attention", UPCAST_MODES)

    filtered = {
        k: v for k, v in params.items()
        if v is not None and k not in ("ctx", "config") and not k.startswith("_")
    }

    if "fast" in filtered:
        raw = filtered["fast"]
        items = set()
        for v in raw:
            for piece in v.split(","):
                piece = piece.strip()
                if piece:
                    items.add(PerformanceFeature(piece))
        filtered["fast"] = items

    if "preview_method" in filtered and isinstance(filtered["preview_method"], str):
        filtered["preview_method"] = LatentPreviewMethod(filtered["preview_method"])

    if filtered.get("quantity", 1) < 1:
        raise typer.BadParameter("--quantity must be at least 1")

    for list_field in ("base_paths", "extra_model_paths_config", "panic_when",
                       "whitelist_custom_nodes", "blacklist_custom_nodes",
                       "image", "video", "audio", "workflows",
                       "add_model_folder_path",
                       "enable_comfy_kitchen_backends", "disable_comfy_kitchen_backends"):
        if list_field in filtered and isinstance(filtered[list_field], (list, tuple)):
            expanded = []
            for v in filtered[list_field]:
                for piece in str(v).split(","):
                    piece = piece.strip()
                    if piece:
                        expanded.append(piece)
            filtered[list_field] = expanded

    # windows_standalone_build enables auto_launch, but disable_auto_launch always wins
    if filtered.get("windows_standalone_build"):
        filtered["auto_launch"] = True
    if filtered.get("disable_auto_launch"):
        filtered["auto_launch"] = False
    if filtered.get("force_fp16"):
        filtered["fp16_unet"] = True
    if filtered.get("disable_comfy_compiler"):
        filtered["disable_cuda_graphs"] = True
    if filtered.get("disable_api_nodes"):
        logger.warning("--disable-api-nodes is deprecated and will be removed in a future version. It currently behaves like --offline. Use --offline to keep the frontend offline, or --disable-partner-nodes to only disable partner nodes.")
        filtered["offline"] = True
    if filtered.get("offline"):
        filtered["disable_partner_nodes"] = True
    if filtered.get("novram"):
        filtered["disable_dynamic_vram"] = True
        filtered["disable_smart_memory"] = True

    # Parse host:port from --listen (e.g. "0.0.0.0:8189" or "[::]:8189")
    if "listen" in filtered:
        listen_val = filtered["listen"]
        parsed_host, parsed_port = _parse_listen_address(listen_val)
        if parsed_port is not None:
            filtered["listen"] = parsed_host
            filtered["port"] = parsed_port

    config = Configuration(**filtered)
    if "database_url" in filtered:
        config._database_url_explicit = True
    return config


def _load_config_file(path: str) -> dict:
    """Load a YAML or JSON config file."""
    import yaml
    with open(path) as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f) or {}


def _find_default_config_file() -> Optional[str]:
    """Find the first default config file that exists."""
    for name in ("config.yaml", "config.json", "config.cfg", "config.ini"):
        if os.path.exists(name):
            return name
    return None


def _format_cli_args(args: list[str]) -> str:
    return shlex.join(args)


def _warn_unknown_cli_args(args: list[str]) -> None:
    if not args:
        return
    typer.echo(
        f"[WARNING] Ignoring unknown CLI argument(s): {_format_cli_args(args)}",
        err=True,
    )


def _remove_unknown_option_args_from_workflows(workflows: list[str]) -> list[str]:
    unknown_args = [w for w in workflows if w != "-" and w.startswith("-")]
    _warn_unknown_cli_args(unknown_args)
    if not unknown_args:
        return workflows
    unknown_set = set(unknown_args)
    return [w for w in workflows if w not in unknown_set]


def _parse_plugin_args(ctx: typer.Context, config: Configuration):
    """Load comfyui.custom_config entry points and parse their args from ctx.args."""
    from importlib.metadata import entry_points
    import configargparse

    parser = configargparse.ArgParser(add_help=False)
    for ep in entry_points(group='comfyui.custom_config'):
        plugin = ep.load()
        result = plugin(parser)
        if result is not None:
            parser = result

    if ctx.args:
        plugin_args, unknown_args = parser.parse_known_args(ctx.args)
        for k, v in vars(plugin_args).items():
            setattr(config, k, v)
        _warn_unknown_cli_args(unknown_args)


def _collect_params(local_vars: dict, kwargs: dict) -> dict:
    """Merge named locals and **kwargs into a single params dict for _build_config."""
    params = {k: v for k, v in local_vars.items() if k not in ("ctx", "kwargs")}
    params.update(kwargs)
    for key in _NULLABLE_LIST_FIELDS:
        if key in params and params[key] is None:
            params[key] = []
    return params


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """ComfyUI - The most powerful and modular diffusion model GUI and backend."""
    if ctx.invoked_subcommand is None:
        # No subcommand specified - this shouldn't happen because
        # entrypoint() inserts "serve" when no subcommand is given.
        # But as a safety net, invoke serve.
        ctx.invoke(serve)


@app.command(context_settings={**_COMFYUI_ENV, "allow_extra_args": True, "ignore_unknown_options": True}, rich_help_panel="Server")
@_with_options(_ALL_SHARED_OPTS, _WORKFLOW_OVERRIDE_OPTS)
def serve(
    ctx: typer.Context,

    daemon: bool = typer.Option(False, "-d", "--daemon/--no-daemon", help="Run as a background daemon."),
    pid_file: Optional[str] = typer.Option(None, "--pid-file", help="PID file path (default: ~/.comfyui/comfyui.pid)."),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Log file path (default: ~/.comfyui/comfyui.log)."),
    listen: str = typer.Option("127.0.0.1", "-H", "--listen", help="Specify the IP address to listen on (default: 127.0.0.1). You can give a list of ip addresses by separating them with a comma like: 127.2.2.2,127.3.3.3 If --listen is provided without an argument, it defaults to 0.0.0.0,:: (listens on all ipv4 and ipv6)"),
    port: int = typer.Option(8188, help="Set the listen port."),
    tls_keyfile: Optional[str] = typer.Option(None, "--tls-keyfile", help="Path to TLS key file."),
    tls_certfile: Optional[str] = typer.Option(None, "--tls-certfile", help="Path to TLS certificate file."),
    enable_cors_header: Optional[str] = typer.Option(None, "--enable-cors-header", help="Enable CORS (Cross-Origin Resource Sharing) with optional origin or allow all with default '*'."),
    max_upload_size: float = typer.Option(100.0, "--max-upload-size", help="Set the maximum upload size in MB."),
    auto_launch: bool = typer.Option(False, "--auto-launch/--no-auto-launch", help="Automatically launch ComfyUI in the default browser."),
    disable_auto_launch: bool = typer.Option(False, "--disable-auto-launch", help="Disable auto launching the browser."),
    external_address: Optional[str] = typer.Option(None, "--external-address", help="Specifies a base URL for external addresses reported by the API, such as for image paths."),
    multi_user: bool = typer.Option(False, "--multi-user/--no-multi-user", help="Enable multi-user mode with per-user storage."),
    enable_compress_response_body: bool = typer.Option(False, "--enable-compress-response-body/--no-enable-compress-response-body", help="Enable compressing response body."),
    enable_manager: bool = typer.Option(False, "--enable-manager/--no-enable-manager", help="Enable the ComfyUI-Manager feature."),
    disable_manager_ui: bool = typer.Option(False, "--disable-manager-ui", help="Disables only the ComfyUI-Manager UI."),
    enable_manager_legacy_ui: bool = typer.Option(False, "--enable-manager-legacy-ui/--no-enable-manager-legacy-ui", help="Enables the legacy UI of ComfyUI-Manager."),
    dont_print_server: bool = typer.Option(False, "--dont-print-server", help="Don't print server output."),
    log_stdout: bool = typer.Option(False, "--log-stdout/--no-log-stdout", help="Send normal process output to stdout instead of stderr (default)."),
    quick_test_for_ci: bool = typer.Option(False, "--quick-test-for-ci/--no-quick-test-for-ci", help="Enable quick testing mode for CI."),
    windows_standalone_build: bool = typer.Option(False, "--windows-standalone-build/--no-windows-standalone-build", help="Enable features for standalone Windows build."),
    create_directories: bool = typer.Option(False, "--create-directories/--no-create-directories", help="Creates the default models/, input/, output/ and temp/ directories, then exits."),
    plausible_analytics_base_url: Optional[str] = typer.Option(None, "--plausible-analytics-base-url", help="Base URL for server-side analytics."),
    plausible_analytics_domain: Optional[str] = typer.Option(None, "--plausible-analytics-domain", help="Domain for analytics events."),
    analytics_use_identity_provider: bool = typer.Option(False, "--analytics-use-identity-provider/--no-analytics-use-identity-provider", help="Use platform identifiers for analytics."),
    distributed_queue_connection_uri: Optional[str] = typer.Option(None, "--distributed-queue-connection-uri", help="Servers and clients will connect to this AMQP URL to form a distributed queue and exchange prompt execution requests and progress updates."),
    distributed_queue_worker: bool = typer.Option(False, "--distributed-queue-worker/--no-distributed-queue-worker", help="Workers will pull requests off the AMQP URL."),
    distributed_queue_frontend: bool = typer.Option(False, "--distributed-queue-frontend/--no-distributed-queue-frontend", help="Frontends will start the web UI and connect to the provided AMQP URL to submit prompts."),
    distributed_queue_name: str = typer.Option("comfyui", "--distributed-queue-name", help="This name will be used by the frontends and workers to exchange prompt requests and replies."),
    max_queue_size: int = typer.Option(65536, "--max-queue-size", help="The API will reject prompt requests if the queue's size exceeds this value."),
    front_end_version: str = typer.Option(DEFAULT_VERSION_STRING, "--front-end-version", help="Specifies the version of the frontend to be used. Format: [owner]/[repo]@[version]."),
    front_end_root: Optional[str] = typer.Option(None, "--front-end-root", help="The local filesystem path to the directory where the frontend is located. Overrides --front-end-version."),
    openai_api_key: Optional[str] = typer.Option(None, "--openai-api-key", envvar="OPENAI_API_KEY", help="Configures the OpenAI API Key for the OpenAI nodes."),
    ideogram_api_key: Optional[str] = typer.Option(None, "--ideogram-api-key", envvar="IDEOGRAM_API_KEY", help="Configures the Ideogram API Key for the Ideogram nodes."),
    anthropic_api_key: Optional[str] = typer.Option(None, "--anthropic-api-key", envvar="ANTHROPIC_API_KEY", help="Configures the Anthropic API key for its nodes related to Claude functionality."),
    google_api_key: Optional[str] = typer.Option(None, "--google-api-key", envvar="GOOGLE_API_KEY", help="Google API key for Gemini models."),
    comfy_api_base: str = typer.Option("https://api.comfy.org", "--comfy-api-base", help="Set the base URL for the ComfyUI API."),
    block_runtime_package_installation: bool = typer.Option(False, "--block-runtime-package-installation/--no-block-runtime-package-installation", help="When set, custom nodes like ComfyUI Manager, Easy Use, Nunchaku and others will not be able to use pip or uv to install packages at runtime (experimental)."),
    workflows: Optional[list[str]] = typer.Option(None, "--workflows", help="Execute the API workflow(s) and exit. Each value can be a file path, a literal JSON string starting with '{', a URI (https://, s3://, hf://, etc.), or '-' for stdin."),
    disable_requests_caching: bool = typer.Option(False, "--disable-requests-caching", help="Disable requests caching."),
    disable_manager_model_fallback: bool = typer.Option(False, "--disable-manager-model-fallback", help="Disable manager model database fallback."),
    refresh_manager_models: bool = typer.Option(False, "--refresh-manager-models/--no-refresh-manager-models", help="Fetch latest model list from GitHub."),
    **kwargs,
):
    """Start the ComfyUI server (default command).

    When no subcommand is given, comfyui defaults to serve. Use --guess-settings
    to auto-detect optimal hardware configuration (recommended for most users).

    \b
    VRAM Management:
      --novram             Aggressively offload all model weights from GPU after
                           each operation. Use when you have 16 GB VRAM or less,
                           or when running large models. Despite the name, the GPU
                           is still used for computation.
      --lowvram            Keep some UNet layers on GPU but offload most. Less
                           aggressive than --novram, better throughput but uses
                           more VRAM.
      --normalvram         Default balanced mode. Models are loaded onto GPU as
                           needed and kept resident when memory allows.
      --highvram           Keep all models in GPU memory. Fastest, but requires
                           enough VRAM to hold all loaded models simultaneously.
      --disable-dynamic-vram
                           Disable runtime VRAM monitoring (PyTorch 2.8+). By
                           default, ComfyUI checks actual RAM and VRAM usage at
                           inference time to decide whether to offload. This flag
                           reverts to static size estimates.
      --reserve-vram N     Keep N GB of VRAM free for other applications. ComfyUI
                           subtracts this from available VRAM when deciding what
                           fits on GPU.

    \b
    Performance (--fast):
      cublas_ops           Use optimized cuBLAS matrix multiply on NVIDIA Ampere+
                           (RTX 30xx, 40xx, 50xx, A100). Auto-enabled by
                           --guess-settings on NVIDIA GPUs.
      fp16_accumulation    Use FP16 for accumulation. Faster but lower precision.
      fp8_matrix_mult      Use FP8 for matrix multiplication.
      autotune             Enable CUDA autotuning for kernel selection.
      dynamic_vram         Enable runtime VRAM monitoring (default on PyTorch 2.8+).
    \b
      Example: comfyui serve --fast cublas_ops,fp16_accumulation

    \b
    Precision:
      Model-specific precision flags (--fp16-vae, --fp32-vae, --bf16-unet, etc.)
      override the default auto-detection. Use --fp32-vae on AMD GPUs if you see
      VAE decode artifacts. Use --force-fp16 to halve memory usage globally.

    \b
    Examples:
      comfyui serve --guess-settings
      comfyui serve --novram --fast cublas_ops
      comfyui serve --listen 0.0.0.0 --port 8188
      comfyui serve --daemon --guess-settings
      comfyui serve --fp32-vae --use-sage-attention
      comfyui serve --workflows my_workflow.json --prompt "a sunset" --steps 20
    """
    _daemon = daemon
    _pid_file = pid_file
    _log_file = log_file

    params = _collect_params(locals(), kwargs)
    for _k in ("daemon", "pid_file", "log_file", "_daemon", "_pid_file", "_log_file"):
        params.pop(_k, None)

    if params.get("otel_service_version") is None:
        from .. import __version__
        params["otel_service_version"] = __version__

    config = _build_config(params)
    _parse_plugin_args(ctx, config)

    if _daemon:
        from ..distributed.config import resolve_distributed_configuration
        if resolve_distributed_configuration(config).externally_launched:
            raise typer.BadParameter("--daemon cannot be combined with an external distributed launcher")
        from .daemon import daemonize, default_pid_file, default_log_file
        daemonize(_pid_file or default_pid_file(), _log_file or default_log_file())

    with _application_runtime(config) as run_driver:
        if not run_driver:
            return
        from .main import _start_comfyui
        try:
            asyncio.run(_start_comfyui(configuration=config))
        except KeyboardInterrupt:
            pass



@app.command(context_settings={**_COMFYUI_ENV, "allow_extra_args": True, "ignore_unknown_options": True}, rich_help_panel="Server")
@_with_options(_ALL_SHARED_OPTS)
def worker(
    ctx: typer.Context,
    distributed_queue_connection_uri: str = typer.Option(..., "--distributed-queue-connection-uri", help="AMQP URL for distributed queue."),
    distributed_queue_name: str = typer.Option("comfyui", "--distributed-queue-name", help="Queue name."),
    block_runtime_package_installation: bool = typer.Option(True, "--block-runtime-package-installation/--no-block-runtime-package-installation", help="Block runtime installs (default True for workers)."),
    **kwargs,
):
    """Run as a distributed queue worker.

    Connects to a RabbitMQ broker and pulls workflow execution requests from a
    shared queue. Multiple workers can process jobs in parallel. Use with
    'comfyui serve --distributed-queue-frontend' on the frontend side.

    \b
    Example:
      comfyui worker --distributed-queue-connection-uri amqp://guest:guest@rabbitmq:5672/
      comfyui worker --distributed-queue-connection-uri amqp://... --guess-settings
    """
    params = _collect_params(locals(), kwargs)

    if params.get("otel_service_version") is None:
        from .. import __version__
        params["otel_service_version"] = __version__

    config = _build_config(params)
    config.distributed_queue_worker = True
    config.distributed_queue_frontend = False
    _parse_plugin_args(ctx, config)

    with _application_runtime(config) as run_driver:
        if not run_driver:
            return
        from ..entrypoints.worker import run_worker
        try:
            asyncio.run(run_worker(config))
        except KeyboardInterrupt:
            pass



_NODES_INDEX_URL = "https://nodes.appmana.com/simple/"


def _install_workflow_requirements(workflow_sources: list[str], *, dry_run: bool = False) -> list[str]:
    """Install missing custom node packages for the given workflows.

    Two-tier resolution: (1) try the pip facade index at nodes.appmana.com
    (which carries pre-built wheels for popular packages); (2) for any
    package that didn't install (typically because no wheel has been built
    yet), fall back to a direct ``pip install git+<repo_url>`` using the
    repository URL from comfy.org's node registry.
    """
    import shutil
    import subprocess

    from ..component_model.asyncio_files import load_workflow_json
    from ..component_model.workflow_dependencies import resolve_workflow_packages_versioned

    packages: set[str] = set()
    for source in workflow_sources:
        if source == "-":
            continue
        try:
            workflow = load_workflow_json(source)
        except Exception:
            continue
        for name, _ in resolve_workflow_packages_versioned(
            workflow, builtin_class_types=_load_available_class_types(),
        ):
            packages.add(name)

    if not packages:
        return []

    # Filter out already-installed packages
    from importlib.metadata import distributions
    installed: set[str] = set()
    for d in distributions():
        name = (d.metadata or {}).get("Name")
        if name:
            installed.add(name.lower().replace("_", "-"))
    missing = sorted(p for p in packages if p not in installed)
    if not missing:
        return []

    if dry_run:
        return missing

    uv = shutil.which("uv")
    if uv is None:
        logger.warning("uv not found, skipping custom node installation")
        return missing

    logger.info("Installing custom nodes: %s", ", ".join(missing))
    cmd = [uv, "pip", "install", "--python", sys.executable,
           "--extra-index-url", _NODES_INDEX_URL] + missing
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        return missing

    # Some packages don't have wheels on the pip facade — build a facade
    # wheel locally for each missing one (still goes through the same
    # _appmana_facade_<name> rename + stub-shim flow as a remote build,
    # so ComfyUI's node loader picks them up correctly), then pip install
    # the resulting wheels.
    from ..custom_node_facade.local_build import build_local_facade_wheel
    refreshed_installed: set[str] = set()
    for d in distributions():
        n = (d.metadata or {}).get("Name")
        if n:
            refreshed_installed.add(n.lower().replace("_", "-"))
    still_missing = [p for p in missing if p not in refreshed_installed]
    if not still_missing:
        return missing

    wheel_paths: list[str] = []
    unresolved: list[str] = []
    for pkg in still_missing:
        wheel = build_local_facade_wheel(pkg)
        if wheel:
            wheel_paths.append(wheel)
        else:
            unresolved.append(pkg)

    if wheel_paths:
        logger.info("Installing %d locally-built facade wheel(s): %s",
                    len(wheel_paths), ", ".join(os.path.basename(w) for w in wheel_paths))
        subprocess.run(
            [uv, "pip", "install", "--python", sys.executable] + wheel_paths,
            check=False,
        )

    if unresolved:
        logger.warning("Could not locate repo URL for: %s", ", ".join(unresolved))
    return missing


# Pre-built stable-ABI wheel indexes for binary CUDA extensions that we ship.
# Each provider serves wheels keyed by torch's CUDA backend tag (cu128, cu130);
# the same wheel works across PyTorch minor versions because of CPython's
# stable ABI + nvidia's per-CUDA-version ABI.
#   sageattention:  https://appmana.github.io/forks-sageattention-stable-abi/<cu_tag>/
#   nunchaku:       https://appmana.github.io/forks-nunchaku-stable-abi/<cu_tag>/
# Source repos: https://github.com/appmana/forks-sageattention-stable-abi
#               https://github.com/appmana/forks-nunchaku-stable-abi
_STABLE_ABI_INDEX_TEMPLATES: dict[str, str] = {
    "sageattention": "https://appmana.github.io/forks-sageattention-stable-abi/{cu_tag}/",
    "nunchaku":      "https://appmana.github.io/forks-nunchaku-stable-abi/{cu_tag}/",
}

# class_type -> stable-ABI extensions required to actually run that class.
# Augmented at runtime by widget-value heuristics (see _detect_stable_abi_needs).
_STABLE_ABI_TRIGGERS_BY_CLASS: dict[str, frozenset[str]] = {
    "PathchSageAttentionKJ": frozenset({"sageattention"}),  # ComfyUI-KJNodes (sic — upstream typo)
    "NunchakuFluxDiTLoader": frozenset({"nunchaku"}),
    "NunchakuQwenImageDiTLoader": frozenset({"nunchaku"}),
    "NunchakuFluxLoraLoader": frozenset({"nunchaku"}),
    "NunchakuFluxLoraStack": frozenset({"nunchaku"}),
    "NunchakuTextEncoderLoader": frozenset({"nunchaku"}),
    "NunchakuTextEncoderLoaderV2": frozenset({"nunchaku"}),
    "NunchakuDepthPreprocessor": frozenset({"nunchaku"}),
    "NunchakuPulidApply": frozenset({"nunchaku"}),
    "NunchakuPulidLoader": frozenset({"nunchaku"}),
}


def _detect_cu_tag() -> str | None:
    try:
        import torch
        cuda = torch.version.cuda
    except Exception:  # noqa: BLE001
        return None
    if not cuda:
        return None
    if cuda.startswith("12.8"):
        return "cu128"
    if cuda.startswith("13."):
        return "cu130"
    return None


def _detect_stable_abi_needs(workflow_sources: list[str]) -> set[str]:
    from ..component_model.asyncio_files import load_workflow_json
    needs: set[str] = set()
    for source in workflow_sources:
        if source == "-":
            continue
        try:
            workflow = load_workflow_json(source)
        except Exception:  # noqa: BLE001
            continue

        is_ui = isinstance(workflow, dict) and "nodes" in workflow and "links" in workflow
        if is_ui:
            for node in workflow.get("nodes") or []:
                ct = node.get("type") or ""
                if ct in _STABLE_ABI_TRIGGERS_BY_CLASS:
                    needs |= _STABLE_ABI_TRIGGERS_BY_CLASS[ct]
                # Widget-driven: WanVideoModelLoader / similar with attention_mode='sageattn*'
                wv = node.get("widgets_values")
                values = wv.values() if isinstance(wv, dict) else (wv if isinstance(wv, list) else [])
                if any(isinstance(v, str) and "sageattn" in v.lower() for v in values):
                    needs.add("sageattention")
        else:
            for nd in (workflow or {}).values():
                if not isinstance(nd, dict):
                    continue
                ct = nd.get("class_type") or ""
                if ct in _STABLE_ABI_TRIGGERS_BY_CLASS:
                    needs |= _STABLE_ABI_TRIGGERS_BY_CLASS[ct]
                inputs = nd.get("inputs") or {}
                if any(isinstance(v, str) and "sageattn" in v.lower() for v in inputs.values()):
                    needs.add("sageattention")
    return needs


def _install_workflow_stable_abi_extensions(workflow_sources: list[str]) -> None:
    """Install sageattention / nunchaku from our stable-ABI indexes when a
    workflow class type or widget value requires them. Mirrors the CI step
    in .github/workflows/test.yml; runs only when --all is passed."""
    import shutil
    import subprocess

    needs = _detect_stable_abi_needs(workflow_sources)
    if not needs:
        return

    cu_tag = _detect_cu_tag()
    if cu_tag is None:
        logger.info(
            "Skipping stable-ABI extensions (%s): torch CUDA backend not in {12.8, 13.x}",
            ", ".join(sorted(needs)),
        )
        return

    from importlib.metadata import distributions
    installed = {(d.metadata or {}).get("Name", "").lower() for d in distributions()}
    missing = sorted(p for p in needs if p not in installed)
    if not missing:
        return

    uv = shutil.which("uv")
    if uv is None:
        logger.warning("uv not found, skipping stable-ABI extension install")
        return

    for pkg in missing:
        index = _STABLE_ABI_INDEX_TEMPLATES[pkg].format(cu_tag=cu_tag)
        logger.info("Installing %s for %s from %s", pkg, cu_tag, index)
        cmd = [uv, "pip", "install", "--python", sys.executable, "--no-deps", pkg, "--index-url", index]
        subprocess.run(cmd, check=True)


def _auto_register_existing_models() -> tuple[list, list[str]]:
    """Search the OS index + walk fallbacks for model files, classify, and
    register parent dirs via ``folder_paths.add_model_folder_path``.

    Returns ``(classifications, scan_summary)`` for use by ``--dry-run``.
    """
    from .local_model_discovery import find_local_model_paths

    discovery = find_local_model_paths(register=True)
    return discovery.classifications, discovery.scan_summary


def _print_dry_run_plan(
    workflow_sources: list[str],
    classifications: list,
    scan_summary: list[str],
    planned_downloads: list[tuple[str, str]],
    custom_node_packages: list[str],
) -> None:
    """Pretty-print what ``--all`` would do."""
    import shutil
    from collections import defaultdict
    from . import folder_paths

    print("=" * 78)
    print(f"workflow:    {', '.join(workflow_sources)}")
    print()

    # Discovered: group by (kind, register_dir), show count
    classified = [c for c in classifications if c.kind and c.register_dir]
    by_kind_dir: dict[tuple[str, str], int] = defaultdict(int)
    for c in classified:
        by_kind_dir[(c.kind, c.register_dir)] += 1
    if by_kind_dir:
        print(f"discovered ({len(classified)} model files in {len(by_kind_dir)} dirs)")
        for (kind, d), n in sorted(by_kind_dir.items(), key=lambda x: (x[0][0], x[0][1])):
            print(f"  {kind:>22}  {d}  ({n} files)")
        print()
    hf_cache_count = sum(1 for c in classifications if c.is_hf_cache)
    unclassified = sum(1 for c in classifications if c.kind is None and not c.is_hf_cache)
    if hf_cache_count or unclassified:
        if hf_cache_count:
            print(f"  {'HF cache':>22}  {hf_cache_count} files (resolved per-file via huggingface_hub)")
        if unclassified:
            print(f"  {'unclassified':>22}  {unclassified} files (no kind inferred)")
        print()

    # Folder paths registered (deduplicated)
    if by_kind_dir:
        print("folder paths to register")
        for (kind, d), _n in sorted(by_kind_dir.items()):
            print(f"  --add-model-folder-path {kind}={d}")
        print()

    if custom_node_packages:
        print(f"custom nodes to install ({len(custom_node_packages)})")
        for package in custom_node_packages:
            print(f"  ↓ {package}")
        print()
    else:
        print("custom nodes to install: 0 (all required packages installed)")
        print()

    # Models to download
    if planned_downloads:
        print(f"models to download ({len(planned_downloads)})")
        for folder, name in planned_downloads:
            print(f"  ↓ {folder}/{name}")
        print()
    else:
        print("models to download: 0 (everything resolved locally)")
        print()

    # Scan summary
    print("scan summary")
    for line in scan_summary:
        print(f"  {line}")
    print()

    # Disk
    output_dir = str(folder_paths.get_output_directory())
    try:
        u = shutil.disk_usage(output_dir)
        print("disk")
        print(f"  {output_dir:<60} free {u.free / 2**30:.1f} GB")
    except OSError:
        pass


def _download_workflow_models(workflow_sources: list[str], dry_run: bool = False) -> list[tuple[str, str]]:
    """Download missing models for the given workflows.

    With ``dry_run=True``, no downloads are issued and the returned list
    contains the ``(folder_name, filename)`` pairs that would have been
    downloaded. With ``dry_run=False`` (the default), downloads run and the
    list returned is the same — what was downloaded.
    """
    from ..component_model.asyncio_files import load_workflow_json
    from ..component_model.workflow_convert import is_ui_workflow, convert_ui_to_api
    from ..component_model.workflow_note_models import extract_models_from_notes
    from ..model_downloader import _known_models_db, add_known_models, get_or_download, canonicalize_path
    from .. import civitai_model_cache
    from . import folder_paths

    # Hydrate the Civitai cache for each workflow's author so community
    # checkpoints/loras uploaded by the same person can resolve at lookup time.
    civitai_model_cache.init_civitai_model_cache()
    for source in workflow_sources:
        if source != "-":
            civitai_model_cache.prefetch_civitai_models_for_workflow_uri(source)

    # Mine each workflow's Note / MarkdownNote nodes for model download URLs
    # (workflow authors document the exact HF / GitHub / Civitai URLs the
    # workflow needs in markdown notes; treat those as runtime-registered
    # KnownDownloadables so lookups by filename find them). Each NoteModel
    # is upgraded to a HuggingFile / FsspecFile(civitai) / UrlFile based on
    # its host so downloads inherit the right cache + auth path.
    for source in workflow_sources:
        if source == "-":
            continue
        try:
            note_workflow = load_workflow_json(source)
        except Exception:  # noqa: BLE001
            continue
        for nm in extract_models_from_notes(note_workflow):
            primary = nm.to_downloadable()
            add_known_models(nm.folder or "checkpoints", primary)
            for alt in nm.alternate_names:
                # Re-emit alternates as their own downloadables so basename
                # lookups for the URL-derived filename also work.
                alt_nm = type(nm)(filename=alt, url=nm.url, folder=nm.folder)
                add_known_models(nm.folder or "checkpoints", alt_nm.to_downloadable())
            logger.debug("Note URL: %s/%s -> %s (%s)",
                         nm.folder, nm.filename, nm.url, type(primary).__name__)

    filename_index: dict[str, list[tuple[str, object]]] = {}
    for db in _known_models_db:
        for folder_name in db.folder_names:
            for item in db:
                for name in [str(item), item.filename, item.save_with_filename] + list(item.alternate_filenames):
                    key = canonicalize_path(name)
                    if key:
                        filename_index.setdefault(key, []).append((folder_name, item))

    planned: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source in workflow_sources:
        if source == "-":
            continue
        try:
            workflow = load_workflow_json(source)
        except Exception:
            continue
        if is_ui_workflow(workflow):
            workflow = convert_ui_to_api(workflow)
        for node_data in workflow.values():
            if not isinstance(node_data, dict):
                continue
            for value in (node_data.get("inputs") or {}).values():
                if not isinstance(value, str) or not value:
                    continue
                key = canonicalize_path(value)
                if key in seen:
                    continue
                seen.add(key)
                matches = filename_index.get(key)
                if matches:
                    folder_name = matches[0][0]
                    known_file = matches[0][1]
                    found = (
                        get_or_download(folder_name, value, known_files=[known_file], offline=True)
                        if dry_run
                        else folder_paths.get_full_path(folder_name, value)
                    )
                    if not found:
                        planned.append((folder_name, value))
                        if not dry_run:
                            logger.info("Downloading %s/%s", folder_name, value)
                            get_or_download(folder_name, value, known_files=[known_file])
    return planned


class _RunWorkflowCommand(typer.core.TyperCommand):
    @staticmethod
    def _option_value_arity_by_name(ctx: click.Context | None) -> dict[str, int]:
        if ctx is None:
            return {}
        arity: dict[str, int] = {}
        for param in ctx.command.params:
            # Typer 0.26 vendors Click, so its TyperOption no longer inherits
            # from the separately installed click.Option class.  Use the
            # stable option attributes shared by both implementations.
            if not hasattr(param, "is_flag") or not hasattr(param, "opts"):
                continue
            consumes = 0 if param.is_flag else max(1, param.nargs)
            for opt in (*param.opts, *param.secondary_opts):
                arity[opt] = consumes
        return arity

    @classmethod
    def _known_option_names(cls, ctx: click.Context | None) -> set[str]:
        return set(cls._option_value_arity_by_name(ctx))

    @staticmethod
    def _set_expr_for_workflow_param(param, raw_value: str) -> str:
        return f"{param.node_id}.inputs.{param.widget_name}={raw_value}"

    @classmethod
    def _workflow_flag_for_param(cls, param, ctx: click.Context | None = None) -> str | None:
        if not param.flag_name:
            return None
        flag = f"--{param.flag_name}"
        if flag in cls._known_option_names(ctx):
            return f"--x-{param.flag_name}"
        return flag

    @classmethod
    def _rewrite_workflow_param_args(cls, args: list[str], ctx: click.Context | None = None) -> list[str]:
        ref = cls._extract_workflow_ref(args, ctx=ctx)
        if not ref:
            return args
        try:
            workflow_params = _discover_from_ref(ref)
        except Exception:
            return args
        by_flag = {
            flag: p
            for p in workflow_params
            if (flag := cls._workflow_flag_for_param(p, ctx=ctx)) is not None
        }
        by_negative_boolean_flag = {
            f"--no-{flag.removeprefix('--')}": p
            for p in workflow_params
            if p.type == "BOOLEAN" and (flag := cls._workflow_flag_for_param(p, ctx=ctx)) is not None
        }
        if not by_flag:
            return args

        rewritten: list[str] = []
        i = 0
        n = len(args)
        while i < n:
            arg = args[i]
            if arg == "--":
                rewritten.extend(args[i:])
                break

            option, eq, value = arg.partition("=")
            negative_boolean_param = by_negative_boolean_flag.get(option)
            if negative_boolean_param is not None:
                if eq:
                    raise click.UsageError(f"Option {option!r} does not take a value.")
                rewritten.extend(["--set", cls._set_expr_for_workflow_param(negative_boolean_param, "false")])
                i += 1
                continue

            param = by_flag.get(option)
            if param is None:
                rewritten.append(arg)
                i += 1
                continue

            if eq:
                rewritten.extend(["--set", cls._set_expr_for_workflow_param(param, value)])
                i += 1
                continue

            if param.type == "BOOLEAN" and (i + 1 >= n or args[i + 1].startswith("-")):
                rewritten.extend(["--set", cls._set_expr_for_workflow_param(param, "true")])
                i += 1
                continue

            if i + 1 >= n:
                raise click.UsageError(f"Option {arg!r} requires a value.")
            rewritten.extend(["--set", cls._set_expr_for_workflow_param(param, args[i + 1])])
            i += 2
        return rewritten

    @classmethod
    def _extract_workflow_ref(cls, args: list[str], ctx: click.Context | None = None) -> str | None:
        arity_by_name = cls._option_value_arity_by_name(ctx)
        i = 0
        n = len(args)
        while i < n:
            a = args[i]
            if a in ("--help", "-h"):
                i += 1
                continue
            if a == "--":
                i += 1
                continue
            if a == "-":
                return "-"
            if a.startswith("-") and len(a) > 1:
                option = a.partition("=")[0]
                if "=" in a:
                    i += 1
                else:
                    i += 1 + arity_by_name.get(option, 1)
                continue
            return a
        return None

    @staticmethod
    def _suggest_addons(params) -> list[tuple[str, str]]:
        from ..component_model.prompt_utils import _MODEL_PRODUCER_CLASS_TYPES
        if not any(p.class_type in _MODEL_PRODUCER_CLASS_TYPES for p in params):
            return []
        return [
            ("--add-lora <name>[:strength]", "Splice a LoRA after the model loader"),
            ("--compile", "Wrap the diffusion transformer in torch.compile"),
        ]

    @classmethod
    def _example_invocation(cls, ref: str, params, ctx: click.Context | None = None) -> str | None:
        from ..entrypoints.workflow_params import TIER_HEADLINE
        command_path = ctx.command_path if ctx is not None else "comfyui run-workflow"
        target = next(
            (p for p in params if p.tier == TIER_HEADLINE and p.type in ("STRING", "INT", "FLOAT")),
            next((p for p in params if p.tier == TIER_HEADLINE), None),
        )
        if target is None:
            return None
        flag = cls._workflow_flag_for_param(target, ctx=ctx)
        if flag:
            return f"{command_path} {ref} {flag} <value>"
        return f"{command_path} {ref} --set {target.node_id}.inputs.{target.widget_name}=<value>"

    def parse_args(self, ctx, args):
        if "--help" in args or "-h" in args:
            ref = self._extract_workflow_ref(args, ctx=ctx)
            if ref:
                try:
                    params = _discover_from_ref(ref)
                    raw_workflow = _DISCOVER_RAW_CACHE.get(ref)
                except Exception as exc:  # noqa: BLE001
                    typer.echo(f"(Could not discover workflow parameters: {exc})", err=True)
                    if ref.startswith(("civitai://v/", "civitai-red://v/")):
                        typer.echo(
                            "\n  Hint: civitai://v/<id> is a *version* id — usually a LoRA, "
                            "checkpoint, or other asset, not a workflow. Pass it to "
                            "--add-lora / --checkpoint instead, e.g.:\n"
                            f"    comfyui run-workflow <workflow> --add-lora {ref}:0.8\n"
                            "  Or use `comfyui models search …` to find the right URI for the asset kind.",
                            err=True,
                        )
                    typer.echo()
                    typer.echo(self.get_help(ctx))
                    ctx.exit()
                    return
                self._render_workflow_help(ctx, ref, params, raw_workflow)
                ctx.exit()
            typer.echo(self.get_help(ctx))
            ctx.exit()
        return super().parse_args(ctx, self._rewrite_workflow_param_args(args, ctx=ctx))

    def _render_workflow_help(self, ctx, ref: str, params, raw_workflow: dict | None = None) -> None:
        from rich import box
        from rich.console import Console, Group
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from ..entrypoints.workflow_params import (
            TIER_ADVANCED, TIER_COMMON, TIER_HEADLINE, assign_ui_groups, count_input_slots,
        )

        console = Console()

        def _flag_for(p) -> str:
            flag = self._workflow_flag_for_param(p, ctx=ctx)
            if flag:
                return flag
            return f"--set {p.node_id}.{p.widget_name}"

        def _make_section(rows: list, heading: str) -> Panel | None:
            if not rows:
                return None
            t = Table.grid(padding=(0, 2), expand=True)
            t.add_column("flag", style="bold cyan", no_wrap=True)
            t.add_column("type", style="green", no_wrap=True)
            t.add_column("value", overflow="fold")
            for p in rows:
                value_repr = repr(p.value)
                if len(value_repr) > 60:
                    value_repr = value_repr[:57] + "..."
                t.add_row(_flag_for(p), p.type, Text(value_repr, style="dim"))
            return Panel(t, title=heading, title_align="left", border_style="cyan", box=box.ROUNDED)

        headline_rows = [p for p in params if p.tier == TIER_HEADLINE]
        common_rows = [p for p in params if p.tier == TIER_COMMON]
        n_advanced = sum(1 for p in params if p.tier == TIER_ADVANCED)

        rendered: list = []

        # Author notes: render Note / MarkdownNote contents as the first
        # section(s) so the workflow's own documentation is the first thing
        # users read.
        if raw_workflow is not None and "nodes" in (raw_workflow or {}) and "links" in raw_workflow:
            for node in raw_workflow.get("nodes") or []:
                ntype = node.get("type")
                if ntype not in ("Note", "MarkdownNote"):
                    continue
                wv = node.get("widgets_values")
                text = ""
                if isinstance(wv, list) and wv:
                    text = wv[0] if isinstance(wv[0], str) else ""
                elif isinstance(wv, dict):
                    text = next((v for v in wv.values() if isinstance(v, str)), "")
                text = (text or "").strip()
                if not text:
                    continue
                title = node.get("title") or ("Markdown note" if ntype == "MarkdownNote" else "Note")
                # MarkdownNote uses GitHub-flavored markdown; wrap in Markdown
                # so links and headings render properly. Plain Note widgets
                # are typically free text — keep as-is, but auto-promote to
                # Markdown if the text looks markdown-shaped (has a link or
                # heading) so workflows that store URLs in plain Notes still
                # render clickably.
                _looks_markdown = ntype == "MarkdownNote" or any(
                    marker in text for marker in ("](http", "\n# ", "\n## ", "\n* ", "\n- [")
                ) or text.lstrip().startswith(("# ", "## ", "- [", "* "))
                if _looks_markdown:
                    # rich.markdown.Markdown emits OSC 8 hyperlinks but hides
                    # the URL — for non-hyperlink terminals, log files, and
                    # piping to less, that loses the actual URL. Inline each
                    # URL after its link text so users can always copy it.
                    visible_text = re.sub(
                        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
                        r"[\1](\2) <\2>",
                        text,
                    )
                    content = Markdown(visible_text)
                else:
                    content = text
                rendered.append(Panel(
                    content,
                    title=title,
                    title_align="left",
                    border_style="yellow",
                    box=box.ROUNDED,
                ))

        # Inputs summary: how many images/videos/audios this workflow accepts.
        if raw_workflow is not None:
            counts = count_input_slots(raw_workflow)
            input_lines = []
            for kind, flag in (("images", "--image"), ("videos", "--video"), ("audios", "--audio")):
                active, total = counts[kind]
                if total == 0:
                    continue
                qualifier = f"up to {total}" if total > 1 else "1"
                bypass_note = f" ({total - active} optional)" if total > active else ""
                input_lines.append(
                    f"  [bold cyan]{flag}[/bold cyan]   {qualifier} {kind}{bypass_note}"
                )
            if input_lines:
                rendered.append(Panel(
                    "\n".join(input_lines),
                    title="Inputs",
                    title_align="left",
                    border_style="cyan",
                    box=box.ROUNDED,
                ))

        # If the UI workflow has author-defined groups, render headline+common
        # together split per group (matching the user's mental model in the
        # editor). Otherwise fall back to flat headline/common panels.
        groups_map = assign_ui_groups(raw_workflow or {}, headline_rows + common_rows) if raw_workflow else {}
        if any(title for title in groups_map.keys()) and len(groups_map) > 1:
            for title, rows in groups_map.items():
                if not rows:
                    continue
                rows_sorted = sorted(rows, key=lambda p: (p.tier, p.node_id, p.widget_name))
                heading = title or "Other parameters"
                if title == f"Workflow parameters — {ref}":
                    heading = title
                panel = _make_section(rows_sorted, heading or f"Workflow parameters — {ref}")
                if panel:
                    rendered.append(panel)
        else:
            headline_panel = _make_section(headline_rows, f"Workflow parameters — {ref}")
            if headline_panel:
                rendered.append(headline_panel)
            common_panel = _make_section(common_rows, "Common parameters")
            if common_panel:
                rendered.append(common_panel)

        # "Suggested for this workflow" panel shows applicable add-ons.
        addons = self._suggest_addons(params)
        if addons:
            t = Table.grid(padding=(0, 2), expand=True)
            t.add_column(style="bold cyan", no_wrap=True)
            t.add_column(overflow="fold")
            for flag, desc in addons:
                t.add_row(flag, desc)
            rendered.append(Panel(t, title="Suggested for this workflow", title_align="left", border_style="cyan", box=box.ROUNDED))

        # Footer with hints + example
        footer_lines = []
        if not headline_rows and not common_rows:
            footer_lines.append("(no parameters detected)")
        if n_advanced:
            footer_lines.append(f"[dim]{n_advanced} advanced params hidden — see them with: comfyui workflows params {ref} --all[/dim]")
        example = self._example_invocation(ref, params, ctx=ctx)
        if example:
            footer_lines.append(f"[bold]Example:[/bold]")
            footer_lines.append(f"  [dim]{example}[/dim]")
        if footer_lines:
            rendered.append(Panel("\n".join(footer_lines), border_style="cyan", box=box.ROUNDED))

        generic_help = self._get_help_text(ctx)
        before_params, after_params = self._split_help_before_default_params(generic_help)
        typer.echo(before_params.rstrip())
        if rendered:
            console.print(Group(*rendered))
        if after_params:
            typer.echo(after_params.lstrip("\n"))

    @staticmethod
    def _get_help_text(ctx: click.Context) -> str:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            help_text = ctx.command.get_help(ctx)
        captured = stdout.getvalue()
        return help_text or captured

    @staticmethod
    def _split_help_before_default_params(help_text: str) -> tuple[str, str]:
        """Split rich Typer help before the standard Arguments/Options panels.

        Legacy Windows consoles make rich substitute square (or ASCII) box
        characters for the rounded ones, so match all border styles."""
        plain_help = re.sub(r"\x1b\[[0-9;]*m", "", help_text)
        match = re.search(r"(?m)^\s*(?:[╭┌]─+|\+-+) (Arguments|Options) ", plain_help)
        if match is None:
            return help_text, ""
        split_at = _plain_index_to_raw_index(help_text, match.start())
        return help_text[:split_at], help_text[split_at:]


def _plain_index_to_raw_index(text: str, plain_index: int) -> int:
    plain_pos = 0
    raw_pos = 0
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    while raw_pos < len(text) and plain_pos < plain_index:
        match = ansi_re.match(text, raw_pos)
        if match is not None:
            raw_pos = match.end()
            continue
        raw_pos += 1
        plain_pos += 1
    return raw_pos


def _run_workflow_cli(config, *, all: bool = False, dry_run: bool = False) -> None:
    from ..distributed.config import resolve_distributed_configuration
    distributed = resolve_distributed_configuration(config)
    if distributed.externally_launched and (all or dry_run):
        raise typer.BadParameter("--all and --dry-run must run before an external distributed launch")

    if all:
        # Package resolution can import modules that read args at import time.
        _set_config_context(config)
        custom_node_packages = _install_workflow_requirements(config.workflows, dry_run=dry_run)
    else:
        custom_node_packages = []

    with _application_runtime(config) as run_driver:
        if not run_driver:
            return

        from ..component_model.entrypoints_common import configure_application_paths
        configure_application_paths(config)

        if all:
            # After torch is available we know cu128 vs cu130, so do this here
            # rather than alongside _install_workflow_requirements.
            if not dry_run:
                _install_workflow_stable_abi_extensions(config.workflows)
            classifications, scan_summary = _auto_register_existing_models()
            planned = _download_workflow_models(config.workflows, dry_run=dry_run)
            if dry_run:
                _print_dry_run_plan(config.workflows, classifications, scan_summary, planned, custom_node_packages)
                return

        from ..execution_context import context_configuration
        from ..nodes.package import import_all_nodes_in_workspace
        with context_configuration(config):
            import_all_nodes_in_workspace(raise_on_failure=False)

        from ..entrypoints.workflow import run_workflows
        try:
            asyncio.run(run_workflows(config.workflows, configuration=config))
        except KeyboardInterrupt:
            pass


def _run_workflow_command(
    ctx: typer.Context,
    workflows: list[str],
    all: bool,
    dry_run: bool,
    local_vars: dict,
    kwargs: dict,
) -> None:
    _warn_unknown_cli_args(ctx.args)
    workflows = _remove_unknown_option_args_from_workflows(workflows)
    _all = all or dry_run
    _dry_run = dry_run
    params = _collect_params({**local_vars, "workflows": workflows}, kwargs)
    params.pop("all", None)
    params.pop("_all", None)
    params.pop("dry_run", None)
    params.pop("_dry_run", None)

    if params.get("output") is not None:
        params["output_directory"] = params["output"]

    if params.get("otel_service_version") is None:
        from .. import __version__
        params["otel_service_version"] = __version__

    config = _build_config(params)
    _run_workflow_cli(config, all=_all, dry_run=_dry_run)


@app.command(
    name="run-workflow",
    context_settings=_COMFYUI_ENV,
    rich_help_panel="Workflows",
    cls=_RunWorkflowCommand,
)
@_with_options(_ALL_SHARED_OPTS, _WORKFLOW_OVERRIDE_OPTS)
def run_workflow(
    ctx: typer.Context,
    workflows: list[str] = _RUN_WORKFLOWS_ARGUMENT,
    all: bool = _RUN_ALL_OPTION,
    dry_run: bool = _RUN_DRY_RUN_OPTION,
    disable_progress: bool = _RUN_DISABLE_PROGRESS_OPTION,
    block_runtime_package_installation: bool = _RUN_BLOCK_RUNTIME_PACKAGE_INSTALLATION_OPTION,
    **kwargs,
):
    """Execute workflow(s) and exit.

    Run one or more workflows without starting the web server. Results are
    printed as JSON to stdout; logs go to stderr. Accepts file paths, URIs
    (https://, hf://, s3://), literal JSON strings, or '-' for stdin.

    \b
    NEXT STEP — INSPECT YOUR WORKFLOW:
    Re-run --help with the workflow path/URL to see this workflow's actual
    parameters (the prompts, steps, seed, LoRA strengths, etc. that this
    specific workflow exposes), plus any author notes embedded in the file:
      comfyui run-workflow WORKFLOW_FILE_OR_URL --help
    The same is true of `comfyui workflows run`. Without a workflow argument,
    --help only shows the *generic* options below — it cannot know which
    parameters your particular workflow takes until you give it one.

    \b
    With --all, automatically install missing custom nodes from
    nodes.appmana.com and download missing models before running:
      comfyui run-workflow workflow.json --all --guess-settings

    \b
    Override workflow parameters inline:
      --prompt "a cat"     Replace the positive text prompt
      --steps 20           Override sampling steps
      --seed 42            Set a fixed seed
      --set 5.inputs.denoise=0.8
                           Override any node input by ID

    \b
    Examples:
      comfyui run-workflow workflow.json --help        # inspect the workflow
      comfyui run-workflow workflow.json --all --guess-settings
      comfyui run-workflow workflow.json --prompt "a sunset" --steps 20 --seed 42
      comfyui run-workflow https://example.com/workflow.json --all
      cat workflow.json | comfyui run-workflow -
      comfyui run-workflow workflow.json --novram --fast cublas_ops
    """
    _run_workflow_command(ctx, workflows, all, dry_run, locals(), kwargs)



_DISCOVER_CACHE: dict[str, list] = {}
_DISCOVER_RAW_CACHE: dict[str, dict] = {}


def _discover_from_ref(ref: str) -> list:
    cached = _DISCOVER_CACHE.get(ref)
    if cached is not None:
        return cached

    import asyncio
    from ..component_model.asyncio_files import stream_json_objects
    from ..entrypoints.workflow import _resolve_workflow
    from ..entrypoints.workflow_params import discover

    async def _load() -> dict:
        async for obj in stream_json_objects(_resolve_workflow(ref)):
            return obj
        raise SystemExit("no workflow object found")

    raw = asyncio.run(_load())
    from ..nodes.package_typing import ExportedNodes
    params = discover(raw, node_mappings=ExportedNodes())
    _DISCOVER_CACHE[ref] = params
    _DISCOVER_RAW_CACHE[ref] = raw
    return params


@app.command(name="create-directories", context_settings=_COMFYUI_ENV, rich_help_panel="Environment", hidden=True)
def create_directories_cmd(
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    output_directory: Optional[str] = typer.Option(None, "--output-directory", help="Output directory."),
    input_directory: Optional[str] = typer.Option(None, "--input-directory", help="Input directory."),
    temp_directory: Optional[str] = typer.Option(None, "--temp-directory", help="Temp directory."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
    logging_level: str = typer.Option("INFO", "--logging-level", help="Log level."),
):
    """Create default model/input/output/temp directories and exit."""
    from ..component_model.setup import setup_pre_torch

    params = {k: v for k, v in locals().items() if k != "ctx"}
    for key in ("base_paths", "extra_model_paths_config"):
        if params.get(key) is None:
            params[key] = []

    config = _build_config(params)

    setup_pre_torch(config)
    _set_config_context(config)

    from ..execution_context import context_configuration
    from .folder_paths import create_directories
    from ..nodes.package import import_all_nodes_in_workspace
    with context_configuration(config):
        import_all_nodes_in_workspace(raise_on_failure=False)
        create_directories()



@app.command(name="list-workflow-templates", context_settings=_COMFYUI_ENV, hidden=True)
def list_workflow_templates(
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    template_dir: Optional[list[str]] = typer.Option(None, "--template-dir", help="Extra directories to scan."),
    convert_to_api: bool = typer.Option(False, "--convert-to-api/--no-convert-to-api", help="Convert UI workflows to API format (boots node system)."),
    all_templates: bool = typer.Option(False, "-a", "--all/--no-all", help="Include API-key-requiring templates."),
):
    """List available workflow templates."""
    import sys
    from .workflow_templates import list_templates
    interactive = sys.stdout.isatty() and format == "table"
    list_templates(
        format=format,
        extra_dirs=template_dir or [],
        convert=convert_to_api,
        show_all=all_templates,
        interactive=interactive,
    )


@app.command(name="list-models", context_settings=_COMFYUI_ENV, hidden=True)
def list_models_cmd(
    format: str = typer.Option("table", "--format", help="Output format: table or json."),
    folder: Optional[str] = typer.Option(None, "--folder", help="Filter by model folder (checkpoints, loras, vae, etc)."),
    no_manager: bool = typer.Option(False, "--no-manager", help="Exclude comfyui_manager models."),
    check_exists: bool = typer.Option(False, "--check-exists/--no-check-exists", help="Check if models exist locally (requires path initialization)."),
    cwd: Optional[str] = typer.Option(None, "-w", "--cwd", help="Working directory."),
    base_directory: Optional[str] = typer.Option(None, "--base-directory", help="Base directory."),
    base_paths: Optional[list[str]] = typer.Option(None, "--base-paths", help="Additional base paths."),
    extra_model_paths_config: Optional[list[str]] = typer.Option(None, "--extra-model-paths-config", help="Extra model paths config."),
    logging_level: str = typer.Option("INFO", "--logging-level", click_type=click.Choice(["DEBUG", "DETAIL", "INFO", "WARNING", "ERROR", "CRITICAL"]), help="Logging level."),
):
    """List known downloadable models."""
    if check_exists:
        from ..component_model.setup import setup_pre_torch

        params = {}
        if cwd is not None:
            params["cwd"] = cwd
        if base_directory is not None:
            params["base_directory"] = base_directory
        if base_paths is not None:
            expanded = []
            for v in base_paths:
                for piece in str(v).split(","):
                    piece = piece.strip()
                    if piece:
                        expanded.append(piece)
            params["base_paths"] = expanded
        else:
            params["base_paths"] = []
        if extra_model_paths_config is not None:
            params["extra_model_paths_config"] = list(extra_model_paths_config)
        else:
            params["extra_model_paths_config"] = []
        params["logging_level"] = logging_level

        config = _build_config(params)
        setup_pre_torch(config)
        _set_config_context(config)

        from ..execution_context import context_configuration
        from ..nodes.package import import_all_nodes_in_workspace
        with context_configuration(config):
            import_all_nodes_in_workspace(raise_on_failure=False)

    from .list_models import list_models
    list_models(format=format, folder=folder, include_manager=not no_manager, check_exists=check_exists)



@app.command(name="serve-pip", rich_help_panel="Package Index")
@_with_options(_LOGGING_OPTS, _PIP_FACADE_OPTS)
def serve_pip(
    listen: str = typer.Option("127.0.0.1", "-H", "--listen", help="Specify the IP address to listen on."),
    port: int = typer.Option(8190, help="Set the listen port."),
    **kwargs,
):
    """Serve a PEP 503 simple index for facade-packaged custom nodes.

    Builds pip-installable wheels on demand from ComfyUI ecosystem custom nodes.
    Combines ComfyUI-Manager's registry with the Comfy Node Registry (CNR) API,
    injects dependency metadata, and serves wheels through a standard pip index.

    The public instance is at https://nodes.appmana.com. Self-host for air-gapped
    environments or to control which nodes are available.

    \b
    Wheel cache: generated wheels are cached to avoid rebuilding. Use any writable
    fsspec URI (local path, s3://, etc.):
      --pip-facade-cache-prefix /var/cache/comfyui
      --pip-facade-cache-prefix s3://bucket/prefix

    \b
    Examples:
      comfyui serve-pip --listen 0.0.0.0 --port 8190
      comfyui serve-pip --pip-facade-only-known-nodes
      uv pip install --extra-index-url http://localhost:8190/simple/ comfyui-ltxvideo
    """
    from ..component_model.setup import setup_pre_torch, setup_post_torch

    params = _collect_params(locals(), kwargs)
    config = _build_config(params)
    # The pip facade is a headless, CPU-only package index: it never loads a
    # model. Force CPU and disable GPU/dynamic-VRAM initialization so it runs
    # cleanly on CPU-only hosts (dynamic VRAM / comfy-aimdo is CUDA-only and
    # would otherwise crash at startup).
    config.cpu = True
    config.disable_dynamic_vram = True
    setup_pre_torch(config)
    _set_config_context(config)
    setup_post_torch(config)

    from .serve_pip import run_serve_pip
    run_serve_pip(config)


@app.command(name="snapshot-pip-registry", rich_help_panel="Package Index")
@_with_options(_LOGGING_OPTS, _PIP_FACADE_OPTS, _PIP_FACADE_SNAPSHOT_OPTS)
def snapshot_pip_registry(**kwargs):
    """Snapshot the resolved facade registry into a compact SQLite artifact."""
    from ..component_model.setup import setup_pre_torch

    params = _collect_params(locals(), kwargs)
    config = _build_config(params)
    setup_pre_torch(config)
    _set_config_context(config)

    from .snapshot_pip_registry import run_snapshot_pip_registry
    run_snapshot_pip_registry(config)


def _load_core_class_types() -> frozenset[str]:
    from ..nodes.package import _import_and_enumerate_nodes_in_module
    from ..nodes.package_typing import ExportedNodes
    from functools import reduce
    from ..nodes import base_nodes
    from comfy_extras import nodes as comfy_extras_nodes
    import comfy_api_nodes
    core_nodes = reduce(
        lambda x, y: x.update(y),
        map(_import_and_enumerate_nodes_in_module, [base_nodes, comfy_extras_nodes, comfy_api_nodes]),
        ExportedNodes(),
    )
    return frozenset(core_nodes.NODE_CLASS_MAPPINGS.keys())


def _load_available_class_types() -> frozenset[str]:
    from ..nodes.package import import_all_nodes_in_workspace
    nodes = import_all_nodes_in_workspace(raise_on_failure=False)
    return frozenset(nodes.NODE_CLASS_MAPPINGS.keys())


@app.command(name="workflow-requirements", rich_help_panel="Workflows", hidden=True)
def workflow_requirements(
    workflow_file: str = typer.Argument(..., help="Workflow file, URI, template name, or literal JSON."),
    format: str = typer.Option("requirements_txt", "--format", "-f", help="Output format: requirements_txt, requirements_txt_versioned, requirements_txt_locked"),
    snapshot_uri: Optional[str] = typer.Option(None, "--pip-facade-snapshot-uri", help="Facade registry snapshot URI. Defaults to the bundled snapshot."),
):
    """Print custom node packages required by a workflow in pip requirements format."""
    from ..component_model.asyncio_files import load_workflow_json
    from ..component_model.workflow_dependencies import resolve_workflow_packages_versioned
    from ..entrypoints.workflow import _resolve_workflow

    workflow = load_workflow_json(_resolve_workflow(workflow_file))
    packages = resolve_workflow_packages_versioned(
        workflow,
        snapshot_uri=snapshot_uri,
        builtin_class_types=_load_core_class_types(),
    )

    for name, version in packages:
        if format == "requirements_txt_versioned" and version:
            typer.echo(f"{name}>={version}")
        elif format == "requirements_txt_locked" and version:
            typer.echo(f"{name}=={version}")
        else:
            typer.echo(name)


@app.command(name="start", rich_help_panel="Daemon")
def start(ctx: typer.Context):
    """Start ComfyUI as a background daemon with auto-detected settings.

    Equivalent to 'comfyui serve --daemon --guess-settings'. Detects your GPU,
    RAM, and available acceleration libraries, then starts ComfyUI in the
    background. Use 'comfyui stop' to shut it down and 'comfyui logs -f' to
    watch output.

    \b
    Examples:
      comfyui start
      comfyui logs -f
      comfyui stop
    """
    raise typer.Exit(subprocess.run(
        [sys.argv[0], "serve", "--daemon", "--guess-settings"],
        check=False,
    ).returncode)


@app.command(name="stop", rich_help_panel="Daemon")
def stop(
    server: Optional[str] = typer.Option(None, "--server", envvar="COMFYUI_SERVER", help="Server URL for HTTP fallback."),
    pid_file: Optional[str] = typer.Option(None, "--pid-file", help="PID file path (default: ~/.comfyui/comfyui.pid)."),
):
    """Stop the ComfyUI daemon.

    Reads the PID from ~/.comfyui/comfyui.pid and sends SIGTERM. Falls back to
    the HTTP /interrupt endpoint if the PID file is missing.
    """
    from .daemon import stop_daemon, default_pid_file

    pf = pid_file or default_pid_file()
    if stop_daemon(pf):
        typer.echo("ComfyUI daemon stopped.")
        return
    try:
        from .server_connection import post_json
        asyncio.run(post_json(server, "/interrupt"))
        typer.echo("Sent interrupt to server.")
    except Exception as exc:
        typer.echo(f"Could not stop daemon: {exc}", err=True)
        raise typer.Exit(1)


@app.command(name="logs", rich_help_panel="Daemon")
def logs(
    follow: bool = typer.Option(False, "-f", "--follow/--no-follow", help="Follow log output."),
    server: Optional[str] = typer.Option(None, "--server", envvar="COMFYUI_SERVER", help="Server URL."),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Log file path (default: ~/.comfyui/comfyui.log)."),
):
    """Tail server logs.

    Without -f, prints the current log contents. With -f, follows the log file
    in real time (like tail -f). Tries the HTTP /internal/logs/raw endpoint
    first, falls back to reading ~/.comfyui/comfyui.log directly.

    \b
    Examples:
      comfyui logs
      comfyui logs -f
      comfyui logs --server http://remote:8188
    """
    from .daemon import default_log_file

    lf = log_file or default_log_file()
    log_path = Path(lf)

    if not follow:
        try:
            from .server_connection import fetch_text
            text = asyncio.run(fetch_text(server, "/internal/logs/raw"))
            typer.echo(text)
            return
        except Exception:
            pass
        if log_path.exists():
            typer.echo(log_path.read_text())
        else:
            typer.echo("No logs found.", err=True)
        return

    import time
    if not log_path.exists():
        typer.echo(f"Waiting for {lf}...", err=True)
        while not log_path.exists():
            time.sleep(0.5)
    with open(log_path) as f:
        f.seek(0, 2)
        try:
            while True:
                line = f.readline()
                if line:
                    typer.echo(line, nl=False)
                else:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass


def _register_sub_apps():
    from .sub_models import models_app
    from .sub_workflows import workflows_app
    from .sub_nodes import nodes_app
    from .sub_jobs import jobs_app
    from .sub_env import env_app

    app.add_typer(models_app, name="models", help="Manage models.", rich_help_panel="Workflows")
    app.add_typer(workflows_app, name="workflows", help="Run and manage workflows.", rich_help_panel="Workflows")
    app.add_typer(nodes_app, name="nodes", help="Inspect installed nodes.", rich_help_panel="Workflows")
    app.add_typer(jobs_app, name="jobs", help="List and cancel server jobs.", rich_help_panel="Daemon")
    app.add_typer(env_app, name="env", help="Environment and diagnostics.", rich_help_panel="Environment")


_KNOWN_COMMANDS = frozenset({
    "serve", "serve-pip", "worker", "start", "stop", "logs",
    "models", "workflows", "nodes", "jobs", "env",
    "run-workflow", "workflow-requirements", "list-workflow-templates",
    "list-models", "create-directories",
    "snapshot-pip-registry",
})


_BARE_FLAG_DEFAULTS: dict[str, str] = {
    "--listen": "0.0.0.0,::",
    "--enable-cors-header": "*",
    "--enable-cors": "*",
    "--cache-ram": "-1.0",
}
"""Flags that argparse accepted with ``nargs='?'``.

Typer/Click options always consume an argument, so a bare ``--listen``
(without a value) fails with *"Option '--listen' requires an argument"*.
We expand these in ``sys.argv`` before Click sees them.
"""


def _expand_bare_flags(argv: list[str]) -> list[str]:
    """Insert default values for flags listed in :data:`_BARE_FLAG_DEFAULTS`
    when they appear without a following value (end-of-args or next token
    starts with ``-``).
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in _BARE_FLAG_DEFAULTS:
            next_token = argv[i + 1] if i + 1 < len(argv) else None
            if next_token is None or next_token.startswith("-"):
                out.append(token)
                out.append(_BARE_FLAG_DEFAULTS[token])
                i += 1
                continue
        out.append(token)
        i += 1
    return out


def entrypoint():
    """Main CLI entrypoint. Defaults to 'serve' when no subcommand is given."""
    _register_sub_apps()

    if len(sys.argv) <= 1:
        sys.argv.insert(1, "serve")
    elif sys.argv[1].startswith("-") and sys.argv[1] not in ("--help", "-h"):
        sys.argv.insert(1, "serve")

    sys.argv[:] = _expand_bare_flags(sys.argv)

    app()

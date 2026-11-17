"""Auto-detect best settings for the current machine.

Called from ``main_pre`` when ``--guess-settings`` is passed.  All detection
is done *without* importing ``torch`` so that environment variables can be set
before the CUDA runtime initialises.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import re
import sys
from typing import Mapping, Optional, Sequence, TYPE_CHECKING

from ..cli_args_types import VRAM_MODES, VAE_MODES, ATTENTION_MODES

if TYPE_CHECKING:
    from ..cli_args_types import Configuration

logger = logging.getLogger(__name__)



def _total_ram_gb() -> float:
    """Return total physical RAM in GiB."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass

    # fallback: /proc/meminfo on Linux
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except (OSError, ValueError):
        pass

    return 0.0


def _has_nvidia_gpu() -> bool:
    """True if ``nvidia-smi`` is on PATH (works on Linux and Windows)."""
    import shutil
    return shutil.which("nvidia-smi") is not None


def _has_amd_gpu() -> bool:
    """Heuristic: check for ROCm tooling or ``/dev/kfd``."""
    import shutil
    if shutil.which("rocm-smi") is not None or shutil.which("rocminfo") is not None:
        return True
    if os.path.exists("/dev/kfd"):
        return True
    return False


_BENIGN_PROCESS_RE = re.compile(
    r"^(python|uv|comfyui|nv|nvidia-smi|gnome-remote-desktop-daemon|xorg|xwayland|kwin|mutter)",
    re.IGNORECASE,
)
_BENIGN_GPU_PROCESS_NAMES = frozenset({
    "gnome-remote-desktop-daemon",
    "steamwebhelper",
})
_COMPETING_GPU_PROCESS_MIN_MEMORY_MIB = 1024


def _parse_nvidia_smi_memory_mib(value: str) -> int | None:
    m = re.match(r"\s*(\d+)\s*(?:MiB)?\s*$", value)
    if not m:
        return None
    return int(m.group(1))


def _current_process_family() -> set[int]:
    pids = {os.getpid()}
    try:
        import psutil

        current = psutil.Process()
        pids.update(parent.pid for parent in current.parents())
        pids.update(child.pid for child in current.children(recursive=True))
    except Exception:
        pass
    return pids


def _competing_gpu_processes() -> list[str]:
    """Return names of material, unrelated processes using the NVIDIA GPU.

    Returns an empty list when ``nvidia-smi`` is unavailable or fails.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
        current_family = _current_process_family()
        names: list[str] = []
        for line in result.stdout.strip().splitlines():
            # Handle both Unix and Windows paths (nvidia-smi may return
            # full paths like C:\Program Files\Discord\Discord.exe)
            fields = [field.strip() for field in line.split(",", maxsplit=2)]
            if len(fields) == 3:
                pid_field, name_field, mem_field = fields
            elif len(fields) == 2:
                pid_field, name_field, mem_field = fields[0], fields[1], None
            else:
                pid_field, name_field, mem_field = None, fields[0], None
            if pid_field is not None:
                try:
                    if int(pid_field) in current_family:
                        continue
                except ValueError:
                    pass
            raw = name_field.replace("\\", "/")
            proc = os.path.basename(raw)
            used_mib = _parse_nvidia_smi_memory_mib(mem_field) if mem_field is not None else None
            is_small = used_mib is not None and used_mib < _COMPETING_GPU_PROCESS_MIN_MEMORY_MIB
            if (
                proc
                and not _BENIGN_PROCESS_RE.match(proc)
                and proc not in _BENIGN_GPU_PROCESS_NAMES
                and not is_small
            ):
                names.append(proc)
        return names
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []


def _nvidia_compute_caps() -> list[tuple[int, int]]:
    """Return list of (major, minor) compute capabilities for each NVIDIA GPU.

    Empty list if nvidia-smi is missing or fails. Probes via
    `nvidia-smi --query-gpu=compute_cap` so the CUDA runtime doesn't initialise.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []
    caps: list[tuple[int, int]] = []
    for line in result.stdout.strip().splitlines():
        m = re.match(r"\s*(\d+)\.(\d+)\s*$", line)
        if m:
            caps.append((int(m.group(1)), int(m.group(2))))
    return caps


def _nvidia_gpu_names() -> list[str]:
    """Return the product name of every NVIDIA GPU in physical index order.

    Empty list if nvidia-smi is missing or fails. Uses ``nvidia-smi`` so the
    CUDA runtime does not initialise before device visibility is settled.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []
    names: dict[int, str] = {}
    for line in result.stdout.strip().splitlines():
        fields = [field.strip() for field in line.split(",", maxsplit=1)]
        if len(fields) == 2 and fields[0].isdigit():
            names[int(fields[0])] = fields[1]
    return [names[index] for index in sorted(names)]


def _parse_gpu_indexes(selection: str, gpu_count: int) -> Optional[list[int]]:
    """Parse a comma-separated GPU index list; None when it is not plain indexes."""
    indexes: list[int] = []
    for token in selection.split(","):
        token = token.strip()
        if not token.isdigit() or int(token) >= gpu_count:
            return None
        indexes.append(int(token))
    return indexes


def _selected_gpu_indexes(configuration: Configuration, gpu_count: int, environment: Mapping[str, str]) -> list[int]:
    """Physical NVIDIA GPU indexes the configuration leaves visible to torch."""
    if configuration.torch_device is not None:
        if not configuration.torch_device.startswith("cuda:"):
            return []
        return _parse_gpu_indexes(configuration.torch_device.split(":", 1)[1], gpu_count) or []
    selection = configuration.cuda_device
    if selection is None and configuration.default_device is not None:
        return list(range(gpu_count))
    if selection is None or selection == "all":
        visible = environment.get("CUDA_VISIBLE_DEVICES")
        if visible is None:
            return list(range(gpu_count))
        return _parse_gpu_indexes(visible, gpu_count) or []
    return _parse_gpu_indexes(selection, gpu_count) or []


def default_tensor_parallel_size(gpu_names: Sequence[str]) -> int:
    """Tensor-parallel size for a set of visible GPUs: the largest power of two
    that fits when every GPU is the same product, otherwise one.

    Attention heads and MLP widths in the supported model families divide by
    powers of two; an odd rank count would fail the shard divisibility checks.
    """
    if len(gpu_names) < 2 or len(set(gpu_names)) != 1:
        return 1
    size = 1
    while size * 2 <= len(gpu_names):
        size *= 2
    return size


_MODEL_PARALLEL_SETTINGS = ("tensor_parallel_size", "pipeline_parallel_size", "ulysses_degree", "ring_degree")


def _guess_tensor_parallel_size(configuration: Configuration) -> None:
    """Default to tensor parallelism across identical NVIDIA GPUs on Linux.

    Explicit model-parallel settings, launcher variables, and Windows (no NCCL)
    leave the configuration alone.
    """
    from ..distributed.config import resolve_distributed_configuration

    if os.name == "nt":
        return
    if any(getattr(configuration, name, None) is not None for name in _MODEL_PARALLEL_SETTINGS):
        return
    if any(f"COMFYUI_{name.upper()}" in os.environ for name in _MODEL_PARALLEL_SETTINGS):
        return
    distributed = resolve_distributed_configuration(configuration)
    names = _nvidia_gpu_names()
    selected = [names[index] for index in _selected_gpu_indexes(configuration, len(names), os.environ)]
    if distributed.externally_launched or distributed.world_size > 1:
        if (
            distributed.world_size >= 2
            and (distributed.world_size & (distributed.world_size - 1)) == 0
            and selected
            and all("GB10" in name for name in selected)
        ):
            logger.info(
                f"External launch across {distributed.world_size} NVIDIA GB10 ranks detected, enabling tensor parallelism across {distributed.world_size}"
            )
            configuration.tensor_parallel_size = distributed.world_size
            configuration.pipeline_parallel_size = 1
        return
    if max(
        distributed.tensor_parallel_size,
        distributed.pipeline_parallel_size,
        distributed.ulysses_degree * distributed.ring_degree,
    ) > 1:
        return
    size = default_tensor_parallel_size(selected)
    if size > 1:
        logger.info(f"{len(selected)} identical NVIDIA GPUs detected ({selected[0]}), enabling tensor parallelism across {size}")
        configuration.tensor_parallel_size = size


def _amd_gfx_version() -> Optional[str]:
    """Return the GFX target ID (e.g. 'gfx1100', 'gfx1201') or None."""
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                m = re.search(r"(gfx\d+)", line)
                if m:
                    return m.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # fallback: /sys/class/kfd
    try:
        topology = "/sys/class/kfd/kfd/topology/nodes"
        if os.path.isdir(topology):
            for node in sorted(os.listdir(topology)):
                props = os.path.join(topology, node, "properties")
                if os.path.isfile(props):
                    with open(props, encoding="utf-8") as fh:
                        for line in fh:
                            if line.startswith("gfx_target_version"):
                                ver = line.split()[-1].strip()
                                if ver and ver != "0":
                                    major = int(ver) // 10000
                                    minor = (int(ver) % 10000) // 100
                                    patch = int(ver) % 100
                                    return f"gfx{major}{minor:01d}{patch:02d}"
    except (OSError, ValueError):
        pass

    return None


def _has_package(name: str) -> bool:
    return importlib.util.find_spec(name) is not None



def apply_guess_settings(configuration: Configuration) -> None:
    """Mutate *configuration* with auto-detected defaults.

    Only touches settings that are still at their parser defaults so that
    explicit CLI flags always win.
    """
    from ..cli_args_types import PerformanceFeature

    is_macos = sys.platform == "darwin"
    is_nvidia = _has_nvidia_gpu()
    is_amd = _has_amd_gpu()
    ram_gb = _total_ram_gb()

    # macOS / Apple Silicon: unified memory means CPU and GPU share RAM,
    # so offloading to CPU just adds copy overhead.
    if is_macos:
        user_set_vram = any(getattr(configuration, f, False) for f in VRAM_MODES)
        if not user_set_vram:
            logger.info("macOS detected (unified memory), enabling gpu_only")
            configuration.gpu_only = True

    if ram_gb and ram_gb < 32 and not configuration.disable_pinned_memory:
        logger.info(f"{ram_gb:.1f} GB RAM detected, disabling pinned memory")
        configuration.disable_pinned_memory = True

    if ram_gb and ram_gb <= 32 and not configuration.disable_smart_memory:
        logger.info(f"{ram_gb:.1f} GB RAM detected, disabling smart memory")
        configuration.disable_smart_memory = True

    if is_nvidia:
        fast = set(configuration.fast) if configuration.fast else set()
        if PerformanceFeature.CublasOps not in fast:
            fast.add(PerformanceFeature.CublasOps)
            configuration.fast = list(fast)
            logger.info("NVIDIA GPU detected, enabling cublas_ops")

        # cudaMallocAsync is unsafe on Windows (CUDA driver bug surfaces as
        # silent corruption / device-side asserts under stream pressure) and
        # unsupported on Pascal (sm_6x). Disable in those cases — elsewhere
        # leave the explicit user/default setting alone.
        caps = _nvidia_compute_caps()
        sm6x = any(major == 6 for major, _ in caps)
        if sys.platform == "win32" or sm6x:
            reason = "Windows" if sys.platform == "win32" else f"SM 6.x ({caps})"
            logger.info(f"{reason} detected, disabling cudaMallocAsync")
            configuration.disable_cuda_malloc = True
            configuration.cuda_malloc = False

    if is_nvidia:
        user_set_vram = any(getattr(configuration, f, False) for f in VRAM_MODES)
        if not user_set_vram:
            procs = _competing_gpu_processes()
            if procs:
                logger.info(f"competing GPU processes detected ({', '.join(procs)}), enabling novram")
                configuration.novram = True
        _guess_tensor_parallel_size(configuration)

    if is_amd:
        if not any(getattr(configuration, f, False) for f in VAE_MODES):
            gfx = _amd_gfx_version()
            if gfx and gfx.startswith("gfx12"):
                logger.info(f"AMD RDNA 4 ({gfx}) detected, enabling fp16 VAE")
                configuration.fp16_vae = True
            else:
                logger.info(f"AMD GPU ({gfx or 'unknown'}) detected, enabling fp32 VAE")
                configuration.fp32_vae = True

    user_set_attn = any(getattr(configuration, f, False) for f in ATTENTION_MODES)
    if not user_set_attn:
        if is_macos:
            logger.info("macOS detected, using PyTorch cross attention")
            configuration.use_pytorch_cross_attention = True
        elif is_amd and sys.platform == "win32":
            logger.info("AMD GPU on Windows detected, enabling quad cross attention")
            configuration.use_quad_cross_attention = True
        elif _has_package("sageattention"):
            logger.info("sageattention found, enabling sage attention")
            configuration.use_sage_attention = True
        elif _has_package("xformers"):
            logger.info("xformers found, keeping xformers enabled")
            configuration.disable_xformers = False
        else:
            logger.info("using default PyTorch cross attention")
            configuration.use_pytorch_cross_attention = True

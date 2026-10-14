"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Comfy

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations
from .cmd.main_pre import tracer

from contextlib import contextmanager
from contextlib import nullcontext
from enum import Enum
from functools import lru_cache
from threading import RLock
from typing import Final
from typing import List
from typing import Optional
from typing import Sequence
import gc
import logging
import os
import platform
import sys
import warnings
import weakref

from opentelemetry.trace import get_current_span
import comfy_aimdo.host_buffer
import comfy_aimdo.vram_buffer
import psutil
import torch

from . import interruption
from . import memory_management
from . import quant_ops
from . import system_memory
from .cli_args import PerformanceFeature
from .cli_args import args
from .component_model.deprecation import _deprecate_method
from .internal_logging import detail
from .model_management_types import ModelManageable

model_management_lock = RLock()

# https://github.com/sayakpaul/diffusers-torchao/blob/bade7a6abb1cab9ef44782e6bcfab76d0237ae1f/inference/benchmark_image.py#L3
# This setting optimizes performance on NVIDIA GPUs with Ampere architecture (e.g., A100, RTX 30 series) or newer.
torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .model_patcher import ModelPatcher


class VRAMState(Enum):
    DISABLED = 0  # No vram present: no need to move models to vram
    NO_VRAM = 1  # Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5  # No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.


class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2


# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0

# Training-related state.
in_training = False
training_fp8_bwd = False


def get_supported_float8_types():
    float8_types = []
    try:
        float8_types.append(torch.float8_e4m3fn)
    except:
        pass
    try:
        float8_types.append(torch.float8_e4m3fnuz)
    except:
        pass
    try:
        float8_types.append(torch.float8_e5m2)
    except:
        pass
    try:
        float8_types.append(torch.float8_e5m2fnuz)
    except:
        pass
    try:
        float8_types.append(torch.float8_e8m0fnu)
    except:
        pass
    return float8_types


FLOAT8_TYPES = get_supported_float8_types()

xpu_available = False
torch_version = ""
try:
    torch_version = torch.version.__version__
    temp = torch_version.split(".")
    torch_version_numeric = (int(temp[0]), int(temp[1]))
except:
    pass

lowvram_available = True
if args.deterministic:
    logger.debug("Using deterministic algorithms for pytorch")
    torch.use_deterministic_algorithms(True, warn_only=True)

directml_device = None
if args.directml is not None:
    logger.warning("WARNING: torch-directml barely works, is very slow, has not been updated in over 1 year and might be removed soon, please don't use it, there are better options.")
    import torch_directml

    device_index = args.directml
    if device_index < 0:
        directml_device = torch_directml.device()
    else:
        directml_device = torch_directml.device(device_index)
    logger.debug("Using directml with device: {}".format(torch_directml.device_name(device_index)))
    # torch_directml.disable_tiled_resources(True)
    lowvram_available = False  # TODO: need to find a way to get free memory in directml before this can be enabled by default.

try:
    _ = torch.xpu.device_count()
    xpu_available = torch.xpu.is_available()
except:
    xpu_available = False

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass

try:

    _ = torch.npu.device_count()
    npu_available = torch.npu.is_available()
except:
    npu_available = False

try:

    _ = torch.mlu.device_count()
    mlu_available = torch.mlu.is_available()
except:
    mlu_available = False

try:
    ixuca_available = hasattr(torch, "corex")
except:
    ixuca_available = False

if args.cpu or (args.torch_device is not None and args.torch_device == "cpu"):
    cpu_state = CPUState.CPU


def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False


def is_ascend_npu():
    global npu_available
    if npu_available:
        return True
    return False


def is_mlu():
    global mlu_available
    if mlu_available:
        return True
    return False


def is_ixuca():
    global ixuca_available
    if ixuca_available:
        return True
    return False


def is_wsl():
    version = platform.uname().release
    if version.endswith("-Microsoft"):
        return True
    elif version.endswith("microsoft-standard-WSL2"):
        return True
    return False


def get_torch_device():
    global directml_device
    global cpu_state
    if directml_device:
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        elif is_ascend_npu():
            return torch.device("npu", torch.npu.current_device())
        elif is_mlu():
            return torch.device("mlu", torch.mlu.current_device())
        else:
            try:
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            except:
                warnings.warn("torch.cuda.current_device() did not return a device, returning a CPU torch device")
                return torch.device("cpu")


def get_all_torch_devices(exclude_current=False):
    global cpu_state
    devices = []
    if cpu_state == CPUState.GPU:
        # NVIDIA + AMD/ROCm both expose their GPUs through torch.cuda.*;
        # without the AMD arm, single-GPU ROCm users get an empty list
        # which silently turns unload_all_models() into a no-op.
        if is_nvidia() or is_amd():
            for i in range(torch.cuda.device_count()):
                devices.append(torch.device("cuda", i))
        elif is_intel_xpu():
            for i in range(torch.xpu.device_count()):
                devices.append(torch.device("xpu", i))
        elif is_ascend_npu():
            for i in range(torch.npu.device_count()):
                devices.append(torch.device("npu", i))
        elif is_mlu():
            for i in range(torch.mlu.device_count()):
                devices.append(torch.device("mlu", i))
        else:
            # Fallback for unhandled GPU backends (e.g. DirectML): at least
            # report the current device so callers like unload_all_models()
            # do not silently no-op.
            devices.append(get_torch_device())
    else:
        devices.append(get_torch_device())
    if exclude_current:
        current = get_torch_device()
        if current in devices:
            devices.remove(current)
    return devices


def get_model_management_devices(exclude_current=False):
    """Return the devices whose models are owned by this process.

    Device discovery remains process-wide so a driver can select local peers.
    Process-peer ranks, however, have an independent model manager and Aimdo
    allocator. They must not inspect or reclaim memory from another rank's GPU.
    """
    if args.model_management_device_scope == "local":
        devices = [get_torch_device()]
        return [] if exclude_current else devices
    return get_all_torch_devices(exclude_current=exclude_current)

def get_gpu_device_options():
    """Return list of device option strings for node widgets.

    Always includes "default" and "cpu". When multiple GPUs are present,
    adds "gpu:0", "gpu:1", etc. (vendor-agnostic labels).
    """
    options = ["default", "cpu"]
    devices = get_all_torch_devices()
    if len(devices) > 1:
        for i in range(len(devices)):
            options.append(f"gpu:{i}")
    return options

def get_gpu_device_options_no_cpu():
    """Variant of get_gpu_device_options that omits "cpu".

    Intended for components like the VAE selector where running on CPU
    is impractical and should not be offered as a choice.
    """
    return [o for o in get_gpu_device_options() if o != "cpu"]

def resolve_gpu_device_option(option: str):
    """Resolve a device option string to a torch.device.

    Returns None for "default" (let the caller use its normal default).
    Returns torch.device("cpu") for "cpu".
    For "gpu:N", returns the Nth torch device. Returns None if the
    index is out of range, the option string is malformed, or
    unrecognized (callers are expected to log their own context-rich
    message before falling back to the default device).
    """
    if option is None or option == "default":
        return None
    if option == "cpu":
        return torch.device("cpu")
    if option.startswith("gpu:"):
        try:
            idx = int(option[4:])
        except ValueError:
            return None
        devices = get_all_torch_devices()
        if 0 <= idx < len(devices):
            return devices[idx]
    return None

@contextmanager
def cuda_device_context(device):
    """Context manager that sets torch.cuda.current_device to match *device*.

    Used when running operations on a non-default CUDA device so that custom
    CUDA kernels (e.g. comfy_kitchen fp8 quantization) pick up the correct
    device index.  The previous device is restored on exit.

    No-op when *device* is not CUDA, has no explicit index, or already matches
    the current device.
    """
    prev = None
    if device.type == "cuda" and device.index is not None:
        prev = torch.cuda.current_device()
        if prev != device.index:
            torch.cuda.set_device(device)
        else:
            prev = None
    try:
        yield
    finally:
        if prev is not None:
            torch.cuda.set_device(prev)

def get_total_memory(dev=None, torch_total_too=False):
    global directml_device
    if dev is None:
        dev = get_torch_device()

    mem_total_torch = 0
    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_total = system_memory.virtual_memory_total()
        mem_total_torch = mem_total
    else:
        if directml_device:
            mem_total = 1024 * 1024 * 1024  # TODO
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            mem_total_xpu = torch.xpu.get_device_properties(dev).total_memory
            mem_total_torch = mem_reserved
            mem_total = mem_total_xpu
        elif is_ascend_npu():
            stats = torch.npu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_npu = torch.npu.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_npu
        elif is_mlu():
            stats = torch.mlu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_mlu = torch.mlu.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_mlu
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda
    if torch_total_too:
        return mem_total, mem_total_torch
    else:
        return mem_total


def mac_version():
    try:
        return tuple(int(n) for n in platform.mac_ver()[0].split("."))
    except:
        return None


# we're required to call get_device_name early on to initialize the methods get_total_memory will call
if torch.cuda.is_available() and hasattr(torch.version, "hip") and torch.version.hip is not None:
    logger.debug(f"Detected HIP device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
total_ram = system_memory.virtual_memory_total() / (1024 * 1024)
logger.debug("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))
cgroup_ram_limit = system_memory.cgroup_memory_limit()
if cgroup_ram_limit is not None:
    logger.debug(
        "RAM limited by cgroup to {:0.0f} MB (host has {:0.0f} MB)".format(
            cgroup_ram_limit / (1024 * 1024),
            psutil.virtual_memory().total / (1024 * 1024),
        )
    )

try:
    logger.debug("pytorch version: {}".format(torch_version))
    mac_ver = mac_version()
    if mac_ver is not None:
        logger.debug("Mac Version {}".format(mac_ver))
except:
    pass


class _ComfyOutOfMemoryException(RuntimeError):
    pass


try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = _ComfyOutOfMemoryException

try:
    ACCELERATOR_ERROR = torch.AcceleratorError
except AttributeError:
    ACCELERATOR_ERROR = RuntimeError


def is_oom(e):
    if isinstance(e, OOM_EXCEPTION):
        return True
    if isinstance(e, ACCELERATOR_ERROR) and (getattr(e, 'error_code', None) == 2 or "out of memory" in str(e).lower()):
        discard_cuda_async_error()
        return True
    return False


def raise_non_oom(e):
    if not is_oom(e):
        raise e


XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if args.disable_xformers:
    XFORMERS_IS_AVAILABLE = False
else:
    try:
        import xformers
        import xformers.ops

        XFORMERS_IS_AVAILABLE = True
        try:
            XFORMERS_IS_AVAILABLE = xformers._has_cpp_library
        except:
            pass
        try:
            XFORMERS_VERSION = xformers.version.__version__
            logger.debug("xformers version: {}".format(XFORMERS_VERSION))
            if XFORMERS_VERSION.startswith("0.0.18"):
                logger.warning("\nWARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
                logger.warning("Please downgrade or upgrade xformers to a different version.\n")
                XFORMERS_ENABLED_VAE = False
        except:
            pass
    except:
        XFORMERS_IS_AVAILABLE = False


def is_nvidia():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False


def is_amd():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.hip:
            return True
    return False


def amd_min_version(device=None, min_rdna_version=0):
    if not is_amd():
        return False

    if is_device_cpu(device):
        return False

    arch = torch.cuda.get_device_properties(device).gcnArchName
    if arch.startswith('gfx') and len(arch) == 7:
        try:
            cmp_rdna_version = int(arch[4]) + 2
        except:
            cmp_rdna_version = 0
        if cmp_rdna_version >= min_rdna_version:
            return True

    return False


MIN_WEIGHT_MEMORY_RATIO = 0.4
if is_nvidia():
    MIN_WEIGHT_MEMORY_RATIO = 0.0

ENABLE_PYTORCH_ATTENTION = False
if args.use_pytorch_cross_attention:
    ENABLE_PYTORCH_ATTENTION = True
    XFORMERS_IS_AVAILABLE = False

try:
    if is_nvidia() or is_amd():
        if torch_version_numeric[0] >= 2:
            if ENABLE_PYTORCH_ATTENTION == False and args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
                ENABLE_PYTORCH_ATTENTION = True
    if is_intel_xpu() or is_ascend_npu() or is_mlu() or is_ixuca():
        if args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
            ENABLE_PYTORCH_ATTENTION = True
except:
    pass

SUPPORT_FP8_OPS = args.supports_fp8_compute

AMD_RDNA2_AND_OLDER_ARCH = ["gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036", "gfx1010", "gfx1011", "gfx1012", "gfx906", "gfx900", "gfx803"]
AMD_ENABLE_MIOPEN_ENV = 'COMFYUI_ENABLE_MIOPEN'

try:
    if is_amd():
        arch = torch.cuda.get_device_properties(get_torch_device()).gcnArchName.split(':')[0]
        if not (any((a in arch) for a in AMD_RDNA2_AND_OLDER_ARCH)):
            if os.getenv(AMD_ENABLE_MIOPEN_ENV) != '1':
                torch.backends.cudnn.enabled = False  # Seems to improve things a lot on AMD
                logger.info("Set: torch.backends.cudnn.enabled = False for better AMD performance.")
        try:
            rocm_version = tuple(map(int, str(torch.version.hip).split(".")[:2]))
        except:
            rocm_version = (6, -1)

        def aotriton_supported():
            """Whether pytorch reports flash attention as usable on this gpu.

            can_use_flash_attention() evaluates runtime eligibility for the given
            parameters; on a ROCm build that includes checking the gpu arch against the
            arches AOTriton was built for. Querying it avoids assuming where the kernel
            images live inside the torch install. The probe tensor is shaped and
            typed to pass the unrelated SDPA checks, so False means no hardware support
            rather than a rejected shape.

            It answers True on a supported arch whose kernel image was never shipped,
            and that only fails at launch, without raising. So run one attention
            through the flash backend and force the pending error check.
            """
            try:
                device = get_torch_device()
                if not torch.backends.cuda.is_flash_attention_available():  # not built with flash attention
                    return False
                q = torch.zeros((1, 1, 8, 64), dtype=torch.float16, device=device)
                params = torch.backends.cuda.SDPAParams(q, q, q, None, 0.0, False, False)
                if not torch.backends.cuda.can_use_flash_attention(params, False):
                    return False
                from torch.nn.attention import SDPBackend
                from torch.nn.attention import sdpa_kernel
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    torch.nn.functional.scaled_dot_product_attention(q, q, q)
                torch.cuda.synchronize()
                torch.zeros(1, device=device).add_(1).item()  # raises if the launch above failed
                return True
            except Exception as e:
                logger.warning("Could not run flash attention, disabling it: {}".format(e))
                return False


        logger.debug("AMD arch: {}".format(arch))
        logger.debug("ROCm version: {}".format(rocm_version))
        if args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
            if aotriton_supported():  # AMD efficient attention implementation depends on aotriton.
                if torch_version_numeric >= (2, 7):  # works on 2.6 but doesn't actually seem to improve much
                    if any((a in arch) for a in ["gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1170", "gfx1171"]):  # TODO: more arches, TODO: gfx950
                        ENABLE_PYTORCH_ATTENTION = True
                if rocm_version >= (7, 0):
                    if any((a in arch) for a in ["gfx1200", "gfx1201"]):
                        ENABLE_PYTORCH_ATTENTION = True
        if torch_version_numeric >= (2, 7) and rocm_version >= (6, 4):
            if any((a in arch) for a in ["gfx1200", "gfx1201", "gfx950", "gfx1170", "gfx1171"]):  # TODO: more arches, "gfx942" gives error on pytorch nightly 2.10 1013 rocm7.0
                SUPPORT_FP8_OPS = True

except:
    pass

if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

PRIORITIZE_FP16 = False  # TODO: remove and replace with something that shows exactly which dtype is faster than the other
try:
    if (is_nvidia() or is_amd()) and PerformanceFeature.Fp16Accumulation in args.fast:
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        PRIORITIZE_FP16 = True  # TODO: limit to cards where it actually boosts performance
        logger.info("Enabled fp16 accumulation.")
except:
    pass


def set_cudnn_benchmark():
    if torch.cuda.is_available() and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = PerformanceFeature.AutoTune in args.fast

try:
    if torch_version_numeric >= (2, 5):
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
except:
    logger.warning("Warning, could not set allow_fp16_bf16_reduction_math_sdp")

if args.lowvram:
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
elif args.novram:
    set_vram_to = VRAMState.NO_VRAM
elif args.highvram or args.gpu_only:
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
FORCE_FP16 = False
FORCE_BF16 = False
if args.force_fp32:
    logger.info("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if args.force_fp16:
    logger.debug("Forcing FP16.")
    FORCE_FP16 = True

if args.force_bf16:
    logger.info("Force BF16")
    FORCE_BF16 = True

if lowvram_available:
    if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        vram_state = set_vram_to

if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

logger.info(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = args.disable_smart_memory

if DISABLE_SMART_MEMORY:
    logger.debug("Disabling smart memory management")


def get_torch_device_name(device):
    if hasattr(device, 'type'):
        if device.type == "cuda":
            try:
                allocator_backend = torch.cuda.get_allocator_backend()
            except:
                allocator_backend = ""
            return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
        elif device.type == "xpu":
            return "{} {}".format(device, torch.xpu.get_device_name(device))
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} {}".format(device, torch.xpu.get_device_name(device))
    elif is_ascend_npu():
        return "{} {}".format(device, torch.npu.get_device_name(device))
    elif is_mlu():
        return "{} {}".format(device, torch.mlu.get_device_name(device))
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))


try:
    logger.debug("Device: {}".format(get_torch_device_name(get_torch_device())))
except:
    logger.warning("Could not pick default device.")
try:
    for device in get_all_torch_devices(exclude_current=True):
        logger.info("Device: {}".format(get_torch_device_name(device)))
except:
    pass

current_loaded_models: Final[List["LoadedModel"]] = []

DIRTY_MMAPS = set()


def _loaded_model_is_reclaimable(loaded_model, device, keep_loaded) -> bool:
    return (
        (device is None or loaded_model.device == device)
        and loaded_model not in keep_loaded
        and not loaded_model.is_dead()
    )


def projected_dynamic_vram_available_memory(devices, keep_models=()):
    """Project per-device capacity using the same keep/ejection predicate as a load."""
    keep_models = tuple(keep_models)
    keep_loaded = tuple(
        loaded_model for loaded_model in current_loaded_models
        if any(loaded_model.model is model for model in keep_models)
    )
    projected = {}
    for device in devices:
        available = memory_management.dynamic_vram_available_memory(device)
        for loaded_model in current_loaded_models:
            if not _loaded_model_is_reclaimable(loaded_model, device, keep_loaded):
                continue
            model = loaded_model.model
            reclaimable = getattr(model, "reclaimable_non_vbar_memory", None)
            if reclaimable is None:
                reclaimable = model.loaded_size
            available += int(reclaimable())
        projected[device] = available
    return projected

PIN_PRESSURE_HYSTERESIS = 256 * 1024 * 1024

#Freeing registerables on pressure does imply a GPU sync, so go big on
#the hysteresis so each expensive sync gives us back a good chunk.
REGISTERABLE_PIN_HYSTERESIS = 2048 * 1024 * 1024
WINDOWS_PIN_EVICTION_SWAP_PERCENT = 5.0
WINDOWS_PIN_EVICTION_EMERGENCY_AVAILABLE = 512 * 1024 ** 2

def module_size(module):
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nbytes
    return module_mem

def module_mmap_residency(module, free=False):
    mmap_touched_mem = 0
    module_mem = 0
    bounced_mmaps = set()
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nbytes
        storage = t._qdata.untyped_storage() if isinstance(t, quant_ops.QuantizedTensor) else t.untyped_storage()
        if not getattr(storage, "_comfy_tensor_mmap_touched", False):
            continue
        mmap_touched_mem += t.nbytes
        if not free:
            continue
        storage._comfy_tensor_mmap_touched = False
        mmap_obj = storage._comfy_tensor_mmap_refs[0]
        if mmap_obj in bounced_mmaps:
            continue
        mmap_obj.bounce()
        bounced_mmaps.add(mmap_obj)
    return mmap_touched_mem, module_mem


def mark_mmap_dirty(storage):
    mmap_refs = getattr(storage, "_comfy_tensor_mmap_refs", None)
    if mmap_refs is not None:
        DIRTY_MMAPS.add(mmap_refs[0])

PIN_SUBSETS = [ "weights", "patches" ]
LOADED_PIN_SUBSETS = [ "weights-loaded", "patches-loaded" ]
FAST_PIN_SUBSETS = [ "weights-fast", "patches-fast" ]

def models_for_pin_eviction(active, current_prompt=None):
    for loaded_model in current_loaded_models:
        model = loaded_model.model
        if model is None or not model.is_dynamic():
            continue
        # a model-parallel rank's stand-in manages its memory in the rank process and holds no pins here
        pin_state = getattr(model.model, "dynamic_pins", {}).get(model.load_device)
        if pin_state is None:
            continue
        if ((active is None or pin_state["active"] == active) and
            (current_prompt is None or pin_state["current_prompt"] == current_prompt)):
            yield model

def free_model_pins(size, subsets, current_prompt, active, registrations=False):
    freed_total = 0
    for model in models_for_pin_eviction(active, current_prompt=current_prompt):
        if size <= 0:
            return freed_total
        if registrations:
            freed = model.unregister_inactive_pins(size, subsets=subsets)
        else:
            freed = model.partially_unload_ram(size, subsets=subsets)
        if freed > 0:
            detail(
                "Pin eviction: model=%s subsets=%s workflow=%s active=%s action=%s freed_mb=%.1f",
                model.model.__class__.__name__, subsets, current_prompt, active,
                "unregister" if registrations else "destroy", freed / (1024 ** 2),
            )
        freed_total += freed
        size -= freed
    return freed_total

def pin_eviction_tiers(loaded, evict_active):
    tiers = [
        (FAST_PIN_SUBSETS, False, False),
        (PIN_SUBSETS, False, None),
        (LOADED_PIN_SUBSETS, False, None),
        (FAST_PIN_SUBSETS, True, False),
        (LOADED_PIN_SUBSETS, True, None),
    ]
    if not loaded:
        tiers.append((PIN_SUBSETS, True, False))
        if evict_active:
            tiers.extend([
                (FAST_PIN_SUBSETS, False, True),
                (FAST_PIN_SUBSETS, True, True),
                (PIN_SUBSETS, True, True),
            ])
    return tiers

def registration_eviction_tiers(evict_active):
    subsets = PIN_SUBSETS + LOADED_PIN_SUBSETS
    tiers = [
        (FAST_PIN_SUBSETS, False, False, False),
        (subsets, False, False, True),
        (FAST_PIN_SUBSETS, True, False, False),
        (subsets, True, False, True),
    ]
    if evict_active:
        tiers.extend([
            (FAST_PIN_SUBSETS, False, True, False),
            (subsets, False, True, True),
            (FAST_PIN_SUBSETS, True, True, False),
            (subsets, True, True, True),
        ])
    return tiers

def free_pins(size, evict_active=False, loaded=False):
    freed = 0
    for subsets, current_prompt, active in pin_eviction_tiers(loaded, evict_active):
        freed += free_model_pins(size - freed, subsets, current_prompt, active)
    return freed

def should_free_pins_for_ram_pressure(shortfall):
    if shortfall <= 0:
        return False
    if not WINDOWS:
        return True
    if system_memory.virtual_memory_available() < WINDOWS_PIN_EVICTION_EMERGENCY_AVAILABLE:
        return True
    try:
        return psutil.swap_memory().percent >= WINDOWS_PIN_EVICTION_SWAP_PERCENT
    except RuntimeError as err:
        logging.warning("Could not read Windows swap usage; falling back to RAM-pressure pin eviction: %s", err)
        return True

def pin_budget_shortfall(size):
    """Bytes by which committing ``size`` more host memory would cut into the
    pin reserve; zero or less means it fits.

    Available memory is MemAvailable capped by every limited cgroup's
    ``limit - working set`` (see ``system_memory``), so memory held by other
    processes, VMs, and the rest of this container is always accounted for.
    """
    headroom = max(pinned_memory_reserve(), memory_management.RAM_CACHE_HEADROOM // 2)
    return size + headroom - system_memory.virtual_memory_available()


def ensure_pin_budget(size, evict_active=False, loaded=False):
    """Whether ``size`` more bytes of host memory may be pinned (or otherwise
    held, e.g. a host activation cache) right now, evicting this process's
    idle pins first if that makes room. Measured at call time, never cached."""
    shortfall = pin_budget_shortfall(size)
    if shortfall <= 0:
        return True
    free_pins(shortfall + PIN_PRESSURE_HYSTERESIS, evict_active=evict_active, loaded=loaded)
    return pin_budget_shortfall(size) <= 0

def free_registrations(shortfall, evict_active=True):
    if MAX_PINNED_MEMORY <= 0:
        return False
    if shortfall <= 0:
        return True

    shortfall += REGISTERABLE_PIN_HYSTERESIS
    for subsets, current_prompt, active, registrations in registration_eviction_tiers(evict_active):
        shortfall -= free_model_pins(shortfall, subsets, current_prompt, active, registrations=registrations)
    return shortfall <= REGISTERABLE_PIN_HYSTERESIS

def ensure_pin_registerable(size, evict_active=True):
    return free_registrations(TOTAL_PINNED_MEMORY + size - MAX_PINNED_MEMORY, evict_active=evict_active)

class LoadedModel:
    def __init__(self, model: ModelManageable):
        self._set_model(model)
        self.device = model.load_device
        self.real_model = None
        self.currently_used = True
        self.model_finalizer = None
        self._patcher_finalizer = None

    def _set_model(self, model: ModelPatcher):
        self._model = weakref.ref(model)
        if model.parent is not None:
            self._parent_model = weakref.ref(model.parent)
            self._patcher_finalizer = weakref.finalize(model, self._switch_parent)
            self._patcher_finalizer.atexit = False

    def _switch_parent(self):
        model = self._parent_model()
        if model is not None:
            self._set_model(model)
            self.device = model.load_device

    @property
    def model(self) -> ModelManageable:
        return self._model()

    def model_memory(self):
        return self.model.model_size()

    def model_loaded_memory(self):
        return self.model.loaded_size()

    def model_offloaded_memory(self):
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device):
        if device == self.model.current_loaded_device():
            return self.model_offloaded_memory()
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        # if self.model.loaded_size() > 0:
        use_more_vram = lowvram_model_memory
        if use_more_vram == 0:
            use_more_vram = 1e32
        self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)

        real_model = self.model.model


        self.real_model = weakref.ref(real_model)
        self.model_finalizer = weakref.finalize(real_model, cleanup_models)
        self.model_finalizer.atexit = False
        return real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter() > 0:
            return True
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None:
            if memory_to_free < self.model.loaded_size():
                freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
                if freed >= memory_to_free:
                    return False
        self.model.detach(unpatch_weights)
        self.model_finalizer.detach()
        self.model_finalizer = None
        self.real_model = None
        return True

    def model_use_more_vram(self, extra_memory, force_patch_weights=False):
        return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)

    def __eq__(self, other):
        return self.model is other.model

    def __str__(self):
        if self.model is not None:
            return f"<LoadedModel {str(self.model)}>"
        else:
            return f"<LoadedModel>"

    def __del__(self):
        if self._patcher_finalizer is not None:
            self._patcher_finalizer.detach()

    def is_dead(self):
        return self.real_model is not None and self.real_model() is not None and self.model is None


def use_more_memory(extra_memory, loaded_models, device):
    for m in loaded_models:
        if m.device == device:
            extra_memory -= m.model_use_more_vram(extra_memory)
            if extra_memory <= 0:
                break


def offloaded_memory(loaded_models, device):
    offloaded_mem = 0
    for m in loaded_models:
        if m.device == device:
            offloaded_mem += m.model_offloaded_memory()
    return offloaded_mem


# Avoid a WMI query during import: it can stall in Windows containers.
WINDOWS = sys.platform == "win32"

EXTRA_RESERVED_VRAM = 400 * 1024 * 1024
if WINDOWS:
    from . import windows

    EXTRA_RESERVED_VRAM = 600 * 1024 * 1024  # Windows is higher because of the shared vram issue
    if total_vram > (15 * 1024):  # more extra reserved vram on 16GB+ cards
        EXTRA_RESERVED_VRAM += 100 * 1024 * 1024

    def get_free_ram():
        return windows.get_free_ram()
else:
    def get_free_ram():
        return system_memory.virtual_memory_available()

if args.reserve_vram is not None:
    EXTRA_RESERVED_VRAM = args.reserve_vram * 1024 * 1024 * 1024
    logger.debug("Reserving {}MB vram for other applications.".format(EXTRA_RESERVED_VRAM / (1024 * 1024)))


def extra_reserved_memory():
    return EXTRA_RESERVED_VRAM


def minimum_inference_memory():
    return (1024 * 1024 * 1024) * 0.8 + extra_reserved_memory()


def trim_memory() -> bool:
    """
    Trims memory usage, returning reserved memory to the system

    Only supported on Windows and Linux
    :return:
    """
    try:
        if sys.platform.startswith('linux'):
            import ctypes
            import ctypes.util
            libc_path = ctypes.util.find_library('c')
            if not libc_path:
                return False

            libc = ctypes.CDLL(libc_path)

            if hasattr(libc, 'malloc_trim'):
                return libc.malloc_trim(0) == 1
            else:
                return False
        elif sys.platform == 'win32':
            import ctypes.wintypes
            try:
                kernel32 = ctypes.WinDLL("kernel32")
                empty_working_set = kernel32.K32EmptyWorkingSet
            except AttributeError:
                psapi = ctypes.WinDLL("psapi")
                empty_working_set = psapi.EmptyWorkingSet
            empty_working_set.argtypes = [ctypes.wintypes.HANDLE]
            empty_working_set.restype = ctypes.wintypes.BOOL
            handle = ctypes.wintypes.HANDLE(-1)
            success = empty_working_set(handle)
            return bool(success)
        else:
            return False
    except Exception as exc_info:
        logger.warning("failed to trim", exc_info=exc_info)
        return False


@tracer.start_as_current_span("Free Memory")
def free_memory(memory_required, device, keep_loaded=[], for_dynamic=False, pins_required=0, ram_required=0) -> List[LoadedModel]:
    span = get_current_span()
    span.set_attribute("memory_required", memory_required)
    with model_management_lock:
        unloaded_models = _free_memory(
            memory_required,
            device,
            keep_loaded,
            for_dynamic=for_dynamic,
            pins_required=pins_required,
            ram_required=ram_required,
        )
        span.set_attribute("unloaded_models", list(map(str, unloaded_models)))
        return unloaded_models


def _free_memory(memory_required, device, keep_loaded=[], for_dynamic=False, pins_required=0, ram_required=0):
    cleanup_models_gc()
    if not for_dynamic:
        detail("Non dynamic memory free called: memory_required=%s pins_required=%s ram_required=%s", memory_required, pins_required, ram_required)
    memory_management.extra_ram_release(max(pins_required, ram_required))
    unloaded_model = []
    can_unload = []
    unloaded_models = []

    for i in range(len(current_loaded_models) - 1, -1, -1):
        shift_model = current_loaded_models[i]
        if _loaded_model_is_reclaimable(shift_model, device, keep_loaded):
            can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
            shift_model.currently_used = False

    can_unload_sorted = sorted(can_unload)
    for x in can_unload_sorted:
        i = x[-1]
        memory_to_free = 1e32
        pins_to_free = 0
        if not DISABLE_SMART_MEMORY or device is None:
            memory_to_free = 0 if device is None else memory_required - get_free_memory(device)
            pins_to_free = pins_required - get_free_ram()
            if current_loaded_models[i].model.is_dynamic() and for_dynamic:
                freed = current_loaded_models[i].model.partially_unload(current_loaded_models[i].model.offload_device, memory_to_free)
                memory_required -= freed
                memory_to_free = 0
        if memory_to_free > 0 and current_loaded_models[i].model_unload(memory_to_free):
            logger.debug(f"Unloading {current_loaded_models[i].model.model.__class__.__name__}")
            unloaded_model.append(i)
        if pins_to_free > 0:
            logger.debug(f"PIN Unloading {current_loaded_models[i].model.model.__class__.__name__}")
            current_loaded_models[i].model.partially_unload_ram(pins_to_free)

    for x in can_unload_sorted:
        i = x[-1]
        ram_to_free = ram_required - system_memory.virtual_memory_available()
        if ram_to_free <= 0 and i not in unloaded_model:
            continue
        resident_memory, _ = current_loaded_models[i].model.model_mmap_residency(free=True)
        if resident_memory > 0:
            logger.debug(f"RAM Unloading {current_loaded_models[i].model.model.__class__.__name__}")

    for i in sorted(unloaded_model, reverse=True):
        unloaded_models.append(current_loaded_models.pop(i))

    if not for_dynamic and pins_required > 0:
        ensure_pin_budget(pins_required)
        ensure_pin_registerable(pins_required)

    if len(unloaded_model) > 0:
        soft_empty_cache()
    elif device is not None:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()
    return unloaded_models


@tracer.start_as_current_span("Load Models GPU")
def load_models_gpu(models: Sequence[ModelManageable], memory_required: int = 0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False) -> None:
    span = get_current_span()
    if memory_required != 0:
        span.set_attribute("memory_required", memory_required)
    # todo: if enables_dynamic_vram(), use torch.inference_mode() contextlib
    # todo: if enables_dynamic_vram(), _load_models_gpu() should be called on a worker pool, because it needs its own torch mempool that aimdo doesn't see
    #       in this scenario, we want to join on the task, it's not clear if it's an issue if the thread gets reused
    # todo: implement a similar logic to
    # def load_models_gpu_thread(models, memory_required, force_patch_weights, minimum_memory_required, force_full_load):
    #     with torch.inference_mode():
    #         load_models_gpu_orig(models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)
    #         soft_empty_cache()
    #
    # def load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
    #     #Deliberately load models outside of the Aimdo mempool so they can be retained accross
    #     #nodes. Use a dummy thread to do it as pytorch documents that mempool contexts are
    #     #thread local. So exploit that to escape context
    #     if enables_dynamic_vram():
    #         t = threading.Thread(
    #             target=load_models_gpu_thread,
    #             args=(models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)
    #         )
    #         t.start()
    #         t.join()
    #     else:
    #         load_models_gpu_orig(models, memory_required=memory_required, force_patch_weights=force_patch_weights,
    #                              minimum_memory_required=minimum_memory_required, force_full_load=force_full_load)
    # todo: this requires the merge, you have to deal with this issue.
    with model_management_lock:
        _load_models_gpu(models, memory_required, force_patch_weights, minimum_memory_required, force_full_load)
        to_load = list(map(str, models))
        span.set_attribute("models", to_load)
        logger.debug(f"Loaded {to_load}")


def prepare_device_model_loads(
    total_memory_required,
    *,
    extra_mem,
    minimum_memory_required,
    free_for_dynamic,
    total_pins_required=None,
    total_ram_required=None,
):
    """Apply the normal pressure-driven load preparation on every target device."""
    total_pins_required = total_pins_required or {}
    total_ram_required = total_ram_required or {}
    models_freed = []
    for device, required in total_memory_required.items():
        if device == torch.device("cpu"):
            continue
        models_freed += free_memory(
            required * 1.1 + extra_mem,
            device,
            for_dynamic=free_for_dynamic,
            pins_required=total_pins_required.get(device, 0),
            ram_required=total_ram_required.get(device, 0),
        )

    # Aimdo's VBAR policy accounts PyTorch allocator reservations as pressure.
    # Flush inactive caching-allocator blocks before prioritizing the next
    # dynamic model, otherwise a preceding text encoder can permanently lower
    # the diffusion model's watermark despite physically free VRAM.
    if (
        free_for_dynamic
        and memory_management.aimdo_enabled
        and any(device != torch.device("cpu") for device in total_memory_required)
    ):
        _soft_empty_cache(force=True)

    for device in total_memory_required:
        if device == torch.device("cpu"):
            continue
        if get_free_memory(device) < minimum_memory_required:
            models_l = free_memory(
                minimum_memory_required,
                device,
                for_dynamic=free_for_dynamic,
            )
            models_freed += models_l
            logger.debug("{} models unloaded.".format(len(models_l)))
    return models_freed


def _load_models_gpu(models: Sequence[ModelManageable], memory_required: int = 0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False) -> None:
    cleanup_models_gc()
    global vram_state

    requested_memory_required = memory_required
    requested_minimum_memory_required = minimum_memory_required
    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
    if minimum_memory_required is None:
        minimum_memory_required = extra_mem
    else:
        minimum_memory_required = max(inference_memory, minimum_memory_required + extra_reserved_memory())

    # Order-preserving dedup. A plain set() would randomize iteration order across runs
    models_temp = {}
    for m in models:
        models_temp[m] = None
        for mm in m.model_patches_models():
            models_temp[mm] = None

    models = list(models_temp)
    models.reverse()

    models_to_load = []
    models_freed = []

    free_for_dynamic = True
    for x in models:
        if not x.is_dynamic():
            free_for_dynamic = False
        loaded_model = LoadedModel(x)
        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models[loaded_model_index]
            loaded.currently_used = True
            models_to_load.append(loaded)
        else:
            models_to_load.append(loaded_model)

    for loaded_model in models_to_load:
        to_unload = []
        for i in range(len(current_loaded_models)):
            if loaded_model.model.is_clone(current_loaded_models[i].model):
                to_unload = [i] + to_unload
        for i in to_unload:
            model_to_unload = current_loaded_models.pop(i)
            model_to_unload.model.detach(unpatch_all=False)
            model_to_unload.model_finalizer.detach()

    total_memory_required = {}
    total_pins_required = {}
    total_ram_required = {}
    for loaded_model in models_to_load:
        if getattr(loaded_model.model, "manages_own_device_memory", lambda: False)():
            continue
        device = loaded_model.device
        total_memory_required[device] = total_memory_required.get(device, 0) + loaded_model.model_memory_required(device)
        resident_memory, model_memory = loaded_model.model.model_mmap_residency()
        pinned_memory = loaded_model.model.pinned_memory_size()
        # FIXME: This can over-free the pins as it budgets to pin the entire model. We should
        # make this JIT to keep as much pinned as possible.
        pins_required = model_memory - pinned_memory
        ram_required = model_memory - resident_memory
        total_pins_required[device] = total_pins_required.get(device, 0) + pins_required
        total_ram_required[device] = total_ram_required.get(device, 0) + ram_required

    models_freed += prepare_device_model_loads(
        total_memory_required,
        extra_mem=extra_mem,
        minimum_memory_required=minimum_memory_required,
        free_for_dynamic=free_for_dynamic,
        total_pins_required=total_pins_required,
        total_ram_required=total_ram_required,
    )

    newly_loaded = []
    try:
        for loaded_model in models_to_load:
            model = loaded_model.model
            torch_dev = model.load_device
            if is_device_cpu(torch_dev):
                vram_set_state = VRAMState.DISABLED
            else:
                vram_set_state = vram_state
            lowvram_model_memory = 0
            self_managed_device = getattr(
                model,
                "manages_own_device_memory",
                lambda: False,
            )()
            if self_managed_device:
                # The remote rank applies its own DynamicVRAM pressure decision.
                set_inference_memory_requirements = getattr(
                    model,
                    "set_inference_memory_requirements",
                    None,
                )
                if set_inference_memory_requirements is not None:
                    set_inference_memory_requirements(
                        requested_memory_required,
                        requested_minimum_memory_required,
                        force_full_load,
                    )
                lowvram_model_memory = 0
            elif lowvram_available and (vram_set_state == VRAMState.LOW_VRAM or vram_set_state == VRAMState.NORMAL_VRAM) and not force_full_load:
                loaded_memory = loaded_model.model_loaded_memory()
                current_free_mem = get_free_memory(torch_dev) + loaded_memory

                lowvram_model_memory = max(0, (current_free_mem - minimum_memory_required), min(current_free_mem * MIN_WEIGHT_MEMORY_RATIO, current_free_mem - minimum_inference_memory()))
                lowvram_model_memory = lowvram_model_memory - loaded_memory

                if lowvram_model_memory == 0:
                    lowvram_model_memory = 0.1

            if not self_managed_device and vram_set_state == VRAMState.NO_VRAM:
                lowvram_model_memory = 0.1

            was_loaded = loaded_model in current_loaded_models
            loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
            vram_used = 0 if is_device_cpu(torch_dev) else loaded_model.model_loaded_memory()
            ram_used = model.loaded_ram_size() if model.is_dynamic() else loaded_model.model_memory() - vram_used
            detail("Model loaded: patcher=%s model=%s ram_mb=%.1f vram_mb=%.1f", model.__class__.__name__, model.model.__class__.__name__, ram_used / (1024 ** 2), vram_used / (1024 ** 2))
            if was_loaded:
                current_loaded_models.remove(loaded_model)
            current_loaded_models.insert(0, loaded_model)
            if not was_loaded:
                newly_loaded.append(loaded_model)
    except Exception:
        # A load request can contain a group of cooperating models on different
        # devices. Keep that request atomic: if a later model fails, do not leave
        # earlier members resident with only part of the group usable.
        _rollback_newly_loaded_models(newly_loaded)
        raise

    span = get_current_span()
    span.set_attribute("models_to_load", list(map(str, models_to_load)))
    span.set_attribute("models_freed", list(map(str, models_freed)))


def _rollback_newly_loaded_models(newly_loaded: Sequence[LoadedModel]) -> None:
    for loaded_model in reversed(newly_loaded):
        if loaded_model in current_loaded_models:
            current_loaded_models.remove(loaded_model)
        try:
            loaded_model.model_unload()
        except Exception:
            logger.exception("Failed to unload a model while rolling back a grouped load")


@_deprecate_method(message="Use load_models_gpu instead", version="0.0.2")
def load_model_gpu(model):
    return load_models_gpu([model])


def loaded_models(only_currently_used=False):
    with model_management_lock:
        output = []
        for m in current_loaded_models:
            if only_currently_used:
                if not m.currently_used:
                    continue

            output.append(m.model)
        return output


def cleanup_models_gc():
    do_gc = False

    for i in range(len(current_loaded_models)):
        cur = current_loaded_models[i]
        if cur.is_dead():
            logger.info("Potential memory leak detected with model {}, doing a full garbage collect, for maximum performance avoid circular references in the model code.".format(cur.real_model().__class__.__name__))
            do_gc = True
            break

    if do_gc:
        gc.collect()
        soft_empty_cache()

        for i in range(len(current_loaded_models)):
            cur = current_loaded_models[i]
            if cur.is_dead():
                logger.warning("WARNING, memory leak with model {}. Please make sure it is not being referenced from somewhere.".format(cur.real_model().__class__.__name__))


def archive_model_dtypes(model):
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            setattr(module, f"{param_name}_comfy_model_dtype", param.dtype)
        for buf_name, buf in module.named_buffers(recurse=False):
            setattr(module, f"{buf_name}_comfy_model_dtype", buf.dtype)


def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if current_loaded_models[i].real_model() is None:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        del x


def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except:  # Old pytorch doesn't have .itemsize
            pass
    return dtype_size


def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")


def unet_initial_load_device(parameters, dtype):
    cpu_dev = torch.device("cpu")
    if memory_management.aimdo_enabled:
        return cpu_dev
    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.SHARED:
        return torch_dev

    if DISABLE_SMART_MEMORY or vram_state == VRAMState.NO_VRAM:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev


def maximum_vram_for_weights(device=None) -> int:
    return get_total_memory(device) * 0.88 - minimum_inference_memory()


def unet_dtype(device=None, model_params=0, supported_dtypes=(torch.float16, torch.bfloat16, torch.float32), weight_dtype: Optional[torch.dtype] = None):
    if model_params < 0:
        model_params = 1000000000000000000000
    if args.fp32_unet:
        return torch.float32
    if args.fp64_unet:
        return torch.float64
    if args.bf16_unet:
        return torch.bfloat16
    if args.fp16_unet:
        return torch.float16
    if args.fp8_e4m3fn_unet:
        return torch.float8_e4m3fn
    if args.fp8_e5m2_unet:
        return torch.float8_e5m2
    if args.fp8_e8m0fnu_unet:
        return torch.float8_e8m0fnu

    fp8_dtype = None
    if weight_dtype in FLOAT8_TYPES:
        fp8_dtype = weight_dtype

    if fp8_dtype is not None:
        if args.fp8_storage and supports_fp8_storage(device, fp8_dtype):
            return fp8_dtype

    if PRIORITIZE_FP16 or weight_dtype == torch.float16:
        if torch.float16 in supported_dtypes and should_use_fp16(device=device, model_params=model_params):
            return torch.float16

    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params):
            if torch.float16 in supported_dtypes:
                return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params):
            if torch.bfloat16 in supported_dtypes:
                return torch.bfloat16

    for dt in supported_dtypes:
        if dt == torch.float16 and should_use_fp16(device=device, model_params=model_params, manual_cast=True):
            if torch.float16 in supported_dtypes:
                return torch.float16
        if dt == torch.bfloat16 and should_use_bf16(device, model_params=model_params, manual_cast=True):
            if torch.bfloat16 in supported_dtypes:
                return torch.bfloat16

    return torch.float32


# None means no manual cast
def unet_manual_cast(weight_dtype, inference_device, supported_dtypes=(torch.float16, torch.bfloat16, torch.float32)):
    if weight_dtype == torch.float32 or weight_dtype == torch.float64:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=False)
    if fp16_supported and weight_dtype == torch.float16:
        return None

    bf16_supported = should_use_bf16(inference_device)
    if bf16_supported and weight_dtype == torch.bfloat16:
        return None

    fp16_supported = should_use_fp16(inference_device, prioritize_performance=True)
    if PRIORITIZE_FP16 and fp16_supported and torch.float16 in supported_dtypes:
        return torch.float16

    for dt in supported_dtypes:
        if dt == torch.float16 and fp16_supported:
            return torch.float16
        if dt == torch.bfloat16 and bf16_supported:
            return torch.bfloat16

    return torch.float32


def text_encoder_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")


def text_encoder_device():
    if args.gpu_only:
        return get_torch_device()
    if memory_management.aimdo_enabled:
        return get_torch_device()
    elif vram_state in (VRAMState.HIGH_VRAM, VRAMState.NORMAL_VRAM):
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def text_encoder_initial_device(load_device, offload_device, model_size=0):
    if memory_management.aimdo_enabled:
        return offload_device

    if load_device == offload_device or model_size <= 1024 * 1024 * 1024:
        return offload_device

    if is_device_mps(load_device):
        return load_device

    mem_l = get_free_memory(load_device)
    mem_o = get_free_memory(offload_device)
    if mem_l > (mem_o * 0.5) and model_size * 1.2 < mem_l:
        return load_device
    else:
        return offload_device


def text_encoder_dtype(device=None):
    if args.fp8_e4m3fn_text_enc:
        return torch.float8_e4m3fn
    elif args.fp8_e5m2_text_enc:
        return torch.float8_e5m2
    elif args.fp16_text_enc:
        return torch.float16
    elif args.bf16_text_enc:
        return torch.bfloat16
    elif args.fp32_text_enc:
        return torch.float32

    if is_device_cpu(device):
        return torch.float16

    return torch.float16


def intermediate_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def intermediate_dtype():
    if args.fp16_intermediates:
        return torch.float16
    else:
        return torch.float32


def vae_device():
    if args.cpu_vae:
        return torch.device("cpu")
    return get_torch_device()


def vae_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")


def vae_dtype(device=None, allowed_dtypes=[]):
    if args.fp16_vae:
        return torch.float16
    elif args.bf16_vae:
        return torch.bfloat16
    elif args.fp32_vae:
        return torch.float32

    for d in allowed_dtypes:
        if d == torch.float16 and should_use_fp16(device):
            return d

        if d == torch.bfloat16 and should_use_bf16(device):
            return d

    return torch.float32


def get_autocast_device(dev):
    if hasattr(dev, 'type'):
        return dev.type
    return "cuda"


def supports_dtype(device, dtype):  # TODO
    if dtype == torch.float32:
        return True
    if is_device_cpu(device):
        return False
    if dtype == torch.float16:
        return True
    if dtype == torch.bfloat16:
        return True
    return False


def supports_cast(device, dtype):  # TODO
    if dtype == torch.float32:
        return True
    if dtype == torch.float16:
        return True
    if directml_device:  # TODO: test this
        return False
    if dtype == torch.bfloat16:
        return True
    if is_device_mps(device):
        return False
    if dtype == torch.float8_e4m3fn:
        return True
    if dtype == torch.float8_e5m2:
        return True
    return False


def _device_cache_key(device):
    device = torch.device(device) if device is not None else get_torch_device()
    return (device.type, device.index)


def supports_fp8_storage(device=None, dtype=None):
    if dtype not in FLOAT8_TYPES:
        return False
    device_type, device_index = _device_cache_key(device)
    return _supports_fp8_storage(device_type, device_index, dtype)


@lru_cache(maxsize=None)
def _supports_fp8_storage(device_type, device_index, dtype):
    device = torch.device(device_type, device_index)
    compute_dtype = torch.bfloat16 if should_use_bf16(device) else torch.float16

    try:
        class NoInitLinear(torch.nn.Linear):
            def reset_parameters(self):
                return None

        layer = NoInitLinear(4, 4, device=device, dtype=dtype)
        state_dict = {
            "weight": torch.randn(4, 4, dtype=torch.float32).to(dtype).to(device),
            "bias": torch.randn(4, dtype=torch.float32).to(dtype).to(device),
        }
        layer.load_state_dict(state_dict, strict=True)
        x = torch.randn(2, 4, dtype=torch.float32, device=device).to(compute_dtype)
        _ = torch.nn.functional.linear(x, layer.weight.to(dtype=compute_dtype), layer.bias.to(dtype=compute_dtype))
        return True
    except Exception:
        return False


def pick_weight_dtype(dtype, fallback_dtype, device=None):
    if dtype is None:
        dtype = fallback_dtype
    elif dtype_size(dtype) > dtype_size(fallback_dtype):
        dtype = fallback_dtype

    if not supports_cast(device, dtype):
        dtype = fallback_dtype

    return dtype


def device_supports_non_blocking(device):
    if torch.jit.is_tracing() or torch.jit.is_scripting():
        return True
    if args.force_non_blocking:
        return True
    if is_device_mps(device):
        return False  # pytorch bug? mps doesn't support non blocking
    if is_intel_xpu():  # xpu does support non blocking but it is slower on iGPUs for some reason so disable by default until situation changes
        return False
    if args.deterministic:  # TODO: figure out why deterministic breaks non blocking from gpu to cpu (previews)
        return False
    if directml_device:
        return False
    return True


def force_channels_last():
    if args.force_channels_last:
        return True

    # TODO
    return False


STREAMS = {}
NUM_STREAMS = 0
if args.async_offload is not None:
    NUM_STREAMS = args.async_offload
else:
    #  Enable by default on Nvidia and AMD
    if is_nvidia() or is_amd():
        NUM_STREAMS = 4

if args.disable_async_offload:
    NUM_STREAMS = 0

if NUM_STREAMS > 0:
    logger.debug("Using async weight offloading with {} streams".format(NUM_STREAMS))


def current_stream(device):
    if device is None:
        return None
    if is_device_cuda(device):
        return torch.cuda.current_stream()
    elif is_device_xpu(device):
        return torch.xpu.current_stream()
    elif is_device_npu(device):
        return torch.npu.current_stream(device)
    else:
        return None


stream_counters = {}

STREAM_CAST_BUFFERS = {}
LARGEST_CASTED_WEIGHT = (None, 0)
STREAM_AIMDO_CAST_BUFFERS = {}
LARGEST_AIMDO_CASTED_WEIGHT = (None, 0)
CROSS_STEP_STATE = weakref.WeakSet()

DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE = 16 * 1024 ** 3

# NOTE: devs/agents: this is temporary and will be removed in a future comfy. Not supported for custom node use.
def _register_cross_step(module):
    CROSS_STEP_STATE.add(module)

def get_cast_buffer(offload_stream, device, size, ref):
    global LARGEST_CASTED_WEIGHT

    if offload_stream is not None:
        wf_context = offload_stream
        if hasattr(wf_context, "as_context"):
            wf_context = wf_context.as_context(offload_stream)
    else:
        wf_context = nullcontext()

    cast_buffer = STREAM_CAST_BUFFERS.get(offload_stream, None)
    if cast_buffer is None or cast_buffer.numel() < size:
        if ref is LARGEST_CASTED_WEIGHT[0]:
            # If there is one giant weight we do not want both streams to
            # allocate a buffer for it. It's up to the caster to get the other
            # offload stream in this corner case
            return None
        if cast_buffer is not None and cast_buffer.numel() > 50 * (1024 ** 2):
            # I want my wrongly sized 50MB+ of VRAM back from the caching allocator right now
            synchronize()
            del STREAM_CAST_BUFFERS[offload_stream]
            del cast_buffer
            soft_empty_cache()
        with wf_context:
            cast_buffer = torch.empty((size), dtype=torch.int8, device=device)
            STREAM_CAST_BUFFERS[offload_stream] = cast_buffer

        if size > LARGEST_CASTED_WEIGHT[1]:
            LARGEST_CASTED_WEIGHT = (ref, size)

    return cast_buffer

def get_aimdo_cast_buffer(offload_stream, device):
    cast_buffer = STREAM_AIMDO_CAST_BUFFERS.get(offload_stream, None)
    if cast_buffer is None:
        cast_buffer = comfy_aimdo.vram_buffer.VRAMBuffer(DEFAULT_AIMDO_CAST_BUFFER_RESERVATION_SIZE, device.index)
        STREAM_AIMDO_CAST_BUFFERS[offload_stream] = cast_buffer
    return cast_buffer

def reset_cast_buffers():
    global LARGEST_CASTED_WEIGHT
    global LARGEST_AIMDO_CASTED_WEIGHT

    LARGEST_CASTED_WEIGHT = (None, 0)
    LARGEST_AIMDO_CASTED_WEIGHT = (None, 0)
    for offload_stream in set(STREAM_CAST_BUFFERS) | set(STREAM_AIMDO_CAST_BUFFERS):
        if offload_stream is not None:
            offload_stream.synchronize()
    synchronize()

    for mmap_obj in DIRTY_MMAPS:
        mmap_obj.bounce()
    DIRTY_MMAPS.clear()

    for module in CROSS_STEP_STATE:
        del module._comfy_cross_step_state
    CROSS_STEP_STATE.clear()

    for loaded_model in current_loaded_models:
        model = loaded_model.model
        if model is not None and model.is_dynamic():
            model.reset_dynamic_buffers()

    STREAM_CAST_BUFFERS.clear()
    STREAM_AIMDO_CAST_BUFFERS.clear()
    soft_empty_cache()


def get_offload_stream(device):
    stream_counter = stream_counters.get(device, 0)
    if NUM_STREAMS == 0:
        return None

    if hasattr(torch.compiler, "is_compiling") and torch.compiler.is_compiling():
        return None

    if device in STREAMS:
        ss = STREAMS[device]
        # Sync the oldest stream in the queue with the current stream, then
        # return the next stream so offload copies can run ahead of compute.
        ss[stream_counter].wait_stream(current_stream(device))
        stream_counter = (stream_counter + 1) % len(ss)
        stream_counters[device] = stream_counter
        return ss[stream_counter]
    elif is_device_cuda(device):
        ss = []
        for k in range(NUM_STREAMS):
            s1 = torch.cuda.Stream(device=device, priority=0)
            s1.as_context = torch.cuda.stream
            ss.append(s1)
        STREAMS[device] = ss
        s = ss[stream_counter]
        stream_counters[device] = stream_counter
        return s
    elif is_device_xpu(device):
        ss = []
        for k in range(NUM_STREAMS):
            s1 = torch.xpu.Stream(device=device, priority=0)
            s1.as_context = torch.xpu.stream
            ss.append(s1)
        STREAMS[device] = ss
        s = ss[stream_counter]
        stream_counters[device] = stream_counter
        return s
    elif is_device_npu(device):
        ss = []
        # Match the CUDA/XPU stream interface used by the shared offload path.
        # torch.npu.stream provides the context manager for torch-npu streams.
        for k in range(NUM_STREAMS):
            s1 = torch.npu.Stream(device=device, priority=0)
            s1.as_context = torch.npu.stream
            ss.append(s1)
        STREAMS[device] = ss
        s = ss[stream_counter]
        stream_counters[device] = stream_counter
        return s
    return None


def sync_stream(device, stream):
    if stream is None or current_stream(device) is None:
        return
    current_stream(device).wait_stream(stream)


def record_stream_for_tensor(tensor, stream):
    if tensor is None or stream is None:
        return
    if isinstance(tensor, quant_ops.QuantizedTensor):
        record_stream_for_tensor(tensor._qdata, stream)
        return
    if not isinstance(tensor, torch.Tensor):
        return
    if tensor.device.type not in ("cuda", "xpu"):
        return
    record_stream = getattr(tensor, "record_stream", None)
    if record_stream is None:
        return
    record_stream(stream)


def cast_to_gathered(tensors, r, non_blocking=False, stream=None, r2=None, target_geometries=None):
    wf_context = nullcontext()
    if stream is not None:
        wf_context = stream
        if hasattr(wf_context, "as_context"):
            wf_context = wf_context.as_context(stream)

    geometries = target_geometries if target_geometries is not None else tensors
    dest_views = memory_management.interpret_gathered_like(geometries, r) if r is not None else [None] * len(tensors)
    dest2_views = memory_management.interpret_gathered_like(geometries, r2) if r2 is not None else None
    with wf_context:
        for tensor in tensors:
            dest_view = dest_views.pop(0)
            dest2_view = dest2_views.pop(0) if dest2_views is not None else None
            if tensor is None:
                continue
            probe_view = dest_view if dest_view is not None else dest2_view
            if probe_view is None:
                can_raw_read = False
            elif isinstance(tensor, quant_ops.QuantizedTensor):
                can_raw_read = isinstance(probe_view, quant_ops.QuantizedTensor)
            else:
                can_raw_read = tensor.shape == probe_view.shape and tensor.dtype == probe_view.dtype
            if can_raw_read and memory_management.read_tensor_file_slice_into(tensor, dest_view, stream=stream, destination2=dest2_view):
                record_stream_for_tensor(dest_view, stream)
                record_stream_for_tensor(dest2_view, stream)
                continue
            storage = tensor._qdata.untyped_storage() if isinstance(tensor, quant_ops.QuantizedTensor) else tensor.untyped_storage()
            mark_mmap_dirty(storage)
            copy_src = tensor
            if isinstance(copy_src, quant_ops.QuantizedTensor) and probe_view is not None and not isinstance(probe_view, quant_ops.QuantizedTensor):
                copy_src = copy_src.dequantize().to(dtype=probe_view.dtype)
            if dest_view is not None:
                dest_view.copy_(copy_src, non_blocking=non_blocking)
            if dest2_view is not None:
                dest2_view.copy_(copy_src if dest_view is None else dest_view, non_blocking=non_blocking)
            record_stream_for_tensor(copy_src, stream)
            record_stream_for_tensor(dest_view, stream)
            record_stream_for_tensor(dest2_view, stream)


def tensor_materialization_geometry(tensor, dtype=None):
    if tensor is None:
        return None
    if isinstance(tensor, quant_ops.QuantizedTensor):
        if dtype is not None:
            return memory_management.TensorGeometry(shape=tensor.shape, dtype=dtype)
        return tensor
    return memory_management.TensorGeometry(shape=tensor.shape, dtype=dtype or tensor.dtype)


def tensor_materialization_size(tensor, dtype=None) -> int:
    geometry = tensor_materialization_geometry(tensor, dtype=dtype)
    if geometry is None:
        return 0
    return memory_management.vram_aligned_size(geometry)


def cast_to(weight, dtype=None, device=None, non_blocking=False, copy=False, stream=None, r=None):
    if device is None or weight.device == device:
        if not copy:
            if dtype is None or weight.dtype == dtype:
                return weight
        if stream is not None:
            wf_context = stream
            if hasattr(wf_context, "as_context"):
                wf_context = wf_context.as_context(stream)
            with wf_context:
                result = weight.to(dtype=dtype, copy=copy)
            record_stream_for_tensor(weight, stream)
            record_stream_for_tensor(result, stream)
            return result
        return weight.to(dtype=dtype, copy=copy)

    if stream is not None:
        wf_context = stream
        if hasattr(wf_context, "as_context"):
            wf_context = wf_context.as_context(stream)
        with wf_context:
            if r is None:
                r = torch.empty_like(weight, dtype=dtype, device=device)
            r.copy_(weight, non_blocking=non_blocking)
        record_stream_for_tensor(weight, stream)
        record_stream_for_tensor(r, stream)
    else:
        if r is None:
            r = torch.empty_like(weight, dtype=dtype, device=device)
        r.copy_(weight, non_blocking=non_blocking)
    return r


def cast_to_device(tensor, device, dtype, copy=False):
    non_blocking = device_supports_non_blocking(device)
    return cast_to(tensor, dtype=dtype, device=device, non_blocking=non_blocking, copy=copy)


PINNED_MEMORY = {}
TOTAL_PINNED_MEMORY = 0
MAX_PINNED_MEMORY = -1

# Pinned pages cannot be reclaimed or swapped, and CUDA host registrations do
# not appear in the kernel's Mlocked/VmPin counters, so nothing but this budget
# keeps pinning from starving the host (or the container) into a livelock or an
# OOM kill. The reserve is the part of the effective memory limit pinning never
# takes: max(4 GiB, 10% of the limit). Those are the constants upstream ComfyUI
# already reserved (its min(0.9 * ram, ram - 4 GiB) ceiling); here they are
# applied to the cgroup-aware limit and, at every pin, to memory available now,
# rather than to total host RAM. The reserve has to hold the file-backed
# working set of everything else (executables, the desktop, the page cache of
# the model files being streamed), which MemAvailable counts as available, and
# whatever other processes allocate after a pin is made.
PINNED_MEMORY_RESERVE_MIN = 4 * 1024 ** 3
PINNED_MEMORY_RESERVE_FRACTION = 10
WINDOWS_PINNED_MEMORY_FRACTION = 0.40  # the Windows driver limit is apparently 50%


def pinned_memory_reserve() -> int:
    """Bytes of the effective memory limit that pinning always leaves free."""
    if args.pinned_memory_reserve is not None:
        return max(0, int(args.pinned_memory_reserve * 1024 ** 3))
    return max(PINNED_MEMORY_RESERVE_MIN, system_memory.virtual_memory_total() // PINNED_MEMORY_RESERVE_FRACTION)


def compute_max_pinned_memory() -> int:
    """Ceiling on bytes registered at once: the effective memory limit (host
    RAM, or the lowest cgroup memory.max / memory.high / memory.limit_in_bytes
    of this process's cgroup and its ancestors) minus the reserve.

    This bounds the total; each pin is additionally checked against memory
    available at that moment by ``ensure_pin_budget``."""
    limit = system_memory.virtual_memory_total()
    ceiling = limit - pinned_memory_reserve()
    if WINDOWS:
        ceiling = min(ceiling, int(limit * WINDOWS_PINNED_MEMORY_FRACTION))
    if args.max_pinned_memory is not None:
        ceiling = min(ceiling, int(args.max_pinned_memory * 1024 ** 3))
    return max(0, ceiling)


def is_integrated_gpu():
    device = get_torch_device()
    return device.type == "cuda" and bool(torch.cuda.get_device_properties(device).is_integrated)

DISABLE_PINNED_MEMORY = args.disable_pinned_memory
if not DISABLE_PINNED_MEMORY and is_integrated_gpu():
    # Integrated GPU VRAM is carved out of system RAM, so pinning host memory only takes RAM from the GPU.
    DISABLE_PINNED_MEMORY = True

if not DISABLE_PINNED_MEMORY:
    if is_nvidia() or is_amd():
        MAX_PINNED_MEMORY = compute_max_pinned_memory()
        logger.info(
            "Enabled pinned memory: ceiling %d MB of %d MB effective memory limit, reserve %d MB, %d MB available now",
            MAX_PINNED_MEMORY // (1024 * 1024),
            system_memory.virtual_memory_total() // (1024 * 1024),
            pinned_memory_reserve() // (1024 * 1024),
            system_memory.virtual_memory_available() // (1024 * 1024),
        )

PINNING_ALLOWED_TYPES = set(["Tensor", "Parameter", "QuantizedTensor"])

_UNSHARED_MAX_PINNED_MEMORY = None


def share_pinned_memory_budget(participants: int) -> None:
    """Split this host's pinned memory budget across model-parallel processes.

    Every pipeline or tensor parallel rank is a separate process that would
    otherwise register up to the whole per-host budget; the sum can exceed
    physical RAM, and pinned pages cannot be reclaimed.
    """
    global MAX_PINNED_MEMORY, _UNSHARED_MAX_PINNED_MEMORY
    if MAX_PINNED_MEMORY <= 0 or participants <= 1:
        return
    if _UNSHARED_MAX_PINNED_MEMORY is None:
        _UNSHARED_MAX_PINNED_MEMORY = MAX_PINNED_MEMORY
    MAX_PINNED_MEMORY = _UNSHARED_MAX_PINNED_MEMORY // participants
    logger.info("Pinned memory budget shared across %d model-parallel processes: %d MB each", participants, MAX_PINNED_MEMORY // (1024 * 1024))


def pinned_hostbuf_size(size):
    if args.high_ram:
        return max(0, int(size * 2))
    return max(0, int(min(size, MAX_PINNED_MEMORY) * 2))

def discard_cuda_async_error():
    try:
        a = torch.tensor([1], dtype=torch.uint8, device=get_torch_device())
        b = torch.tensor([1], dtype=torch.uint8, device=get_torch_device())
        _ = a + b
        synchronize()
    except RuntimeError:
        # Dump it! We already know about it from the synchronous return
        pass


def pin_memory(tensor, evict_active=True):
    global TOTAL_PINNED_MEMORY
    if MAX_PINNED_MEMORY <= 0:
        return False

    if type(tensor).__name__ not in PINNING_ALLOWED_TYPES:
        return False

    if not is_device_cpu(tensor.device):
        return False

    if tensor.is_pinned():
        # NOTE: Cuda does detect when a tensor is already pinned and would
        # error below, but there are proven cases where this also queues an error
        # on the GPU async. So dont trust the CUDA API and guard here
        return False

    if not tensor.is_contiguous():
        return False

    size = tensor.nbytes
    memory_management.extra_ram_release(memory_management.RAM_CACHE_HEADROOM)
    if not ensure_pin_budget(size, evict_active=evict_active):
        return False
    if not ensure_pin_registerable(size, evict_active=evict_active):
        return False

    ptr = tensor.data_ptr()
    if ptr == 0:
        return False

    if torch.cuda.cudart().cudaHostRegister(ptr, size, 1) == 0:
        PINNED_MEMORY[ptr] = size
        TOTAL_PINNED_MEMORY += size
        return True
    else:
        logger.warning("Pin error.")
        discard_cuda_async_error()

    return False


def unpin_memory(tensor):
    global TOTAL_PINNED_MEMORY
    if MAX_PINNED_MEMORY <= 0:
        return False

    if not is_device_cpu(tensor.device):
        return False

    ptr = tensor.data_ptr()
    size = tensor.nbytes

    size_stored = PINNED_MEMORY.get(ptr, None)
    if size_stored is None:
        logger.warning("Tried to unpin tensor not pinned by ComfyUI")
        return False

    if size != size_stored:
        logger.warning("Size of pinned tensor changed")
        return False

    if torch.cuda.cudart().cudaHostUnregister(ptr) == 0:
        size = PINNED_MEMORY.pop(ptr)
        TOTAL_PINNED_MEMORY -= size
        return True
    else:
        logger.warning("Unpin error.")
        discard_cuda_async_error()

    return False


def sage_attention_enabled():
    return args.use_sage_attention

def comfy_kitchen_attention_enabled():
    return args.use_ck_attention

def flash_attention_enabled():
    return args.use_flash_attention


def xformers_enabled():
    global directml_device
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if is_ascend_npu():
        return False
    if is_mlu():
        return False
    if is_ixuca():
        return False
    if directml_device:
        return False
    return XFORMERS_IS_AVAILABLE


def flash_attn_enabled():
    global directml_device
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if directml_device:
        return False
    return flash_attention_enabled()


def xformers_enabled_vae():
    enabled = xformers_enabled()
    if not enabled:
        return False

    return XFORMERS_ENABLED_VAE


def pytorch_attention_enabled():
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION


def pytorch_attention_enabled_vae():
    if is_amd():
        return False  # enabling pytorch attention on AMD currently causes crash when doing high res
    return pytorch_attention_enabled()


def pytorch_attention_flash_attention():
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        # TODO: more reliable way of checking for flash attention?
        if is_nvidia():
            return True
        if is_intel_xpu():
            return True
        if is_ascend_npu():
            return True
        if is_mlu():
            return True
        if is_amd():
            return True  # if you have pytorch attention enabled on AMD it probably supports at least mem efficient attention
        if is_ixuca():
            return True
    return False


def force_upcast_attention_dtype():
    upcast = args.force_upcast_attention

    macos_version = mac_version()
    if macos_version is not None and ((14, 5) <= macos_version):  # black image bug on recent versions of macOS, I don't think it's ever getting fixed
        upcast = True

    if upcast:
        return {torch.float16: torch.float32}
    else:
        return None


#Developers and agents: You almost never want to call this function from Model code as it does
#not account for ComfyUIs smart memory feature combining with Dynamic VRAM, where inactive models
#are preserved in VRAM right up until there is higher priority demand (I.E whatever you want to do
#that makes you meansure VRAM from model code). Instead call get_free_memory() on the ModelPatcher
#for your BaseModel object (.current_patcher) instead to count this VRAM as free and then Dynamic
#VRAM will evict that extra VRAM for you when you use it.
def get_free_memory(dev=None, torch_free_too=False):
    global directml_device
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = system_memory.virtual_memory_available()
        mem_free_torch = mem_free_total
    else:
        if directml_device:
            mem_free_total = 1024 * 1024 * 1024  # TODO
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_xpu + mem_free_torch
        elif is_ascend_npu():
            stats = torch.npu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_npu, _ = torch.npu.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_npu + mem_free_torch
        elif is_mlu():
            stats = torch.mlu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_mlu, _ = torch.mlu.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_mlu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total


def cpu_mode():
    global cpu_state
    return cpu_state == CPUState.CPU


def mps_mode():
    global cpu_state
    return cpu_state == CPUState.MPS


def is_device_type(device, type):
    if hasattr(device, 'type'):
        if (device.type == type):
            return True
    return False


def is_device_cpu(device):
    return is_device_type(device, 'cpu')


def is_device_mps(device):
    return is_device_type(device, 'mps')


def is_device_xpu(device):
    return is_device_type(device, 'xpu')


def is_device_cuda(device):
    return is_device_type(device, 'cuda')

def is_device_npu(device):
    return is_device_type(device, 'npu')

def set_torch_device(device):
    """Set the current device for the given torch device. Supports CUDA and XPU."""
    if is_device_cuda(device):
        torch.cuda.set_device(device)
    elif is_device_xpu(device):
        torch.xpu.set_device(device)

def is_directml_enabled():
    global directml_device
    if directml_device is not None:
        return True

    return False


def should_use_fp16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if device is not None:
        if is_device_cpu(device):
            return False

    if args.force_fp16:
        return True

    if FORCE_FP32:
        return False

    if is_directml_enabled():
        return True

    if (device is not None and is_device_mps(device)) or mps_mode():
        return True

    if cpu_mode():
        return False

    if is_intel_xpu():
        return torch.xpu.get_device_properties(device).has_fp16

    if is_ascend_npu():
        return True

    if is_mlu():
        return True

    if is_ixuca():
        return True

    if is_amd():
        return True
    try:
        props = torch.cuda.get_device_properties(device)
        if props.major >= 8:
            return True

        if props.major < 6:
            return False
    except (ValueError, RuntimeError):
        logger.warning("No CUDA devices were present, even though CUDA is available in this torch installation. This assumes the CPU device will be selected for computation")
        return False
    except AssertionError:
        logger.warning("Torch was not compiled with cuda support")
        return False

    # FP16 is confirmed working on a 1080 (GP104) and on latest pytorch actually seems faster than fp32
    nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050", "p40", "p100", "p6", "p4"]
    for x in nvidia_10_series:
        if x in props.name.lower():
            if WINDOWS or manual_cast:
                return True
            else:
                return False  # weird linux behavior where fp32 is faster

    if manual_cast:
        free_model_memory = maximum_vram_for_weights(device)
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    # FP16 is just broken on these cards
    nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True


def should_use_bf16(device=None, model_params=0, prioritize_performance=True, manual_cast=False):
    if FORCE_BF16:
        return True

    if device is not None:
        if is_device_cpu(device):  # TODO ? bf16 works on CPU but is extremely slow
            return False

    if FORCE_FP32:
        return False

    if directml_device:
        return False

    if (device is not None and is_device_mps(device)) or mps_mode():
        return bfloat16_support_mps(device)

    if cpu_mode():
        return False

    if is_intel_xpu():
        return torch.xpu.is_bf16_supported()

    if is_ascend_npu():
        return True

    if is_ixuca():
        return True

    if is_amd():
        arch = torch.cuda.get_device_properties(device).gcnArchName
        if any((a in arch) for a in AMD_RDNA2_AND_OLDER_ARCH):  # RDNA2 and older don't support bf16
            if manual_cast:
                return True
            return False

    try:
        props_major = min(torch.cuda.get_device_properties(torch.device(f"cuda:{i}")).major for i in range(torch.cuda.device_count()))
        if props_major >= 8:
            return True
    except (ValueError, RuntimeError):
        logger.warning("No CUDA devices were present, even though CUDA is available in this torch installation. This assumes the CPU device will be selected for computation")
        return False
    except AssertionError:
        logger.warning("Torch was not compiled with CUDA support")
        return False

    bf16_works = torch.cuda.is_bf16_supported()

    if bf16_works and manual_cast:
        free_model_memory = maximum_vram_for_weights(device)
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    return False


def bfloat16_support_mps(device):
    # test bfloat 16
    try:
        x = torch.ones(1, dtype=torch.bfloat16, device=device)
        x = x + 1.0
        _ = repr(x)
        supported = True
        del x
    except:
        supported = False
    return supported


def supports_fp8_compute(device=None):
    if SUPPORT_FP8_OPS:
        return True

    if not is_nvidia():
        return False

    try:
        props = torch.cuda.get_device_properties(device)
    except (RuntimeError, ValueError, AssertionError):
        return False

    if props.major >= 9:
        return True
    if props.major < 8:
        return False
    if props.minor < 9:
        return False

    if torch_version_numeric < (2, 3):
        return False

    if WINDOWS:
        if torch_version_numeric < (2, 4):
            return False

    return True


def supports_nvfp4_compute(device=None):
    if isinstance(device, torch.device) and device.type != "cuda":
        return False
    if not is_nvidia() or not torch.cuda.is_available():
        return False

    props = torch.cuda.get_device_properties(device)
    if props.major < 10:
        return False

    return True


def supports_mxfp8_compute(device=None):
    if isinstance(device, torch.device) and device.type != "cuda":
        return False
    if not is_nvidia() or not torch.cuda.is_available():
        return False

    if torch_version_numeric < (2, 10):
        return False

    props = torch.cuda.get_device_properties(device)
    if props.major < 10:
        return False

    return True


def supports_int8_compute(device=None):
    """INT8 tensor-core matmul support: NVIDIA sm_75 (Turing) and newer.

    Unlike fp8, this includes all of Ampere (sm_80/sm_86).
    """
    if not is_nvidia():
        return False

    try:
        props = torch.cuda.get_device_properties(device)
    except (RuntimeError, ValueError, AssertionError):
        return False

    return (props.major, props.minor) >= (7, 5)

def supports_fp64(device=None):
    if (device is not None and is_device_mps(device)) or mps_mode():
        return False

    if is_intel_xpu():
        return False

    if is_directml_enabled():
        return False

    if is_ixuca():
        return False

    return True

def extended_fp16_support():
    # TODO: check why some models work with fp16 on newer torch versions but not on older
    if torch_version_numeric < (2, 7):
        return False

    return True


LORA_COMPUTE_DTYPES = {}


def lora_compute_dtype(device):
    dtype = LORA_COMPUTE_DTYPES.get(device, None)
    if dtype is not None:
        return dtype

    if should_use_fp16(device):
        dtype = torch.float16
    else:
        dtype = torch.float32

    LORA_COMPUTE_DTYPES[device] = dtype
    return dtype


def synchronize():
    if cpu_mode():
        return
    if is_intel_xpu():
        torch.xpu.synchronize()
    elif is_ascend_npu():
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def soft_empty_cache(force=False):
    with model_management_lock:
        _soft_empty_cache(force=force)


def _soft_empty_cache(force=False):
    if cpu_mode():
        return
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
    elif is_ascend_npu():
        torch.npu.empty_cache()
    elif is_mlu():
        torch.mlu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def unload_all_models():
    with model_management_lock:
        for device in get_model_management_devices():
            free_memory(1e30, device)
        trim_memory()


def unload_model_and_clones(model: ModelPatcher, unload_additional_models=True, all_devices=False):
    'Unload only model and its clones - primarily for multigpu cloning purposes.'
    with model_management_lock:
        _unload_model_and_clones(model, unload_additional_models=unload_additional_models, all_devices=all_devices)


def _unload_model_and_clones(model: ModelPatcher, unload_additional_models=True, all_devices=False):
    initial_keep_loaded: list[LoadedModel] = current_loaded_models.copy()
    additional_models = []
    if unload_additional_models:
        additional_models = model.get_nested_additional_models()
    keep_loaded = []
    for loaded_model in initial_keep_loaded:
        if loaded_model.model is not None:
            if model.clone_base_uuid == loaded_model.model.clone_base_uuid:
                continue
            # check additional models if they are a match
            skip = False
            for add_model in additional_models:
                if add_model.clone_base_uuid == loaded_model.model.clone_base_uuid:
                    skip = True
                    break
            if skip:
                continue
        keep_loaded.append(loaded_model)
    if not all_devices:
        free_memory(1e30, get_torch_device(), keep_loaded)
    else:
        for device in get_model_management_devices():
            free_memory(1e30, device, keep_loaded)

def debug_memory_summary():
    if is_amd() or is_nvidia():
        return torch.cuda.memory.memory_summary()
    return ""


@_deprecate_method(version="*", message="The comfy.model_management.resolve_lowvram_weight function will be removed soon, please stop using it.")
def resolve_lowvram_weight(weight, model, key):
    return weight


def interrupt_current_processing(value=True):
    interruption.interrupt_current_processing(value)


def processing_interrupted():
    interruption.processing_interrupted()


def throw_exception_if_processing_interrupted():
    interruption.throw_exception_if_processing_interrupted()

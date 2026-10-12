import logging
import importlib

logger = logging.getLogger(__name__)

from . import memory_management
from . import model_management

import comfy_aimdo.control
import comfy_aimdo.host_buffer
import comfy_aimdo.model_vbar
import comfy_aimdo.torch
import comfy_aimdo.vram_buffer
import torch

from .cli_args import args, dynamic_vram_requested, dynamic_vram_supported, enables_dynamic_vram

if dynamic_vram_requested() and not dynamic_vram_supported():
    logger.warning("Unsupported Pytorch detected. DynamicVRAM support requires Pytorch version 2.8 or later (2.12+ is recommended). Falling back to legacy ModelPatcher. VRAM estimates may be unreliable especially on Windows")
    memory_management.aimdo_allocator = None
elif enables_dynamic_vram() and model_management.get_torch_device().type == "cuda":
    torch.cuda.init()
    torch_devices = tuple(
        device for device in model_management.get_all_torch_devices()
        if device.type == "cuda"
    )
    device_indices = tuple(
        device.index if device.index is not None else torch.cuda.current_device()
        for device in torch_devices
    )
    if args.model_management_device_scope == "local":
        current_device = model_management.get_torch_device()
        device_indices = (
            current_device.index
            if current_device.index is not None
            else torch.cuda.current_device(),
        )

    simple_vram_headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
    control_initialized = comfy_aimdo.control.init(
        simple_vram_headroom=simple_vram_headroom,
        nvml_pressure=not args.disable_nvml_pressure,
    )

    if control_initialized:
        importlib.reload(comfy_aimdo.host_buffer)
        importlib.reload(comfy_aimdo.model_vbar)
        importlib.reload(comfy_aimdo.vram_buffer)

    try:
        device_initialized = comfy_aimdo.control.init_devices([
            (device_index, int(args.vram_headroom * 1024 ** 3))
            for device_index in device_indices
        ])
    except TypeError:
        # comfy-aimdo 0.4.9 protocol.
        device_initialized = comfy_aimdo.control.init_device(device_indices[0])

    if comfy_aimdo.control.lib is not None and device_initialized:
        memory_management.set_aimdo_log_level(comfy_aimdo.control, args.aimdo_logging_level)

        memory_management.aimdo_allocator = comfy_aimdo.torch.get_torch_allocator()
        logger.info("DynamicVRAM support detected and enabled")
    else:
        logger.info("No working comfy-aimdo install detected. DynamicVRAM support disabled. Falling back to legacy ModelPatcher. VRAM estimates may be unreliable especially on Windows")
        memory_management.aimdo_allocator = None
else:
    # Dynamic VRAM (comfy-aimdo) is CUDA-only. On CPU-only environments (the
    # headless serve-pip facade, CPU inference) leave the legacy ModelPatcher in
    # place; init_device(None) would otherwise crash startup.
    memory_management.aimdo_allocator = None

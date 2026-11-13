"""Process-wide CUDA managed-memory allocator for NVIDIA shared-memory devices.

On an integrated NVIDIA GPU with concurrent managed access (DGX Spark GB10) the device and the
CPU share one pool of RAM. Allocating torch's device memory with ``cudaMallocManaged`` keeps
those pages ordinary, pageable system memory that the kernel can reclaim and swap, instead of
pinned device allocations the GPU driver must find and map itself. The allocator has to be
installed before torch initializes CUDA, so detection talks to the driver API directly.
"""
from __future__ import annotations

import ctypes
import logging
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)

CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89
CU_MEM_ATTACH_GLOBAL = 1
CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3
CU_MEM_ADVISE_SET_ACCESSED_BY = 5
CUDA_ERROR_INVALID_CONTEXT = 201

_lock = threading.Lock()
_driver = None
_allocator = None
_active = False
_resolved: Optional[bool] = None


def _load_driver():
    global _driver
    if _driver is None:
        if not sys.platform.startswith("linux"):
            return None
        try:
            driver = ctypes.CDLL("libcuda.so.1")
        except OSError:
            return None
        driver.cuMemAllocManaged.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_uint]
        driver.cuMemAdvise.argtypes = [ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int, ctypes.c_int]
        driver.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        driver.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        driver.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        _driver = driver
    return _driver


def nvidia_shared_memory_devices() -> bool:
    """Whether every visible CUDA device is an integrated NVIDIA GPU with concurrent managed access.

    Queried through the driver API so that torch's CUDA state, and with it the allocator,
    stays uninitialized.
    """
    import torch

    if not torch.version.cuda or getattr(torch.version, "hip", None):
        return False
    driver = _load_driver()
    if driver is None or driver.cuInit(0) != 0:
        return False
    count = ctypes.c_int(0)
    if driver.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value == 0:
        return False
    for ordinal in range(count.value):
        device = ctypes.c_int(0)
        if driver.cuDeviceGet(ctypes.byref(device), ordinal) != 0:
            return False
        for attribute in (CU_DEVICE_ATTRIBUTE_INTEGRATED, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS):
            value = ctypes.c_int(0)
            if driver.cuDeviceGetAttribute(ctypes.byref(value), attribute, device.value) != 0 or not value.value:
                return False
    return True


def resolve(configured: Optional[bool]) -> bool:
    """``--unified-memory-allocator``: explicit True/False wins; None means NVIDIA shared-memory devices only."""
    if configured is not None:
        return bool(configured)
    return nvidia_shared_memory_devices()


def _make_current(driver, device: int) -> bool:
    context = ctypes.c_void_p()
    return (driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), device) == 0
            and driver.cuCtxSetCurrent(context) == 0)


_ALLOC_FN = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)
_FREE_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p)


def _managed_alloc(size, device, stream):
    del stream
    driver = _driver
    pointer = ctypes.c_uint64(0)
    result = driver.cuMemAllocManaged(ctypes.byref(pointer), size, CU_MEM_ATTACH_GLOBAL)
    if result == CUDA_ERROR_INVALID_CONTEXT and _make_current(driver, device):
        result = driver.cuMemAllocManaged(ctypes.byref(pointer), size, CU_MEM_ATTACH_GLOBAL)
    if result != 0:
        logger.error("cuMemAllocManaged of %d bytes on device %d failed with CUDA error %d", size, device, result)
        return 0
    for advice in (CU_MEM_ADVISE_SET_PREFERRED_LOCATION, CU_MEM_ADVISE_SET_ACCESSED_BY):
        result = driver.cuMemAdvise(pointer.value, size, advice, device)
        if result != 0:
            logger.error("cuMemAdvise(%d) on device %d failed with CUDA error %d", advice, device, result)
    return pointer.value


def _managed_free(pointer, size, device, stream):
    del size, device, stream
    driver = _driver
    if not pointer or driver is None:
        return
    # cudaFree semantics: work queued on any stream may still read the block.
    driver.cuCtxSynchronize()
    driver.cuMemFree_v2(pointer)


_c_alloc = _ALLOC_FN(_managed_alloc)
_c_free = _FREE_FN(_managed_free)


def _build_allocator():
    import torch

    class ManagedMemoryAllocator(torch.cuda.memory.CUDAPluggableAllocator):
        def __init__(self):  # pylint: disable=super-init-not-called
            alloc = ctypes.cast(_c_alloc, ctypes.c_void_p).value
            free = ctypes.cast(_c_free, ctypes.c_void_p).value
            self._allocator = torch._C._cuda_customAllocator(alloc, free)

    return ManagedMemoryAllocator()


def install(configured: Optional[bool]) -> bool:
    """Resolve the setting and, when it is on, make managed memory torch's CUDA allocator.

    Must run before torch initializes CUDA. Idempotent; returns whether the allocator is active.
    """
    global _allocator, _active, _resolved
    with _lock:
        if _resolved is not None:
            return _active
        import torch

        _resolved = resolve(configured)
        if _resolved and torch.cuda.is_initialized():
            if configured:
                raise RuntimeError("--unified-memory-allocator must be applied before CUDA is initialized")
            logger.warning("Unified memory allocator left off: CUDA was initialized before ComfyUI's setup ran")
            _resolved = False
        if _resolved:
            if _load_driver() is None:
                raise RuntimeError("--unified-memory-allocator requires the NVIDIA driver library libcuda.so.1")
            _allocator = _build_allocator()
            torch.cuda.memory.change_current_allocator(_allocator)
            _active = True
        logger.info("Unified memory allocator: %s (%s)", "managed memory" if _active else "off",
                    "auto" if configured is None else "configured")
        return _active


def is_active() -> bool:
    return _active

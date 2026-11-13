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
import os
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
_NCCL2_ENVIRONMENT = "TORCH_DIST_USE_NCCL2"

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


_SMALL_BLOCK = 1 << 20
_LARGE_GRANULE = 2 << 20


def size_class(size: int) -> int:
    """Bytes actually allocated for a request: powers of two below 1 MiB, 2 MiB multiples above."""
    if size <= _SMALL_BLOCK:
        return max(512, 1 << (size - 1).bit_length())
    return (size + _LARGE_GRANULE - 1) // _LARGE_GRANULE * _LARGE_GRANULE


class _BlockCache:
    """Managed blocks kept for reuse, because cuMemAllocManaged and cuMemFree cost about a millisecond each.

    A freed block may still be read by work queued on any stream (a pluggable allocator sees no
    record_stream), so it waits in ``pending`` until the next context synchronization, exactly
    the guarantee cudaFree gives, and only then becomes reusable.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.sizes = {}
        self.ready = {}
        self.pending = []
        self.cached = 0

    def take(self, device, block):
        blocks = self.ready.get((device, block))
        if not blocks and any(key == (device, block) for key, _ in self.pending):
            self.promote()
            blocks = self.ready.get((device, block))
        if not blocks:
            return 0
        self.cached -= block
        return blocks.pop()

    def promote(self):
        _driver.cuCtxSynchronize()
        for key, pointer in self.pending:
            self.ready.setdefault(key, []).append(pointer)
        self.pending.clear()

    def release(self) -> int:
        if not self.pending and not any(self.ready.values()):
            return 0
        _driver.cuCtxSynchronize()
        for key, pointer in self.pending:
            self.ready.setdefault(key, []).append(pointer)
        self.pending.clear()
        released = 0
        for (_, block), pointers in self.ready.items():
            for pointer in pointers:
                _driver.cuMemFree_v2(pointer)
                del self.sizes[pointer]
                released += block
            pointers.clear()
        self.cached = 0
        return released


_cache = _BlockCache()


def _allocate_managed(driver, block, device):
    pointer = ctypes.c_uint64(0)
    result = driver.cuMemAllocManaged(ctypes.byref(pointer), block, CU_MEM_ATTACH_GLOBAL)
    if result == CUDA_ERROR_INVALID_CONTEXT and _make_current(driver, device):
        result = driver.cuMemAllocManaged(ctypes.byref(pointer), block, CU_MEM_ATTACH_GLOBAL)
    if result != 0:
        return 0, result
    for advice in (CU_MEM_ADVISE_SET_PREFERRED_LOCATION, CU_MEM_ADVISE_SET_ACCESSED_BY):
        advised = driver.cuMemAdvise(pointer.value, block, advice, device)
        if advised != 0:
            logger.error("cuMemAdvise(%d) on device %d failed with CUDA error %d", advice, device, advised)
    return pointer.value, 0


def _managed_alloc(size, device, stream):
    del stream
    if size == 0:
        return 0
    block = size_class(size)
    with _cache.lock:
        pointer = _cache.take(device, block)
        if pointer:
            return pointer
        pointer, result = _allocate_managed(_driver, block, device)
        if not pointer:
            _cache.release()
            pointer, result = _allocate_managed(_driver, block, device)
        if not pointer:
            logger.error("cuMemAllocManaged of %d bytes on device %d failed with CUDA error %d", block, device, result)
            return 0
        _cache.sizes[pointer] = block
        return pointer


def _managed_free(pointer, size, device, stream):
    del size, stream
    if not pointer or _driver is None:
        return
    with _cache.lock:
        block = _cache.sizes[pointer]
        _cache.pending.append(((device, block), pointer))
        _cache.cached += block


def release_cached() -> int:
    """Return every cached managed block to the driver; the bytes released."""
    if not _active:
        return 0
    with _cache.lock:
        return _cache.release()


def cached_bytes() -> int:
    """Bytes of managed blocks freed by torch and kept for reuse."""
    return _cache.cached


def held_bytes() -> int:
    """Bytes of managed blocks allocated from the driver, in use or cached."""
    return sum(_cache.sizes.values())


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


def _use_stock_nccl_backend():
    """Route ``backend="nccl"`` to torch's stock ProcessGroupNCCL.

    torch's nccl2 implementation (the default "nccl" since torch 2.14) registers every
    allocator segment with its communicators: its caching-allocator hook takes an allocator
    snapshot and attaches a trace tracker, both of which a pluggable allocator rejects, and
    the destructor's detachMemoryHook then terminates the process. The stock backend uses
    the allocator only for tensors. torch resolves "nccl" lazily, when the first process
    group is created, from this variable.
    """
    previous = os.environ.get(_NCCL2_ENVIRONMENT)
    if previous not in (None, "0"):
        logger.warning("%s=%s overridden to 0: torch's nccl2 backend cannot run on a pluggable CUDA allocator",
                       _NCCL2_ENVIRONMENT, previous)
    os.environ[_NCCL2_ENVIRONMENT] = "0"


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
            _use_stock_nccl_backend()
            _active = True
        logger.info("Unified memory allocator: %s (%s)", "managed memory" if _active else "off",
                    "auto" if configured is None else "configured")
        return _active


def is_active() -> bool:
    return _active

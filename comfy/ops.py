"""
    This file is part of ComfyUI.
    Copyright (C) 2024 Stability AI

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

from typing import Optional
import contextlib
import json
import logging
import os
import threading
import typing

from torch import Tensor
import torch

import comfy_aimdo.model_vbar
import comfy_aimdo.torch

from . import float as comfy_float
from . import memory_management
from . import model_management
from . import pinned_memory
from . import rmsnorm
from . import utils
from . import weight_cast
from . import weight_cast_ops
from .cli_args import PerformanceFeature
from .cli_args import args
import dataclasses
from .execution_context import current_execution_context
from .interruption import throw_exception_if_processing_interrupted

logger = logging.getLogger(__name__)


def _streams_in_native_dtype(tensor):
    """Weights that carry a per-tensor scale must cross the dynamic-VRAM vbar in
    their native low-precision layout, not be densely cast to the compute dtype
    mid-stream. That covers comfy_kitchen QuantizedTensors (qdata + scale) and
    plain scaled-fp8 weights (1-byte float storage); the dtype cast / dequant
    happens later in post_cast or the forward, exactly as in the resident path."""
    if isinstance(tensor, QuantizedTensor):
        return True
    dtype = getattr(tensor, "dtype", None)
    return bool(dtype is not None and dtype.is_floating_point and tensor.element_size() == 1)


_DYNAMIC_VRAM_FP8_POLICIES = {"auto", "resident", "materialize"}


def dynamic_vram_fp8_policy():
    policy = os.environ.get("COMFY_DYNAMIC_VRAM_FP8_POLICY", "auto").strip().lower()
    if policy not in _DYNAMIC_VRAM_FP8_POLICIES:
        logger.warning("Unknown COMFY_DYNAMIC_VRAM_FP8_POLICY=%r; using auto", policy)
        return "auto"
    return policy


def dynamic_vram_diag_enabled():
    return os.environ.get("COMFY_DYNAMIC_VRAM_DIAG", "").strip().lower() in {"1", "true", "yes", "on"}


def direct_materialize_pinning_enabled():
    return os.environ.get("COMFY_DIRECT_MATERIALIZE_PINNING", "").strip().lower() in {"1", "true", "yes", "on"}


def _tensor_diag(tensor):
    if tensor is None:
        return "None"
    if isinstance(tensor, QuantizedTensor):
        qdata = getattr(tensor, "_qdata", None)
        params = getattr(tensor, "_params", None)
        param_parts = []
        if params is not None:
            for name in ("scale", "block_scale"):
                value = getattr(params, name, None)
                if isinstance(value, torch.Tensor):
                    param_parts.append(f"{name}={tuple(value.shape)}/{value.dtype}/{value.device}")
            orig_dtype = getattr(params, "orig_dtype", None)
            if orig_dtype is not None:
                param_parts.append(f"orig_dtype={orig_dtype}")
        return (
            f"QuantizedTensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
            f"qdata={tuple(qdata.shape) if isinstance(qdata, torch.Tensor) else None}/"
            f"{getattr(qdata, 'dtype', None)}/{getattr(qdata, 'device', None)}, "
            f"layout={getattr(tensor, '_layout_cls', None)}, {', '.join(param_parts)})"
        )
    return f"Tensor(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device})"


def _vbar_diag(vbar):
    try:
        residency = vbar.get_residency()
        resident_pages = sum(1 for page in residency if page & 1)
        pinned_pages = sum(1 for page in residency if page & 2)
        return (
            f"loaded={vbar.loaded_size()} watermark={vbar.get_watermark()} "
            f"pages={len(residency)} resident_pages={resident_pages} pinned_pages={pinned_pages}"
        )
    except Exception as exc:
        return f"unavailable={exc}"


def _cuda_mem_diag(device):
    if not torch.cuda.is_available() or torch.device(device).type != "cuda":
        return "cuda=unavailable"
    try:
        free, total = torch.cuda.mem_get_info(device)
        return (
            f"cuda_free={free} cuda_total={total} "
            f"torch_alloc={torch.cuda.memory_allocated(device)} "
            f"torch_reserved={torch.cuda.memory_reserved(device)}"
        )
    except Exception as exc:
        return f"cuda=unavailable:{exc}"

_RUN_EVERY_OP_ENABLED = model_management.torch_version_numeric >= (2, 5)

def run_every_op():
    global _RUN_EVERY_OP_ENABLED
    if not _RUN_EVERY_OP_ENABLED or torch.compiler.is_compiling():
        return

    throw_exception_if_processing_interrupted()


def gqa_repeat_factor(query_heads, key_heads, value_heads):
    if key_heads != value_heads:
        raise ValueError(f"Key/value head count mismatch for GQA: {key_heads} != {value_heads}")
    if query_heads == key_heads:
        return 1
    if query_heads % key_heads != 0:
        raise ValueError(f"Query heads must be divisible by key/value heads for GQA: {query_heads} vs {key_heads}")
    return query_heads // key_heads


def repeat_kv_for_gqa(k, v, query_heads, head_dim):
    n_rep = gqa_repeat_factor(query_heads, k.shape[head_dim], v.shape[head_dim])
    if n_rep > 1:
        k = k.repeat_interleave(n_rep, dim=head_dim)
        v = v.repeat_interleave(n_rep, dim=head_dim)
    return k, v


def _scaled_dot_product_attention(q, k, v, *args, **kwargs):
    attn_mask = args[0] if len(args) > 0 else kwargs.get("attn_mask")
    if kwargs.get("enable_gqa", False) and attn_mask is not None:
        k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
        kwargs["enable_gqa"] = False
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)


scaled_dot_product_attention = _scaled_dot_product_attention
_cudnn_attention_disabled = False


def _check_cudnn_nvrtc_compatibility():
    """Check whether cuDNN attention is likely compatible with PyTorch CUDA."""
    try:
        pytorch_cuda = torch.version.cuda
        if pytorch_cuda is None:
            return False

        pytorch_cuda_major = int(pytorch_cuda.split('.')[0])
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version is None:
            return False

        cudnn_major = cudnn_version // 10000
        return cudnn_major >= 9 and pytorch_cuda_major >= 12
    except Exception:
        return False


try:
    if torch.cuda.is_available():
        from torch.nn.attention import SDPBackend
        from torch.nn.attention import sdpa_kernel
        import inspect
        if "set_priority" in inspect.signature(sdpa_kernel).parameters:
            nvidia_cuda = model_management.is_nvidia()
            if nvidia_cuda:
                SDPA_BACKEND_PRIORITY = [
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ]
            else:
                # TheRock's multi-architecture wheels do not currently provide
                # reliable fused SDPA kernels across every supported AMD target.
                # The math backend remains accelerated by ROCm BLAS and avoids
                # invalid kernels or silent corruption from an incompatible
                # AOTriton selection.
                SDPA_BACKEND_PRIORITY = [SDPBackend.MATH]

            if nvidia_cuda and _check_cudnn_nvrtc_compatibility():
                SDPA_BACKEND_PRIORITY.insert(0, SDPBackend.CUDNN_ATTENTION)
            else:
                logger.debug("Skipping cuDNN attention backend due to potential version compatibility")

            SDPA_CONTIGUOUS_KV_L2_SIZE = 0
            if model_management.is_amd():
                props = torch.cuda.get_device_properties(model_management.get_torch_device())
                if props.gcnArchName.split(':')[0] == "gfx1151":  # aotriton gets a lot slower once one head of the strided k/v doesn't fit in L2
                    SDPA_CONTIGUOUS_KV_L2_SIZE = props.L2_cache_size

            def _scaled_dot_product_attention_sdpa2(q, k, v, *args, **kwargs):
                global _cudnn_attention_disabled
                try:
                    if nvidia_cuda and q.nelement() < 1024 * 128:  # arbitrary number, for small inputs cudnn attention seems slower
                        return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
                    attn_mask = args[0] if len(args) > 0 else kwargs.get("attn_mask")
                    if kwargs.get("enable_gqa", False) and attn_mask is not None and not model_management.is_nvidia():
                        k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
                        kwargs["enable_gqa"] = False
                    if SDPA_CONTIGUOUS_KV_L2_SIZE and sum(t.shape[-2] * t.shape[-1] for t in (k, v) if not t.is_contiguous()) * k.element_size() > SDPA_CONTIGUOUS_KV_L2_SIZE:
                        k, v = k.contiguous(), v.contiguous()
                    with sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True):
                        if kwargs.get("enable_gqa", False) and attn_mask is not None and q.shape[-3] != k.shape[-3]:
                            dropout_p = args[1] if len(args) > 1 else kwargs.get("dropout_p", 0.0)
                            is_causal = args[2] if len(args) > 2 else kwargs.get("is_causal", False)
                            params = torch.backends.cuda.SDPAParams(q, k, v, attn_mask, dropout_p, is_causal, True)
                            supports_native_gqa = (
                                torch.backends.cuda.can_use_flash_attention(params)
                                or torch.backends.cuda.can_use_cudnn_attention(params)
                                or torch.backends.cuda.can_use_efficient_attention(params)
                            )
                            if not supports_native_gqa:
                                k, v = repeat_kv_for_gqa(k, v, q.shape[-3], -3)
                                kwargs["enable_gqa"] = False
                        return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
                except RuntimeError as e:
                    error_msg = str(e)
                    if "cuDNN" in error_msg or "cudnn" in error_msg.lower() or "nvrtc" in error_msg.lower():
                        if not _cudnn_attention_disabled:
                            logger.warning(f"cuDNN attention failed, falling back to other backends: {error_msg}")
                            _cudnn_attention_disabled = True
                        fallback_priority = [b for b in SDPA_BACKEND_PRIORITY if b != SDPBackend.CUDNN_ATTENTION]
                        with sdpa_kernel(fallback_priority, set_priority=True):
                            return torch.nn.functional.scaled_dot_product_attention(q, k, v, *args, **kwargs)
                    raise

            scaled_dot_product_attention = _scaled_dot_product_attention_sdpa2
        else:
            logger.warning("Torch version too old to set sdpa backend priority, even though you are using CUDA")
except Exception as exc_info:
    if torch.cuda.is_available():
        logger.debug("Could not set sdpa backend priority.", exc_info=exc_info)

NVIDIA_MEMORY_CONV_BUG_WORKAROUND = False
try:
    if model_management.is_nvidia():
        cudnn_version = torch.backends.cudnn.version()
        if (cudnn_version >= 91002 and cudnn_version < 91500) and model_management.torch_version_numeric >= (2, 9) and model_management.torch_version_numeric <= (2, 10):
            #TODO: change upper bound version once it's fixed'
            NVIDIA_MEMORY_CONV_BUG_WORKAROUND = True
            logging.info("working around nvidia conv3d memory bug.")
except:
    pass

cast_to = model_management.cast_to  # TODO: remove once no more references

_DEFERRED_VBAR_UNPINS = []
_DEFERRED_VBAR_UNPINS_LOCK = threading.RLock()


def cast_to_input(weight, input, non_blocking=False, copy=True):
    return model_management.cast_to(weight, input.dtype, input.device, non_blocking=non_blocking, copy=copy)


def materialize_meta_param(s, param_keys):
    for param_key in param_keys:
        param = getattr(s, param_key, None)
        if param is not None and getattr(param, "is_meta", False) and not hasattr(param.untyped_storage(), "_comfy_tensor_file_slice"):
            setattr(s, param_key, torch.nn.Parameter(torch.zeros(param.shape, dtype=param.dtype), requires_grad=param.requires_grad))


def _defer_vbar_unpin(alloc, device):
    with _DEFERRED_VBAR_UNPINS_LOCK:
        if device is None or device.type != "cuda":
            comfy_aimdo.model_vbar.vbar_unpin(alloc)
            return
        event = torch.cuda.Event()
        event.record(model_management.current_stream(device))
        _DEFERRED_VBAR_UNPINS.append((event, alloc))


def _drain_deferred_vbar_unpins(block=False):
    with _DEFERRED_VBAR_UNPINS_LOCK:
        if not _DEFERRED_VBAR_UNPINS:
            return
        pending = []
        for event, alloc in _DEFERRED_VBAR_UNPINS:
            if block:
                event.synchronize()
                comfy_aimdo.model_vbar.vbar_unpin(alloc)
            elif event.query():
                comfy_aimdo.model_vbar.vbar_unpin(alloc)
            else:
                pending.append((event, alloc))
        _DEFERRED_VBAR_UNPINS[:] = pending


def finish_weight_cast_execution():
    """Finish deferred weight releases at an outer execution boundary.

    Individual layer releases remain asynchronous. Callers use this only after
    the produced tensors are no longer being extended by more model work, so
    dynamic-VRAM pins cannot leak into the next model's residency decision.
    """
    _drain_deferred_vbar_unpins(block=True)
def vbar_above_watermark(s):
    # prefetch ring manages residency itself, hands off
    if getattr(s, "_prefetch", None) is not None:
        return False
    vbar, alloc, size = s._v
    return alloc - vbar.base_addr + size > vbar.get_watermark() * (32 << 20)  # watermark is in VBAR pages


# FIXME: add n=1 cache hit fast path
def cast_modules_with_vbar(
    comfy_modules,
    dtype,
    device,
    bias_dtype,
    non_blocking,
    want_requant=False,
    dedicated_buffer=False,
    prefetch_hint=False,
    return_faulted=False,
):
    offload_stream = None
    cast_buffer = None
    cast_buffer_offset = 0
    if return_faulted:
        fully_faulted = all(
            not getattr(module, param_key + "_function", [])
            for module in comfy_modules
            for param_key in ("weight", "bias")
        )

    def ensure_offload_stream(module, required_size, check_largest):
        nonlocal offload_stream
        nonlocal cast_buffer

        if offload_stream is None:
            offload_stream = model_management.get_offload_stream(device)
        if offload_stream is None or not check_largest or len(comfy_modules) != 1:
            return

        current_size = 0 if cast_buffer is None else cast_buffer.size()
        if current_size < required_size and module is model_management.LARGEST_AIMDO_CASTED_WEIGHT[0]:
            offload_stream = model_management.get_offload_stream(device)
            cast_buffer = None
        if required_size > model_management.LARGEST_AIMDO_CASTED_WEIGHT[1]:
            model_management.LARGEST_AIMDO_CASTED_WEIGHT = (module, required_size)

    def get_cast_buffer(buffer_size, reclaim_vbar=None):
        nonlocal offload_stream
        nonlocal cast_buffer
        nonlocal cast_buffer_offset

        if buffer_size == 0:
            return None

        if dedicated_buffer:
            return torch.empty((buffer_size,), dtype=torch.uint8, device=device)

        if offload_stream is None:
            return torch.empty((buffer_size,), dtype=torch.uint8, device=device)

        cast_buffer = model_management.get_aimdo_cast_buffer(offload_stream, device)
        try:
            alloc = cast_buffer.get(buffer_size, cast_buffer_offset)
        except RuntimeError as exc:
            logger.debug("Falling back to torch cast buffer after aimdo cast buffer allocation failed: %s", exc)
            if reclaim_vbar is not None:
                if dynamic_vram_diag_enabled():
                    logger.warning(
                        "DYNAMIC_VRAM_DIAG cast-buffer fallback before free: requested=%s offset=%s %s %s",
                        buffer_size,
                        cast_buffer_offset,
                        _vbar_diag(reclaim_vbar),
                        _cuda_mem_diag(device),
                    )
                freed = reclaim_vbar.free_memory(1e30)
                if dynamic_vram_diag_enabled():
                    logger.warning(
                        "DYNAMIC_VRAM_DIAG cast-buffer fallback after free: requested=%s freed=%s %s %s",
                        buffer_size,
                        freed,
                        _vbar_diag(reclaim_vbar),
                        _cuda_mem_diag(device),
                    )
                if freed < buffer_size:
                    model_management.free_memory(
                        buffer_size + 512 * 1024 ** 2,
                        device,
                        for_dynamic=True,
                    )
                    if dynamic_vram_diag_enabled():
                        logger.warning(
                            "DYNAMIC_VRAM_DIAG cast-buffer fallback after global free: requested=%s %s %s",
                            buffer_size,
                            _vbar_diag(reclaim_vbar),
                            _cuda_mem_diag(device),
                        )
            try:
                alloc = cast_buffer.get(buffer_size, cast_buffer_offset)
            except RuntimeError as retry_exc:
                logger.debug("Falling back to torch cast buffer after aimdo cast buffer retry failed: %s", retry_exc)
                buffer = torch.empty((buffer_size,), dtype=torch.uint8, device=device)
            else:
                buffer = comfy_aimdo.torch.aimdo_to_tensor(alloc, device)
        else:
            buffer = comfy_aimdo.torch.aimdo_to_tensor(alloc, device)
        cast_buffer_offset += buffer_size
        return buffer

    def target_geometry_for(tensor, target_dtype):
        if tensor is None:
            return None
        if target_dtype is None:
            return tensor
        if _streams_in_native_dtype(tensor):
            # Stream the weight in its native low-precision layout (QuantizedTensor
            # qdata+scale, or plain scaled-fp8 bytes) and let post_cast / the
            # forward apply the dtype cast. Materializing the target dtype here
            # routes the tensor through cast_to_gathered(target_geometries=...),
            # a raw fp8->bf16 cast that drops the per-tensor weight scale, so the
            # streamed result silently diverges from the resident path.
            return model_management.tensor_materialization_geometry(tensor)
        return model_management.tensor_materialization_geometry(tensor, dtype=target_dtype)

    for s in comfy_modules:
        _drain_deferred_vbar_unpins(block=False)
        fault_failed = False
        if dynamic_vram_diag_enabled():
            logger.warning(
                "DYNAMIC_VRAM_DIAG before fault module=%s alloc=%s dtype=%s bias_dtype=%s want_requant=%s policy=%s weight=%s bias=%s %s %s",
                getattr(s, "seed_key", type(s).__name__),
                s._v[2],
                dtype,
                bias_dtype,
                want_requant,
                dynamic_vram_fp8_policy(),
                _tensor_diag(s.weight),
                _tensor_diag(s.bias),
                _vbar_diag(s._v[0]),
                _cuda_mem_diag(device),
            )
        try:
            signature = comfy_aimdo.model_vbar.vbar_fault(s._v)
        except RuntimeError as exc:
            logger.debug(
                "Dynamic VBAR fault failed for %s; using temporary cast buffer: %s",
                getattr(s, "seed_key", type(s).__name__),
                exc,
            )
            signature = None
            fault_failed = True
        if signature is None and not prefetch_hint:
            _drain_deferred_vbar_unpins(block=True)
            try:
                signature = comfy_aimdo.model_vbar.vbar_fault(s._v)
                fault_failed = False
            except RuntimeError as exc:
                logger.debug(
                    "Dynamic VBAR fault retry failed for %s; using temporary cast buffer: %s",
                    getattr(s, "seed_key", type(s).__name__),
                    exc,
                )
                signature = None
                fault_failed = True
        if signature is None:
            try:
                comfy_aimdo.model_vbar.vbar_unpin(s._v)
            except Exception as exc:
                if fault_failed:
                    logger.debug("Dynamic VBAR unpin after failed fault failed: %s", exc)
        if dynamic_vram_diag_enabled():
            logger.warning(
                "DYNAMIC_VRAM_DIAG after fault module=%s signature=%s fault_failed=%s %s %s",
                getattr(s, "seed_key", type(s).__name__),
                signature is not None,
                fault_failed,
                _vbar_diag(s._v[0]),
                _cuda_mem_diag(device),
            )
        if signature is None and prefetch_hint:
            logger.debug(
                "Dynamic VBAR prefetch hint deferred for %s: allocated=%s dtype=%s bias_dtype=%s want_requant=%s policy=%s",
                getattr(s, "seed_key", type(s).__name__),
                s._v[2],
                dtype,
                bias_dtype,
                want_requant,
                dynamic_vram_fp8_policy(),
            )
            s._prefetch = None
            continue
        resident = comfy_aimdo.model_vbar.vbar_signature_compare(signature, s._v_signature)
        if return_faulted and (signature is None or not resident):
            fully_faulted = False
        prefetch = {
            "signature": signature,
            "resident": resident,
        }

        if resident:
            s._prefetch = prefetch
            continue

        materialize_meta_param(s, ["weight", "bias"])
        xfer_dest = comfy_aimdo.torch.aimdo_to_tensor(s._v, device) if signature is not None else None
        if signature is None:
            logger.debug(
                "Dynamic VBAR fault returned no signature for %s: allocated=%s dtype=%s bias_dtype=%s want_requant=%s policy=%s",
                getattr(s, "seed_key", type(s).__name__),
                s._v[2],
                dtype,
                bias_dtype,
                want_requant,
                dynamic_vram_fp8_policy(),
            )
        source_geometry = [
            model_management.tensor_materialization_geometry(s.weight),
            model_management.tensor_materialization_geometry(s.bias),
        ]
        cast_geometry = [
            target_geometry_for(s.weight, dtype),
            target_geometry_for(s.bias, bias_dtype),
        ]
        cast_dest = None
        needs_cast = False
        direct_materialize = cast_geometry != source_geometry
        # A scale-carrying weight (QuantizedTensor or scaled-fp8) must keep its
        # native layout across the vbar. The dense-materialize transfer path
        # (cast_to_gathered with target_geometries) casts mid-stream and drops
        # the per-tensor scale; force the pin/needs_cast streaming path instead so
        # the native payload is preserved and the dtype cast happens in post_cast.
        if _streams_in_native_dtype(s.weight):
            direct_materialize = False
        has_patch_functions = len(getattr(s, "weight_function", [])) > 0 or len(getattr(s, "bias_function", [])) > 0

        xfer_source = [ s.weight, s.bias ]
        fast_disk = s._pin_state["fast_disk"]
        subset = "weights-fast" if fast_disk else "weights"
        pin = pinned_memory.get_pin(s, subset=subset) if not direct_materialize else None
        if pin is None and not fast_disk and not direct_materialize:
            loaded_pin = pinned_memory.get_pin(s, subset="weights-loaded")
            if loaded_pin is not None or signature is not None:
                subset = "weights-loaded"
                pin = loaded_pin
        if pin is not None:
            xfer_source = [ pin ]

        if not direct_materialize:
            for data, geometry in zip([ s.weight, s.bias ], cast_geometry):
                if data is None:
                    continue
                if data.dtype != geometry.dtype:
                    needs_cast = True
                    cast_dest = xfer_dest
                    xfer_dest = None
                    break

        dest_geometry = cast_geometry if direct_materialize else xfer_source
        dest_size = memory_management.vram_aligned_size(dest_geometry)
        if dynamic_vram_diag_enabled():
            logger.warning(
                "DYNAMIC_VRAM_DIAG geometry module=%s source_geometry=%r cast_geometry=%r direct_materialize=%s needs_cast=%s dest_size=%s allocated=%s pin=%s weight=%s bias=%s",
                getattr(s, "seed_key", type(s).__name__),
                source_geometry,
                cast_geometry,
                direct_materialize,
                needs_cast,
                dest_size,
                s._v[2],
                pin is not None,
                _tensor_diag(s.weight),
                _tensor_diag(s.bias),
            )
        if xfer_dest is not None and dest_size > s._v[2]:
            logger.debug(
                "Dynamic VBAR allocation too small for %s: allocated=%s requested=%s dtype=%s bias_dtype=%s want_requant=%s policy=%s",
                getattr(s, "seed_key", type(s).__name__),
                s._v[2],
                dest_size,
                dtype,
                bias_dtype,
                want_requant,
                dynamic_vram_fp8_policy(),
            )
            comfy_aimdo.model_vbar.vbar_unpin(s._v)
            xfer_dest = None
            signature = None
            prefetch["signature"] = None
            if prefetch_hint:
                s._prefetch = None
                continue
        ensure_offload_stream(s, dest_size if xfer_dest is None else 0, True)
        if xfer_dest is None:
            reclaim_vbar = s._v[0] if signature is None else None
            xfer_dest = get_cast_buffer(dest_size, reclaim_vbar=reclaim_vbar)

        def cast_maybe_lowvram_patch(xfer_source, xfer_dest, stream, xfer_dest2=None):
            if xfer_source is not None:
                if getattr(xfer_source, "is_lowvram_patch", False):
                    if xfer_dest is not None:
                        xfer_source.prepare(xfer_dest, stream, copy=True, commit=False)
                        xfer_source = [ xfer_dest ]
                        xfer_dest = xfer_dest2
                        xfer_dest2 = None
                    elif xfer_dest2 is not None:
                        xfer_source.prepare(xfer_dest2, stream, copy=True, commit=False)
                        return
                    else:
                        return
                model_management.cast_to_gathered(xfer_source, xfer_dest, non_blocking=non_blocking, stream=stream, r2=xfer_dest2)

        def handle_pin(m, pin, source, dest, subset="weights", size=None):
            if pin is not None:
                cast_maybe_lowvram_patch([pin], dest, offload_stream)
                return
            if signature is None or not fast_disk or args.high_ram:
                pinned_memory.pin_memory(m, subset=subset, size=size)
                pin = pinned_memory.get_pin(m, subset=subset)
            cast_maybe_lowvram_patch(source, pin, offload_stream, xfer_dest2=dest)

        if direct_materialize:
            # Materializing into a different geometry (e.g. dequantizing a
            # QuantizedTensor or dtype cast) bypasses the host pin: send the
            # source straight to the device destination, interpreting the
            # destination with the cast geometry.
            model_management.cast_to_gathered(
                xfer_source,
                xfer_dest,
                non_blocking=non_blocking,
                stream=offload_stream,
                target_geometries=cast_geometry,
            )
        elif not isinstance(s, WritableWeight) or s.read_before_write:
            handle_pin(s, pin, xfer_source, xfer_dest, subset=subset, size=dest_size)

        for param_key in ("weight", "bias"):
            lowvram_source = getattr(s, param_key + "_lowvram_function", None)
            if lowvram_source is not None:
                ensure_offload_stream(s, cast_buffer_offset, False)
                lowvram_size = lowvram_source.memory_required()
                lowvram_dest = get_cast_buffer(lowvram_size)
                lowvram_source.prepare(lowvram_dest, None, copy=False, commit=True)

                subset = "patches-fast" if fast_disk else "patches"
                pin = pinned_memory.get_pin(lowvram_source, subset=subset)
                if pin is None and not fast_disk:
                    loaded_pin = pinned_memory.get_pin(lowvram_source, subset="patches-loaded")
                    if loaded_pin is not None:
                        subset = "patches-loaded"
                        pin = loaded_pin
                    elif signature is not None:
                        subset = "patches-loaded"
                handle_pin(lowvram_source, pin, lowvram_source, lowvram_dest, subset=subset, size=lowvram_size)


        prefetch["xfer_dest"] = xfer_dest
        prefetch["cast_dest"] = cast_dest
        prefetch["cast_geometry"] = cast_geometry
        prefetch["needs_cast"] = needs_cast
        s._prefetch = prefetch

    if return_faulted:
        return offload_stream, fully_faulted
    return offload_stream


def resolve_cast_module_with_vbar(
    s, dtype, device, bias_dtype, compute_dtype, want_requant, return_weights=True
):

    prefetch = s._prefetch

    if prefetch["resident"]:
        weight = s._v_weight
        bias = s._v_bias
    else:
        xfer_dest = prefetch["xfer_dest"]
        if prefetch["needs_cast"]:
            cast_dest = prefetch["cast_dest"] if prefetch["cast_dest"] is not None else torch.empty((memory_management.vram_aligned_size(prefetch["cast_geometry"]),), dtype=torch.uint8, device=device)
            for pre_cast, post_cast in zip(memory_management.interpret_gathered_like([s.weight, s.bias ], xfer_dest),
                                           memory_management.interpret_gathered_like(prefetch["cast_geometry"], cast_dest)):
                if post_cast is not None:
                    post_cast.copy_(pre_cast)
            xfer_dest = cast_dest

        params = memory_management.interpret_gathered_like(prefetch["cast_geometry"], xfer_dest)
        weight = params[0]
        bias = params[1]
        if prefetch["signature"] is not None:
            # FIXME: Make this non_gemv_specific.
            if (isinstance(s, MixedPrecisionOp) and s.comfy_gemv_optimized and isinstance(weight, QuantizedTensor) and not s._full_precision_mm
                    and s.weight_lowvram_function is None and len(s.weight_function) == 0):
                weight = weight.optimize_for_gemv(xfer_dest)
            s._v_weight = weight
            s._v_bias = bias
        s._v_signature = prefetch["signature"]

    def post_cast(s, param_key, x, dtype, resident, update_weight):
        lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
        fns = getattr(s, param_key + "_function", [])

        if x is None:
            return None

        orig = x

        def to_dequant(tensor, dtype):
            tensor = tensor.to(dtype=dtype)
            if isinstance(tensor, QuantizedTensor):
                tensor = tensor.dequantize()
            return tensor

        keep_quantized = want_requant and isinstance(x, QuantizedTensor) and len(fns) == 0
        if (
            (return_weights and not keep_quantized and orig.dtype != dtype)
            or len(fns) > 0
            or (return_weights and isinstance(x, QuantizedTensor) and not want_requant)
        ):
            x = to_dequant(x, dtype)
        if not resident and lowvram_fn is not None:
            x = to_dequant(x, dtype if compute_dtype is None else compute_dtype)
            x = lowvram_fn(x)
            if (want_requant and len(fns) == 0 or update_weight):
                seed = utils.string_to_seed(s.seed_key)
                if isinstance(orig, QuantizedTensor):
                    y = QuantizedTensor.from_float(x, s.layout_type, scale="recalculate", stochastic_rounding=seed)
                else:
                    y = comfy_float.stochastic_rounding(x, orig.dtype, seed=seed)
            if want_requant and len(fns) == 0:
                x = y
            if update_weight:
                orig.copy_(y)
        for f in fns:
            x = f(x)
        return x

    update_weight = prefetch["signature"] is not None
    weight = post_cast(s, "weight", weight, dtype, prefetch["resident"], update_weight)
    if bias is not None:
        bias = post_cast(s, "bias", bias, bias_dtype, prefetch["resident"], update_weight)

    if prefetch["signature"] is not None:
        prefetch["resident"] = True

    return (weight, bias) if return_weights else None


def cast_bias_weight(s, input=None, dtype=None, device=None, bias_dtype=None, offloadable=False, compute_dtype=None, want_requant=False):
    # NOTE: offloadable=False is a legacy mode and if you are a custom node author reading this please pass
    # offloadable=True and call uncast_bias_weight() after your last usage of the weight/bias. This
    # will add async-offload support to your cast and improve performance.
    if input is not None:
        if dtype is None:
            if isinstance(input, QuantizedTensor):
                dtype = input.params.orig_dtype
            else:
                dtype = input.dtype
        if bias_dtype is None:
            bias_dtype = dtype
        if device is None:
            device = input.device

    def format_return(result, offloadable):
        weight, bias, offload_stream = result
        return (weight, bias, offload_stream) if offloadable else (weight, bias)

    non_blocking = model_management.device_supports_non_blocking(device)

    if s._v is not None and model_management.is_device_cpu(device):

        #vbar doesn't support CPU weights, but some custom nodes have weird paths
        #that might switch the layer to the CPU and expect it to work. We have to take
        #a clone conservatively as we are mmapped and some SFT files are packed misaligned
        #If you are a custom node author reading this, please move your layer to the GPU
        #or declare your ModelPatcher as CPU in the first place.
        materialize_meta_param(s, ["weight", "bias"])
        weight = s.weight.to(dtype=dtype, copy=True)
        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()
        bias = s.bias.to(dtype=bias_dtype, copy=True) if s.bias is not None else None
        return format_return((weight, bias, (None, None, None)), offloadable)

    elif s._v is not None and s.weight.device != device:
        prefetched = getattr(s, "_prefetch", None) is not None
        offload_stream = None
        offload_device = None
        if not prefetched:
            offload_stream = cast_modules_with_vbar([s], dtype, device, bias_dtype, non_blocking, want_requant=want_requant)
            model_management.sync_stream(device, offload_stream)

        weight, bias = resolve_cast_module_with_vbar(s, dtype, device, bias_dtype, compute_dtype, want_requant)

        if not prefetched:
            if s._prefetch["signature"] is not None:
                offload_device = device
            for param_key in ("weight", "bias"):
                lowvram_fn = getattr(s, param_key + "_lowvram_function", None)
                if lowvram_fn is not None:
                    lowvram_fn.clear_prepared()
            s._prefetch = None
        return format_return((weight, bias, (offload_stream, offload_device, None)), offloadable)


    if offloadable and (device != s.weight.device or
                        (s.bias is not None and device != s.bias.device)):
        offload_stream = model_management.get_offload_stream(device)
    else:
        offload_stream = None

    bias = None
    weight = None

    if offload_stream is not None and not args.cuda_malloc:
        cast_buffer_size = memory_management.vram_aligned_size([ s.weight, s.bias ])
        cast_buffer = model_management.get_cast_buffer(offload_stream, device, cast_buffer_size, s)
        #The streams can be uneven in buffer capability and reject us. Retry to get the other stream
        if cast_buffer is None:
            offload_stream = model_management.get_offload_stream(device)
            cast_buffer = model_management.get_cast_buffer(offload_stream, device, cast_buffer_size, s)
        if cast_buffer is not None:
            params = memory_management.interpret_gathered_like([ s.weight, s.bias ], cast_buffer)
            weight = params[0]
            bias = params[1]

    weight_has_function = len(s.weight_function) > 0
    bias_has_function = len(s.bias_function) > 0

    weight = model_management.cast_to(s.weight, None, device, non_blocking=non_blocking, copy=weight_has_function, stream=offload_stream, r=weight)

    if s.bias is not None:
        bias = model_management.cast_to(s.bias, None, device, non_blocking=non_blocking, copy=bias_has_function, stream=offload_stream, r=bias)

    model_management.sync_stream(device, offload_stream)

    bias_a = bias
    weight_a = weight

    if s.bias is not None:
        bias = bias.to(dtype=bias_dtype)
        for f in s.bias_function:
            bias = f(bias)

    keep_quantized_weight = want_requant and isinstance(weight, QuantizedTensor) and not weight_has_function
    if weight_has_function or (not keep_quantized_weight and weight.dtype != dtype) or (isinstance(weight, QuantizedTensor) and not want_requant):
        weight = weight.to(dtype=dtype)
        if isinstance(weight, QuantizedTensor):
            weight = weight.dequantize()
        for f in s.weight_function:
            weight = f(weight)

    return format_return((weight, bias, (offload_stream, weight_a, bias_a)), offloadable)


def uncast_bias_weight(s, weight, bias, offload_stream):
    if offload_stream is None:
        return
    stream, weight_a, bias_a = offload_stream
    device=None
    #FIXME: This is really bad RTTI
    if weight_a is not None and not isinstance(weight_a, torch.Tensor):
        device = weight_a
        _defer_vbar_unpin(s._v, device)
    if stream is None:
        return
    if device is None:
        if weight_a is not None:
            device = weight_a.device
        else:
            if bias_a is None:
                return
            device = bias_a.device
    stream.wait_stream(model_management.current_stream(device))


class CastBiasWeightContext:
    """Legacy cast lifetime helper for modules outside the injected ops path."""

    def __init__(self, *args, **kwargs):
        self.module = args[0] if args else None
        self.state = (
            (None, None)
            if self.module is None
            else cast_bias_weight(*args, **kwargs)
        )

    def __enter__(self):
        result = self.state
        if len(result) < 3 or result[2] is None:
            self.state = self.module = None
        return result[:2]

    def __exit__(self, *_args):
        if self.module is None:
            return
        module, state = self.module, self.state
        self.state = self.module = None
        uncast_bias_weight(module, *state)


def _legacy_weight_cast_prefetch(module, device, dtype, bias_dtype, compute_dtype, want_requant):
    non_blocking = device is not None and model_management.device_supports_non_blocking(device)
    if device is None or module._v is None or model_management.is_device_cpu(device):
        return None
    offload_stream = cast_modules_with_vbar(
        [module],
        dtype,
        device,
        bias_dtype,
        non_blocking,
        want_requant=want_requant,
        dedicated_buffer=True,
        prefetch_hint=False,
    )
    if module._prefetch is None:
        return None
    ready_event = None
    if offload_stream is not None and device is not None and device.type == "cuda":
        ready_event = torch.cuda.Event()
        ready_event.record(offload_stream)
    return offload_stream, device, ready_event


def _legacy_weight_cast_resolve(module, input, dtype, bias_dtype, compute_dtype, want_requant, prefetch_state=None):
    if prefetch_state is not None:
        offload_stream, device, ready_event = prefetch_state
        if ready_event is not None:
            model_management.current_stream(device).wait_event(ready_event)
        else:
            model_management.sync_stream(device, offload_stream)
        weight, bias = resolve_cast_module_with_vbar(module, dtype, device, bias_dtype, compute_dtype, want_requant)
        if module._prefetch["signature"] is not None:
            release_state = (offload_stream, device, None)
        else:
            release_state = (offload_stream, weight, bias)
        for param_key in ("weight", "bias"):
            lowvram_fn = getattr(module, param_key + "_lowvram_function", None)
            if lowvram_fn is not None:
                lowvram_fn.clear_prepared()
        module._prefetch = None
        return weight, bias, release_state
    return cast_bias_weight(
        module,
        input,
        dtype=dtype,
        bias_dtype=bias_dtype,
        offloadable=True,
        compute_dtype=compute_dtype,
        want_requant=want_requant,
    )


def _legacy_weight_cast_release(module, weight, bias, token):
    uncast_bias_weight(module, weight, bias, token)


weight_cast_ops.set_callbacks(_legacy_weight_cast_resolve, _legacy_weight_cast_release, _legacy_weight_cast_prefetch)


def _cast_weight_bias(module, input=None, *, dtype=None, device=None, bias_dtype=None,
                      compute_dtype=None, want_requant=False):
    runtime = weight_cast.get_weight_cast_runtime(module, input)
    return runtime.resolve(
        module,
        cast_bias_weight,
        input,
        dtype=dtype,
        device=device,
        bias_dtype=bias_dtype,
        compute_dtype=compute_dtype,
        want_requant=want_requant,
    )


def _release_weight_bias(module, output, state):
    if isinstance(state, tuple):
        module_key, invocation_id = state
        if isinstance(module_key, torch.Tensor):
            torch.ops.comfy_weight.release_tensor_(output, module_key, invocation_id)
        else:
            torch.ops.comfy_weight.release_(output, module_key, invocation_id)
        return output
    runtime = weight_cast.get_weight_cast_runtime_by_name(state.backend)
    return runtime.release(module, uncast_bias_weight, output, state)


class CastWeightBiasOp:
    comfy_cast_weights = False
    comfy_force_forward = False
    weight_function = []
    bias_function = []
    _v = None
    _v_signature = None
    _v_weight = None
    _v_bias = None
    _prefetch = None


class SkipInit:
    def reset_parameters(self):
        return None


class skip_init:
    class Linear(SkipInit, torch.nn.Linear):
        pass

    class Conv1d(SkipInit, torch.nn.Conv1d):
        pass

    class Conv2d(SkipInit, torch.nn.Conv2d):
        pass

    class Conv3d(SkipInit, torch.nn.Conv3d):
        pass

    class GroupNorm(SkipInit, torch.nn.GroupNorm):
        pass

    class LayerNorm(SkipInit, torch.nn.LayerNorm):
        pass

    class ConvTranspose2d(SkipInit, torch.nn.ConvTranspose2d):
        pass

    class ConvTranspose1d(SkipInit, torch.nn.ConvTranspose1d):
        pass

    class Embedding(SkipInit, torch.nn.Embedding):
        def forward(self, *args, **kwargs) -> Tensor:
            if "out_dtype" in kwargs:
                kwargs.pop("out_dtype")
            return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(cls, dims, *args, **kwargs):
        if dims == 2:
            return cls.Conv2d(*args, **kwargs)
        if dims == 3:
            return cls.Conv3d(*args, **kwargs)
        raise ValueError(f"unsupported dimensions: {dims}")


class WritableWeight(torch.nn.Module, CastWeightBiasOp):
    """One runtime tensor, either realized or cast from writable file backing."""
    # False only when the next use overwrites the complete tensor before reading it.
    read_before_write = True

    def __init__(self, shape, dtype, device="meta"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(shape, dtype=dtype, device=device), requires_grad=False)
        self.bias = None
        self._pins = {}
        self._pin_subset = None

    @contextlib.contextmanager
    def access(self, device, write=True):
        if not self.comfy_cast_weights:
            yield self.weight, None
            return
        assert getattr(self, "_prefetch", None) is not None, "Writable weights require prefetch"
        with CastBiasWeightContext(self, dtype=self.weight.dtype, device=device, offloadable=True) as values:
            try:
                yield values
            finally:
                if write and self._v_signature is None:
                    self.writeback(values[0])

    def writeback(self, value):
        subset = self._pin_subset
        pin = pinned_memory.get_pin(self, subset) if subset is not None else None
        if pin is None:
            subset = "weights-fast" if self._pin_state["fast_disk"] else "weights-loaded" if self._v_signature is not None else "weights"
            pinned_memory.pin_memory(self, subset=subset)
            pin = pinned_memory.get_pin(self, subset)
        if pin is not None:
            dest = memory_management.interpret_gathered_like([self.weight], pin)[0]
            dest.copy_(value, non_blocking=True)
            # FIXME use offload streams
            torch.cuda.current_stream(value.device).synchronize()
            self._pins[subset]["dirty"] = True
        else:
            memory_management.write_tensor_file_slice(self.weight, value)

class disable_weight_init:
    @staticmethod
    def _zero_init_parameter(module, name):
        param = getattr(module, name)
        device = None if getattr(param, "is_meta", False) else param.device
        setattr(module, name, torch.nn.Parameter(torch.zeros(param.shape, device=device, dtype=param.dtype), requires_grad=False))

    @staticmethod
    def _lazy_load_from_state_dict(module, state_dict, prefix, local_metadata,
                                   missing_keys, unexpected_keys, weight_shape,
                                   bias_shape=None):
        assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
        prefix_len = len(prefix)
        for k, v in state_dict.items():
            key = k[prefix_len:]
            if key == "weight":
                if not assign_to_params_buffers:
                    v = v.clone()
                module.weight = torch.nn.Parameter(v, requires_grad=False)
            elif bias_shape is not None and key == "bias" and v is not None:
                if not assign_to_params_buffers:
                    v = v.clone()
                module.bias = torch.nn.Parameter(v, requires_grad=False)
            else:
                unexpected_keys.append(k)

        if module.weight is None:
            module.weight = torch.nn.Parameter(torch.zeros(weight_shape), requires_grad=False)
            missing_keys.append(prefix + "weight")

        if bias_shape is not None and module.bias is None and getattr(module, "comfy_need_lazy_init_bias", False):
            module.bias = torch.nn.Parameter(torch.zeros(bias_shape), requires_grad=False)
            missing_keys.append(prefix + "bias")

    class Linear(torch.nn.Linear, CastWeightBiasOp):

        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            # don't trust subclasses that BYO state dict loader to call us.
            if (not memory_management.aimdo_enabled
                or type(self)._load_from_state_dict is not disable_weight_init.Linear._load_from_state_dict):
                super().__init__(in_features, out_features, bias, device, dtype)
                return

            # Issue is with `torch.empty` still reserving the full memory for the layer.
            # Windows doesn't over-commit memory so without this, We are momentarily commit
            # charged for the weight even though we might zero-copy it when we load the
            # state dict. If the commit charge exceeds the ceiling we can destabilize the
            # system.
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None
            self.comfy_need_lazy_init_bias=bias
            self.weight_comfy_model_dtype = dtype
            self.bias_comfy_model_dtype = dtype

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                strict, missing_keys, unexpected_keys, error_msgs):

            if (not memory_management.aimdo_enabled
                or type(self)._load_from_state_dict is not disable_weight_init.Linear._load_from_state_dict):
                return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                     missing_keys, unexpected_keys, error_msgs)
            disable_weight_init._lazy_load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                missing_keys,
                unexpected_keys,
                weight_shape=(self.out_features, self.in_features),
                bias_shape=(self.out_features,),
            )


        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, cast_state = _cast_weight_bias(self, input, want_requant=isinstance(self.weight, QuantizedTensor))
            x = torch.nn.functional.linear(input, weight, bias)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = self._conv_forward(input, weight, bias)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = self._conv_forward(input, weight, bias)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv3d(torch.nn.Conv3d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def _conv_forward(self, input, weight, bias, autopad=None, *args, **kwargs):
            if autopad == "causal_zero":
                weight = weight[:, :, -input.shape[2]:, :, :]
            if NVIDIA_MEMORY_CONV_BUG_WORKAROUND and weight.dtype in (torch.float16, torch.bfloat16):
                out = torch.cudnn_convolution(input, weight, self.padding, self.stride, self.dilation, self.groups, benchmark=False, deterministic=False, allow_tf32=True)
                if bias is not None:
                    out += bias.reshape((1, -1) + (1,) * (out.ndim - 2))
                return out
            else:
                return super()._conv_forward(input, weight, bias, *args, **kwargs)

        def forward_comfy_cast_weights(self, input, autopad=None):
            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = self._conv_forward(input, weight, bias, autopad=autopad)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0 or "autopad" in kwargs:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class BatchNorm2d(torch.nn.BatchNorm2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias, cast_state = _cast_weight_bias(self, input)
            running_mean = self.running_mean.to(device=input.device, dtype=weight.dtype) if self.running_mean is not None else None
            running_var = self.running_var.to(device=input.device, dtype=weight.dtype) if self.running_var is not None else None
            x = torch.nn.functional.batch_norm(input, running_mean, running_var, weight, bias, self.training, self.momentum, self.eps)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias, cast_state = _cast_weight_bias(self, input)
            else:
                weight = None
                bias = None
                cast_state = None
            x = torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)
            return _release_weight_bias(self, x, cast_state) if cast_state is not None else x

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class RMSNorm(torch.nn.RMSNorm, CastWeightBiasOp):
        def reset_parameters(self):
            self.bias = None
            return None

        def cast_weight(self, input):
            return CastBiasWeightContext(self if self.weight is not None else None, input, offloadable=True)

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias, cast_state = _cast_weight_bias(self, input)
            else:
                weight = None
                bias = None
                cast_state = None
            x = rmsnorm.rms_norm(input, weight, self.eps)  # TODO: switch to commented out line when old torch is deprecated
            # x = torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)
            return _release_weight_bias(self, x, cast_state) if cast_state is not None else x

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose2d(torch.nn.ConvTranspose2d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 2
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = torch.nn.functional.conv_transpose2d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose1d(torch.nn.ConvTranspose1d, CastWeightBiasOp):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 1
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = torch.nn.functional.conv_transpose1d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)
            return _release_weight_bias(self, x, cast_state)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Embedding(torch.nn.Embedding, CastWeightBiasOp):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                     norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                     _freeze=False, device=None, dtype=None):
            # don't trust subclasses that BYO state dict loader to call us.
            if (not memory_management.aimdo_enabled
                    or type(self)._load_from_state_dict is not disable_weight_init.Embedding._load_from_state_dict):
                super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm,
                                 norm_type, scale_grad_by_freq, sparse, _weight,
                                 _freeze, device, dtype)
                return

            torch.nn.Module.__init__(self)
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.sparse = sparse
            # Keep shape/dtype visible for module introspection without reserving storage.
            embedding_dtype = dtype if dtype is not None else torch.get_default_dtype()
            self.weight = torch.nn.Parameter(
                torch.empty((num_embeddings, embedding_dim), device="meta", dtype=embedding_dtype),
                requires_grad=False,
            )
            self.bias = None
            self.weight_comfy_model_dtype = dtype

        def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                                strict, missing_keys, unexpected_keys, error_msgs):

            if (not memory_management.aimdo_enabled
                    or type(self)._load_from_state_dict is not disable_weight_init.Embedding._load_from_state_dict):
                return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                     missing_keys, unexpected_keys, error_msgs)
            disable_weight_init._lazy_load_from_state_dict(
                self,
                state_dict,
                prefix,
                local_metadata,
                missing_keys,
                unexpected_keys,
                weight_shape=(self.num_embeddings, self.embedding_dim),
            )

        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input, out_dtype=None):
            output_dtype = out_dtype
            if self.weight.dtype == torch.float16 or self.weight.dtype == torch.bfloat16:
                out_dtype = None
            weight, bias, cast_state = _cast_weight_bias(self, device=input.device, dtype=out_dtype)
            x = torch.nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).to(dtype=output_dtype)
            return _release_weight_bias(self, x, cast_state)

        def host_rows(self, input, out_dtype=None):
            # table not resident: gather the rows from the host copy instead of streaming the table
            x = self.weight[input.reshape(-1).cpu()].to(input.device)
            x = x.view(*input.shape, x.shape[-1])
            return x if out_dtype is None else x.to(dtype=out_dtype)

        def forward(self, *args, **kwargs):
            run_every_op()
            if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                if "out_dtype" in kwargs:
                    kwargs.pop("out_dtype")
                return super().forward(*args, **kwargs)

    @classmethod
    def conv_nd(s, dims, *args, **kwargs):
        if dims == 2:
            return s.Conv2d(*args, **kwargs)
        elif dims == 3:
            return s.Conv3d(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")


class manual_cast(disable_weight_init):
    class Linear(disable_weight_init.Linear):
        comfy_cast_weights = True

    class Conv1d(disable_weight_init.Conv1d):
        comfy_cast_weights = True

    class Conv2d(disable_weight_init.Conv2d):
        comfy_cast_weights = True

    class Conv3d(disable_weight_init.Conv3d):
        comfy_cast_weights = True

    class BatchNorm2d(disable_weight_init.BatchNorm2d):
        comfy_cast_weights = True

    class GroupNorm(disable_weight_init.GroupNorm):
        comfy_cast_weights = True

    class LayerNorm(disable_weight_init.LayerNorm):
        comfy_cast_weights = True

    class ConvTranspose2d(disable_weight_init.ConvTranspose2d):
        comfy_cast_weights = True

    class ConvTranspose1d(disable_weight_init.ConvTranspose1d):
        comfy_cast_weights = True

    class RMSNorm(disable_weight_init.RMSNorm):
        comfy_cast_weights = True

    class Embedding(disable_weight_init.Embedding):
        comfy_cast_weights = True


def fp8_linear(self, input):
    """
    Legacy FP8 linear function for backward compatibility.
    Uses QuantizedTensor subclass for dispatch.
    """
    dtype = self.weight.dtype
    if dtype not in [torch.float8_e4m3fn]:
        return None

    input_dtype = input.dtype
    input_shape = input.shape
    tensor_3d = input.ndim == 3

    if tensor_3d:
        input = input.reshape(-1, input_shape[2])

    if input.ndim != 2:
        return None
    lora_compute_dtype = model_management.lora_compute_dtype(input.device)
    w, bias, cast_state = _cast_weight_bias(
        self,
        input,
        dtype=dtype,
        bias_dtype=input_dtype,
        compute_dtype=lora_compute_dtype,
        want_requant=True,
    )
    scale_weight = torch.ones((), device=input.device, dtype=torch.float32)

    scale_input = torch.ones((), device=input.device, dtype=torch.float32)
    input = torch.clamp(input, min=-448, max=448, out=input)
    input_fp8 = input.to(dtype).contiguous()
    layout_params_input = TensorCoreFP8Layout.Params(scale=scale_input, orig_dtype=input_dtype, orig_shape=tuple(input_fp8.shape))
    quantized_input = QuantizedTensor(input_fp8, "TensorCoreFP8Layout", layout_params_input)

    # Wrap weight in QuantizedTensor - this enables unified dispatch
    # Call F.linear - __torch_dispatch__ routes to fp8_linear handler in quant_ops.py!
    layout_params_weight = TensorCoreFP8Layout.Params(scale=scale_weight, orig_dtype=input_dtype, orig_shape=tuple(w.shape))
    quantized_weight = QuantizedTensor(w, "TensorCoreFP8Layout", layout_params_weight)
    o = torch.nn.functional.linear(quantized_input, quantized_weight, bias)

    o = _release_weight_bias(self, o, cast_state)
    if tensor_3d:
        o = o.reshape((input_shape[0], input_shape[1], w.shape[0]))

    return o

class fp8_ops(manual_cast):
    class Linear(manual_cast.Linear):
        def reset_parameters(self):
            self.scale_weight = None
            self.scale_input = None
            return None

        def forward_comfy_cast_weights(self, input):
            if len(self.weight_function) == 0 and len(self.bias_function) == 0:
                try:
                    out = fp8_linear(self, input)
                    if out is not None:
                        return out
                except Exception as e:
                    logging.info("Exception during fp8 op: {}".format(e))

            weight, bias, cast_state = _cast_weight_bias(self, input)
            x = torch.nn.functional.linear(input, weight, bias)
            return _release_weight_bias(self, x, cast_state)


class scaled_fp8_op_base(manual_cast):
    pass

CUBLAS_IS_AVAILABLE = False
try:
    from cublas_ops import CublasLinear
    from cublas_ops import cublas_half_matmul
    CUBLAS_IS_AVAILABLE = True
except ImportError:
    pass

if CUBLAS_IS_AVAILABLE:
    class cublas_ops(manual_cast):
        class Linear(CublasLinear, manual_cast.Linear):
            def reset_parameters(self):
                return None

            def forward_comfy_cast_weights(self, input):
                weight, bias, cast_state = _cast_weight_bias(self, input)
                x = cublas_half_matmul(input, weight, bias, self._epilogue_str, self.has_bias)
                return _release_weight_bias(self, x, cast_state)

            def forward(self, *args, **kwargs):
                run_every_op()
                if self.comfy_cast_weights or len(self.weight_function) > 0 or len(self.bias_function) > 0:
                    return self.forward_comfy_cast_weights(*args, **kwargs)
                else:
                    return super().forward(*args, **kwargs)
else:
    class cublas_ops(disable_weight_init):
        pass


Operations = typing.Type[typing.Union[manual_cast, fp8_ops, disable_weight_init, skip_init, scaled_fp8_op_base]]

# ==============================================================================
# Mixed Precision Operations
# ==============================================================================
from . import quant_ops
from .quant_ops import QUANT_ALGOS
from .quant_ops import QuantizedTensor
from .quant_ops import TensorCoreFP8Layout
from .quant_ops import get_layout_class
from .quant_ops import int8_quantization_available
from .quant_ops import mixed_precision_quantization_available


def _swiglu_eager(value):
    gate, up = value.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).mul_(up)


def _situ_eager(x):
    # out of place: the training path runs through here
    hidden, gate = x.chunk(2, dim=-1)
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


INPUT_ACT_EAGER = {
    "gelu_tanh": lambda value: torch.nn.functional.gelu(value, approximate="tanh"),
    "swiglu": _swiglu_eager,
    "situ": _situ_eager,
}


@contextlib.contextmanager
def _input_act_weight(x, input_act, act_weight, act_eps):
    if input_act == "rms_norm" and isinstance(act_weight, torch.nn.Module):
        with act_weight.cast_weight(x) as (weight, _):
            eps = act_weight.eps
            if eps is None:
                eps = torch.finfo(torch.float64 if x.dtype == torch.float64 else torch.float32).eps
            yield weight, eps
    else:
        yield act_weight, act_eps


def _eager_input_act(x, input_act, act_weight=None, act_eps=0.0):
    if input_act is None:
        return x
    if input_act == "rms_norm":
        with _input_act_weight(x, input_act, act_weight, act_eps) as (weight, eps):
            return rmsnorm.rms_norm(x, weight, eps)
    return INPUT_ACT_EAGER[input_act](x)


def _linear_residual(out, residual, residual_scale):
    return out if residual is None else torch.addcmul(residual, out, residual_scale)


def _fp16_linear_wanted(x):
    """kitchen's fp16-accumulate GEMM replaces a plain linear when the user opted into
        fp16 accumulation and the activation is fp16 on CUDA; weights come through cast_bias_weight."""
    return (getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
            and x.dtype == torch.float16 and x.is_cuda)


def _try_quantized_linear_input_act(x, weight, bias, input_act, act_weight, act_eps,
                                   residual, residual_scale):
    if model_management.in_training or not isinstance(weight, QuantizedTensor):
        return NotImplemented
    with _input_act_weight(x, input_act, act_weight, act_eps) as (cast_act_weight, cast_act_eps):
        return weight.linear_input_act(
            x, bias, input_act=input_act, input_act_weight=cast_act_weight, input_act_eps=cast_act_eps,
            residual=residual, residual_scale=residual_scale,
        )


def linear_input_act(linear, x, input_act, act_weight=None, act_eps=0.0,
                     residual=None, residual_scale=None):
    run_every_op()
    weight = linear.weight
    quantized = isinstance(weight, QuantizedTensor)
    if (model_management.in_training or getattr(linear, "comfy_force_forward", False) or getattr(linear, "_full_precision_mm", False)
            or not (quantized or _fp16_linear_wanted(x))):
        out = linear(_eager_input_act(x, input_act, act_weight, act_eps))
        return _linear_residual(out, residual, residual_scale)

    with CastBiasWeightContext(linear, x, offloadable=True,
                               compute_dtype=x.dtype if quantized else None,
                               want_requant=quantized) as (weight, bias):
        if not quantized:
            return linear_input_act_(x, weight, bias, input_act, act_weight, act_eps,
                                     residual, residual_scale)
        out = _try_quantized_linear_input_act(x, weight, bias, input_act, act_weight, act_eps,
                                              residual, residual_scale)
        if out is not NotImplemented:
            return out

    # A tensor-level fallback cannot replace module-owned input processing.
    out = linear(_eager_input_act(x, input_act, act_weight, act_eps))
    return _linear_residual(out, residual, residual_scale)


def linear_input_act_(x, weight, bias, input_act=None, act_weight=None, act_eps=0.0,
                      residual=None, residual_scale=None, *, fp16_accumulation=True):
    """``linear(act(x))`` with optional residual addcmul, fused where the weight supports it.

    Linear weights are already prepared. An RMSNorm module in act_weight owns
    its norm-weight cast context; a tensor uses the supplied act_eps directly.
    """
    if input_act is None and residual is None:
        return torch.nn.functional.linear(x, weight, bias)

    out = _try_quantized_linear_input_act(x, weight, bias, input_act, act_weight, act_eps,
                                          residual, residual_scale)
    if out is not NotImplemented:
        return out
    x = _eager_input_act(x, input_act, act_weight, act_eps)

    if (not model_management.in_training
            and not isinstance(weight, QuantizedTensor)
            and fp16_accumulation and _fp16_linear_wanted(x)):
        return quant_ops.ck.fp16_linear(x, weight, bias, residual=residual, residual_scale=residual_scale)

    return _linear_residual(torch.nn.functional.linear(x, weight, bias), residual, residual_scale)


def _quantized_layout_supports_fast_matmul(layout_type):
    if layout_type is None:
        return False
    try:
        layout_cls = get_layout_class(layout_type)
    except Exception:
        return True
    if layout_cls is None:
        return True
    supports_fast_matmul = getattr(layout_cls, "supports_fast_matmul", None)
    if supports_fast_matmul is None:
        return True
    try:
        return supports_fast_matmul()
    except Exception:
        return True


def should_keep_quantized_vbar(module, tensor):
    if not isinstance(tensor, QuantizedTensor):
        return False
    policy = dynamic_vram_fp8_policy()
    if policy == "resident":
        return True
    if policy == "materialize":
        return False
    layout_type = getattr(module, "layout_type", None)
    return (
        layout_type is not None
        and not getattr(module, "_full_precision_mm", False)
    )


# Grouped-scale integer weights that run through the shared INT8 GEMM (comfy_kitchen
# AsymW4A8Int8Layout): format -> code width, which also fixes the packed weight width K*bits/8.
_GROUPED_INT8_FORMATS = {"asym_w4a8_int8": 4, "w6a8_int8": 6}


class QuantLinearFunc(torch.autograd.Function):
    """Custom autograd function for quantized linear: quantized forward, optionally FP8 backward.

    When training_fp8_bwd is enabled:
      - Forward: quantize input per layout (FP8/NVFP4), use quantized matmul
      - Backward: all matmuls use FP8 tensor cores via torch.mm dispatch
      - Cached input is FP8 (half the memory of bf16)

    When training_fp8_bwd is disabled:
      - Forward: quantize input per layout, use quantized matmul
      - Backward: dequantize weight to compute_dtype, use standard matmul
    """

    @staticmethod
    def forward(ctx, input_float, weight, bias, layout_type, input_scale, compute_dtype):
        input_shape = input_float.shape
        inp = input_float.detach().flatten(0, -2)  # zero-cost view to 2D

        # Quantize input for forward (same layout as weight)
        if layout_type is not None:
            q_input = QuantizedTensor.from_float(inp, layout_type, scale=input_scale)
        else:
            q_input = inp

        w = weight.detach() if weight.requires_grad else weight
        b = bias.detach() if bias is not None and bias.requires_grad else bias

        output = torch.nn.functional.linear(q_input, w, b)

        # Unflatten output to match original input shape
        if len(input_shape) > 2:
            output = output.unflatten(0, input_shape[:-1])

        # Save for backward
        ctx.input_shape = input_shape
        ctx.has_bias = bias is not None
        ctx.compute_dtype = compute_dtype
        ctx.weight_requires_grad = weight.requires_grad
        ctx.fp8_bwd = model_management.training_fp8_bwd

        if ctx.fp8_bwd:
            # Cache FP8 quantized input — half the memory of bf16
            if isinstance(q_input, QuantizedTensor) and layout_type.startswith('TensorCoreFP8'):
                ctx.q_input = q_input  # already FP8, reuse
            else:
                # NVFP4 or other layout — quantize input to FP8 for backward
                ctx.q_input = QuantizedTensor.from_float(inp, "TensorCoreFP8E4M3Layout")
            ctx.save_for_backward(weight)
        else:
            ctx.q_input = None
            ctx.save_for_backward(input_float, weight)

        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        compute_dtype = ctx.compute_dtype
        grad_2d = grad_output.flatten(0, -2).to(compute_dtype)

        # Value casting — only difference between fp8 and non-fp8 paths
        if ctx.fp8_bwd:
            weight, = ctx.saved_tensors
            # Wrap as FP8 QuantizedTensors → torch.mm dispatches to _scaled_mm
            grad_mm = QuantizedTensor.from_float(grad_2d, "TensorCoreFP8E5M2Layout")
            if isinstance(weight, QuantizedTensor) and weight._layout_cls.startswith("TensorCoreFP8"):
                weight_mm = weight
            elif isinstance(weight, QuantizedTensor):
                weight_mm = QuantizedTensor.from_float(weight.dequantize().to(compute_dtype), "TensorCoreFP8E4M3Layout")
            else:
                weight_mm = QuantizedTensor.from_float(weight.to(compute_dtype), "TensorCoreFP8E4M3Layout")
            input_mm = ctx.q_input
        else:
            input_float, weight = ctx.saved_tensors
            # Standard tensors → torch.mm does regular matmul
            grad_mm = grad_2d
            if isinstance(weight, QuantizedTensor):
                weight_mm = weight.dequantize().to(compute_dtype)
            else:
                weight_mm = weight.to(compute_dtype)
            input_mm = input_float.flatten(0, -2).to(compute_dtype) if ctx.weight_requires_grad else None

        # Computation — same for both paths, dispatch handles the rest
        grad_input = torch.mm(grad_mm, weight_mm)
        if len(ctx.input_shape) > 2:
            grad_input = grad_input.unflatten(0, ctx.input_shape[:-1])

        grad_weight = None
        if ctx.weight_requires_grad:
            grad_weight = torch.mm(grad_mm.t(), input_mm)

        grad_bias = None
        if ctx.has_bias:
            grad_bias = grad_2d.sum(dim=0)

        return grad_input, grad_weight, grad_bias, None, None, None

# Quantized-weight module helpers

def _quantized_apply(module, fn, recurse=True):
    """Re-wrap Parameters after fn so .to()/.cuda() propagate through QuantizedTensor weights."""
    if recurse:
        for child in module.children():
            child._apply(fn)
    for key, param in module._parameters.items():
        if param is None:
            continue
        p = fn(param)
        if (not torch.is_inference_mode_enabled()) and p.is_inference():
            p = p.clone()
        module.register_parameter(key, torch.nn.Parameter(p, requires_grad=False))
    for key, buf in module._buffers.items():
        if buf is not None:
            module._buffers[key] = fn(buf)
    return module


def _load_quantized_module(module, super_load, state_dict, prefix, local_metadata, strict,
                            missing_keys, unexpected_keys, error_msgs, load_extra_params=False):
    """Shared _load_from_state_dict body for quantized-weight modules.

    Pops weight (+ scales, +/- extras), populates module.weight as a Parameter
    or Parameter-wrapped QuantizedTensor, then calls super_load and strips
    consumed keys from missing_keys. Reads compute_dtype from factory_kwargs
    and disabled formats from module._disabled_formats.
    """
    device = module.factory_kwargs["device"]
    compute_dtype = module.factory_kwargs["dtype"]
    disabled_formats = module._disabled_formats
    disabled_storage_formats = module._disabled_storage_formats
    layer_name = prefix.rstrip('.')

    weight = state_dict.pop(f"{prefix}weight", None)
    if weight is None:
        logging.warning(f"Missing weight for layer {layer_name}")
        module.weight = None
        return
    manually_loaded_keys = [f"{prefix}weight"]

    def pop_scale(name, dtype=None):
        key = f"{prefix}{name}"
        v = state_dict.pop(key, None)
        if v is not None:
            v = v.to(device=device)
            if dtype is not None:
                v = v.view(dtype=dtype)
            manually_loaded_keys.append(key)
        return v

    layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
    if layer_conf is not None:
        layer_conf = json.loads(layer_conf.numpy().tobytes())

    if layer_conf is None:
        module.weight = torch.nn.Parameter(weight.to(device=device, dtype=compute_dtype), requires_grad=False)
    else:
        module.quant_format = layer_conf.get("format", None)
        module._full_precision_mm_config = layer_conf.get("full_precision_matrix_mult", False)
        if not module._full_precision_mm:
            module._full_precision_mm = module._full_precision_mm_config
        if module.quant_format in disabled_formats:
            module._full_precision_mm = True
        if module.quant_format is None:
            raise ValueError(f"Unknown quantization format for layer {layer_name}")

        qconfig = QUANT_ALGOS[module.quant_format]
        module.layout_type = qconfig["comfy_tensor_layout"]
        layout_cls = get_layout_class(module.layout_type)

        # Per-format scales; fp8 dtype views handle both legacy uint8-on-disk and native fp8.
        if module.quant_format in ("float8_e4m3fn", "float8_e5m2"):
            scales = {"scale": pop_scale("weight_scale")}
        elif module.quant_format == "int8_tensorwise":
            scale = pop_scale("weight_scale")
            if scale is None:
                raise ValueError(f"Missing INT8 weight scale for layer {layer_name}")
            scales = {"scale": scale}
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            if layer_conf.get("convrot", params_conf.get("convrot", False)):
                scales["convrot"] = True
                scales["convrot_groupsize"] = int(
                    layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
                )
        elif module.quant_format == "svdquant_w4a4":
            ws = pop_scale("weight_scale")
            pd = pop_scale("weight_proj_down")
            pu = pop_scale("weight_proj_up")
            sf = pop_scale("weight_smooth_factor")
            if ws is None or pd is None or pu is None or sf is None:
                raise ValueError(f"Missing SVDQuant W4A4 tensors for layer {layer_name}")
            # wscales / projections / smoothing stay in the checkpoint
            # compute dtype (bf16/fp16) — the kernel reads them as-is.
            scales = {"scale": ws, "proj_down": pd, "proj_up": pu, "smooth_factor": sf,
                      "act_unsigned": bool(layer_conf.get("act_unsigned", False))}
        elif module.quant_format == "awq_w4a16":
            ws = pop_scale("weight_scale")
            zs = pop_scale("weight_zeros")
            if ws is None or zs is None:
                raise ValueError(f"Missing AWQ W4A16 scales/zeros for layer {layer_name}")
            scales = {"scale": ws, "zeros": zs,
                      "group_size": int(layer_conf.get("group_size", 64))}
        elif module.quant_format == "mxfp8":
            bs = pop_scale("weight_scale", torch.float8_e8m0fnu)
            if bs is None:
                raise ValueError(f"Missing MXFP8 block scales for layer {layer_name}")
            scales = {"scale": bs}
        elif module.quant_format == "nvfp4":
            ts = pop_scale("weight_scale_2")
            bs = pop_scale("weight_scale", torch.float8_e4m3fn)
            if ts is None or bs is None:
                raise ValueError(f"Missing NVFP4 scales for layer {layer_name}")
            scales = {"scale": ts, "block_scale": bs}
        elif module.quant_format == "convrot_w4a4":
            scale = pop_scale("weight_scale")
            if scale is None:
                raise ValueError(f"Missing ConvRot W4A4 weight scale for layer {layer_name}")
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            scales = {
                "scale": scale,
                "convrot_groupsize": int(
                    layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
                ),
                "quant_group_size": 64,
                "linear_dtype": layer_conf.get("linear_dtype", params_conf.get("linear_dtype", "int4")),
            }
        elif module.quant_format in _GROUPED_INT8_FORMATS:
            # asym_w4a8_int8: int4 weight (packed int8 [N,K/2]) + optional Lloyd-Max codebook;
            # w6a8_int8: uniform int6 weight (packed int8 [N,3K/4]). Both carry an fp8 per-group
            # scale (weight_s_rel) and an fp32 per-channel scale (weight_s_channel).
            bits = _GROUPED_INT8_FORMATS[module.quant_format]
            if weight.shape[1] * 8 != module._orig_shape[1] * bits:
                raise ValueError(
                    f"{module.quant_format} layer {layer_name}: packed weight width {weight.shape[1]} "
                    f"does not match K={module._orig_shape[1]} at {bits} bits"
                )
            scale = pop_scale("weight_s_rel")
            if scale is None:
                raise ValueError(f"Missing {module.quant_format} group scale (weight_s_rel) for layer {layer_name}")
            if scale.dtype == torch.uint8:
                scale = scale.view(torch.float8_e4m3fn)
            params_conf = layer_conf.get("params", {})
            if not isinstance(params_conf, dict):
                params_conf = {}
            scales = {
                "scale": scale,
                "s_channel": pop_scale("weight_s_channel"),
                "codebook": pop_scale("weight_codebook"),
                "group_size": int(layer_conf.get("group_size", params_conf.get("group_size", 16))),
                "convrot_groupsize": int(
                    layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256))
                ),
            }
        else:
            raise ValueError(f"Unsupported quantization format: {module.quant_format}")

        params = layout_cls.Params(**scales, orig_dtype=compute_dtype, orig_shape=module._orig_shape)
        quantized_weight = QuantizedTensor(weight.to(device=device, dtype=qconfig["storage_t"]), module.layout_type, params)
        if module.quant_format in disabled_storage_formats:
            module.layout_type = None
            module.weight = torch.nn.Parameter(quantized_weight.dequantize().to(device=device, dtype=compute_dtype), requires_grad=False)
        else:
            module.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)

        if load_extra_params:
            for param_name in qconfig["parameters"]:
                if param_name in {"weight_scale", "weight_scale_2"}:
                    continue
                param_key = f"{prefix}{param_name}"
                _v = state_dict.pop(param_key, None)
                if _v is None:
                    continue
                module.register_parameter(param_name, torch.nn.Parameter(_v.to(device=device), requires_grad=False))
                manually_loaded_keys.append(param_key)

    super_load(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    for key in manually_loaded_keys:
        if key in missing_keys:
            missing_keys.remove(key)


def _quantized_weight_state_dict(module, sd, prefix, extra_quant_conf=None, extra_quant_params=()):
    """Shared state_dict body. extra_quant_conf merges into the comfy_quant JSON;
    extra_quant_params names attributes written as additional top-level keys."""
    if not hasattr(module, 'weight'):
        logging.warning(f"Warning: state dict on uninitialized op {prefix}")
        return sd
    bias = getattr(module, 'bias', None)
    if bias is not None:
        sd[f"{prefix}bias"] = bias
    if module.weight is None:
        return sd
    if isinstance(module.weight, QuantizedTensor):
        sd.update(module.weight.state_dict(f"{prefix}weight"))
        quant_conf = {"format": module.quant_format}
        extra_layout_conf = getattr(module.weight.layout_cls, "extra_state_dict_conf", None)
        if extra_layout_conf is not None:
            quant_conf.update(extra_layout_conf())
        weight_params = getattr(module.weight, "_params", None)
        if weight_params is not None:
            # Per-layer topology/grouping facts that live on the layout params
            # (e.g. SVDQuant act_unsigned, AWQ group_size) round-trip through
            # the comfy_quant JSON.
            if getattr(weight_params, "act_unsigned", False):
                quant_conf["act_unsigned"] = True
            params_group_size = getattr(weight_params, "group_size", None)
            if params_group_size is not None:
                quant_conf["group_size"] = int(params_group_size)
            if module.quant_format == "int8_tensorwise" and getattr(weight_params, "convrot", False):
                quant_conf["convrot"] = True
                quant_conf["convrot_groupsize"] = int(getattr(weight_params, "convrot_groupsize", 256))
            elif module.quant_format == "convrot_w4a4":
                quant_conf["convrot_groupsize"] = getattr(weight_params, "convrot_groupsize", 256)
                linear_dtype = getattr(weight_params, "linear_dtype", "int4")
                if linear_dtype != "int4":
                    quant_conf["linear_dtype"] = linear_dtype
            elif module.quant_format in _GROUPED_INT8_FORMATS:
                quant_conf["group_size"] = int(getattr(weight_params, "group_size", 16))
                quant_conf["convrot_groupsize"] = int(getattr(weight_params, "convrot_groupsize", 256))
        if getattr(module, '_full_precision_mm_config', False):
            quant_conf["full_precision_matrix_mult"] = True
        if extra_quant_conf:
            quant_conf.update(extra_quant_conf)
        sd[f"{prefix}comfy_quant"] = torch.tensor(list(json.dumps(quant_conf).encode("utf-8")), dtype=torch.uint8)
        for name in extra_quant_params:
            value = getattr(module, name, None)
            if value is not None:
                sd[f"{prefix}{name}"] = value
    else:
        sd[f"{prefix}weight"] = module.weight
    return sd


class MixedPrecisionOp(CastWeightBiasOp):
    quant_format = None
    # Prefer small-M execution; Kitchen owns the layout and preserves GEMM support.
    # Keep opt-in until extensive testing allows unconditional relayout and a single
    # resident layout/kernel path instead of parallel canonical and packed paths.
    comfy_gemv_optimized = False

    def can_use_quantized_matmul(self, disabled_formats):
        return (self.quant_format in QUANT_ALGOS
                and not self._full_precision_mm_config
                and self.quant_format not in self._disabled_formats
                and self.quant_format not in disabled_formats)


def mixed_precision_ops(quant_config={}, compute_dtype=torch.bfloat16, full_precision_mm=False, disabled=[], disabled_storage=[]):
    class MixedPrecisionOps(manual_cast):
        _quant_config = quant_config
        _compute_dtype = compute_dtype
        _full_precision_mm = full_precision_mm
        _disabled = disabled
        _disabled_storage = disabled_storage

        class Linear(torch.nn.Module, MixedPrecisionOp):
            _disabled_formats = disabled
            _disabled_storage_formats = disabled_storage

            def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
                super().__init__()

                self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}

                self.in_features = in_features
                self.out_features = out_features
                self._orig_shape = (out_features, in_features)
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)

                self.tensor_class = None
                self._full_precision_mm = MixedPrecisionOps._full_precision_mm
                self._full_precision_mm_config = False
                self.weight_function = []
                self.bias_function = []
                self.comfy_cast_weights = False

            def reset_parameters(self):
                return None

            def _load_from_state_dict(self, *args):
                _load_quantized_module(self, super()._load_from_state_dict, *args, load_extra_params=True)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix, extra_quant_params=("input_scale", "pre_quant_scale"))

            def _forward(self, input, weight, bias):
                return torch.nn.functional.linear(input, weight, bias)

            def forward_comfy_cast_weights(
                self,
                input,
                compute_dtype=None,
                want_requant=False,
                weight_only_quant=False,
            ):
                if weight_only_quant:
                    weight, bias, cast_state = _cast_weight_bias(
                        self,
                        input=None,
                        dtype=self.weight.dtype,
                        device=input.device,
                        bias_dtype=input.dtype,
                        compute_dtype=compute_dtype,
                        want_requant=True,
                    )
                    weight = weight.to(dtype=input.dtype)
                else:
                    weight, bias, cast_state = _cast_weight_bias(
                        self,
                        input,
                        compute_dtype=compute_dtype,
                        want_requant=want_requant,
                    )
                    if self._full_precision_mm and isinstance(weight, QuantizedTensor):
                        weight = weight.dequantize()
                x = self._forward(input, weight, bias)
                return _release_weight_bias(self, x, cast_state)

            def forward(self, input, *args, **kwargs):
                run_every_op()

                pre_quant_scale = getattr(self, "pre_quant_scale", None)
                if pre_quant_scale is not None:
                    input = input * model_management.cast_to_device(pre_quant_scale, input.device, input.dtype)

                if (
                    weight_cast.is_torch_compiling()
                    and weight_cast.graph_visible_backend_unavailable_reason() is None
                    and not model_management.is_device_cpu(input.device)
                ):
                    return self.forward_comfy_cast_weights(input, input.dtype, want_requant=False)

                input_shape = input.shape
                reshaped_nd = False
                #If cast needs to apply lora, it should be done in the compute dtype
                compute_dtype = input.dtype

                force_cast_blocks_quantized = getattr(self, 'comfy_force_cast_weights', False) and not isinstance(self.weight, QuantizedTensor)
                _use_quantized = (
                        getattr(self, 'layout_type', None) is not None and
                        not isinstance(input, QuantizedTensor) and not self._full_precision_mm and
                        not force_cast_blocks_quantized and
                        len(self.weight_function) == 0 and len(self.bias_function) == 0
                )
                quantize_input = QUANT_ALGOS.get(getattr(self, 'quant_format', None), {}).get("quantize_input", True)

                # Training path: quantized forward with compute_dtype backward via autograd function
                if (input.requires_grad and _use_quantized and quantize_input):

                    weight, bias, cast_state = _cast_weight_bias(
                        self,
                        input,
                        compute_dtype=compute_dtype,
                        want_requant=True
                    )

                    scale = getattr(self, 'input_scale', None)
                    if scale is not None:
                        scale = model_management.cast_to_device(scale, input.device, None)

                    output = QuantLinearFunc.apply(
                        input, weight, bias, self.layout_type, scale, compute_dtype
                    )

                    return _release_weight_bias(self, output, cast_state)

                # Inference path (unchanged)
                keep_quantized_weight = isinstance(input, QuantizedTensor)
                if _use_quantized and quantize_input:
                    layout_cls = get_layout_class(self.layout_type)
                    if not getattr(layout_cls, "QUANTIZES_INPUT", True):
                        # Layouts whose kernels fuse activation handling
                        # internally (SVDQuant W4A4, AWQ W4A16) take the float
                        # input as-is; keep the quantized weight so dispatch
                        # reaches the fused kernel.
                        keep_quantized_weight = True
                    else:
                        # Reshape higher-rank tensors to 2D for quantization.
                        input_reshaped = input.reshape(-1, input_shape[-1]) if input.ndim >= 3 else input

                        # Fall back to non-quantized for non-2D tensors
                        if input_reshaped.ndim == 2:
                            # Layouts can decline small batches; the plain-input
                            # path dequantizes the weight and runs a float
                            # linear instead.
                            layout_gate = getattr(layout_cls, "should_quantize_input", None)
                            if layout_gate is None or layout_gate(input_reshaped):
                                reshaped_nd = input.ndim >= 3
                                quantize_activation = getattr(layout_cls, "quantize_activation", None)
                                if quantize_activation is not None:
                                    input = quantize_activation(input_reshaped)
                                else:
                                    # dtype is now implicit in the layout class
                                    scale = getattr(self, 'input_scale', None)
                                    if scale is not None:
                                        scale = model_management.cast_to_device(scale, input.device, None)
                                    input = QuantizedTensor.from_float(input_reshaped, self.layout_type, scale=scale)
                                keep_quantized_weight = True

                weight_only_quant = _use_quantized and not quantize_input and isinstance(self.weight, QuantizedTensor)
                if weight_only_quant:
                    output = self.forward_comfy_cast_weights(
                        input,
                        compute_dtype,
                        want_requant=keep_quantized_weight,
                        weight_only_quant=True,
                    )
                else:
                    output = self.forward_comfy_cast_weights(
                        input,
                        compute_dtype,
                        want_requant=keep_quantized_weight,
                    )

                # Reshape output back to original rank if input was >2D
                if reshaped_nd:
                    output = output.reshape((*input_shape[:-1], self.weight.shape[0]))

                return output

            def convert_weight(self, weight, inplace=False, **kwargs):
                if isinstance(weight, QuantizedTensor):
                    return weight.dequantize()
                else:
                    return weight

            def set_weight(self, weight, inplace_update=False, seed=None, return_weight=False, **kwargs):
                if getattr(self, 'layout_type', None) is not None:
                    try:
                        # dtype is now implicit in the layout class
                        weight = QuantizedTensor.from_float(weight, self.layout_type, scale="recalculate", stochastic_rounding=seed, inplace_ops=True,
                                                            **_requantize_kwargs(self.weight)).to(self.weight.dtype)
                    except NotImplementedError:
                        # Offline-calibrated layouts (SVDQuant, AWQ) cannot
                        # requantize a patched weight; keep it dense instead.
                        # Correct but loses the int4 memory/speed for this layer.
                        logging.warning(
                            "LoRA bake requantization is not supported for %s; keeping the patched weight dense",
                            self.layout_type)
                        self.layout_type = None
                        weight = weight.to(self.weight.dtype)
                else:
                    weight = weight.to(self.weight.dtype)
                if return_weight:
                    return weight

                assert inplace_update is False  # TODO: eventually remove the inplace_update stuff
                self.weight = torch.nn.Parameter(weight, requires_grad=False)

            def _apply(self, fn, recurse=True):  # This is to get torch.compile + moving weights to another device working
                return _quantized_apply(self, fn, recurse)

        class MoEExperts(torch.nn.Module, MixedPrecisionOp):
            """Container for E quantized expert weights, indexed via expert_weight(i).

            The bank lives on self.weight as a single 3D tensor — either a
            compute_dtype Parameter or a Parameter wrapping a QuantizedTensor
            with leading expert dim.

            State-dict layout matches mixed_precision_ops.Linear with a leading
            expert dim:
                {prefix}.weight          quant data (storage_t), leading dim = E
                {prefix}.weight_scale    block / per-tensor scale
                {prefix}.weight_scale_2  [E] or scalar           NVFP4 only
                {prefix}.bias            [E, out_features]       optional, compute_dtype
                {prefix}.comfy_quant     json -> {{"format": "...", "num_experts": E}}

            Without comfy_quant the weight loads as a plain compute_dtype 3D Parameter [E, out, in].
            """

            _disabled_formats = disabled
            _disabled_storage_formats = disabled_storage

            def __init__(self, num_experts: int, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
                super().__init__()
                self.num_experts = num_experts
                self.in_features = in_features
                self.out_features = out_features
                self._orig_shape = (num_experts, out_features, in_features)
                self.factory_kwargs = {"device": device, "dtype": MixedPrecisionOps._compute_dtype}
                if bias:
                    self.bias = torch.nn.Parameter(torch.empty(num_experts, out_features, **self.factory_kwargs))
                else:
                    self.register_parameter("bias", None)

                # Populated by _load_from_state_dict:
                self.weight = None
                self.quant_format = None
                self.layout_type = None
                self._full_precision_mm = MixedPrecisionOps._full_precision_mm
                self._full_precision_mm_config = False
                self._resident_bank = None

            def reset_parameters(self):
                return None

            def _apply(self, fn, recurse=True):
                return _quantized_apply(self, fn, recurse)

            def _load_from_state_dict(self, state_dict, prefix, *args):
                layer_conf = state_dict.get(f"{prefix}comfy_quant", None)
                quant_format = json.loads(layer_conf.numpy().tobytes()).get("format") if layer_conf is not None else None
                scale = state_dict.get(f"{prefix}weight_scale", None)
                if quant_format in _GROUPED_INT8_FORMATS or (quant_format == "int8_tensorwise" and scale is not None and scale.ndim == 3):
                    # per-row scaled layouts are 2-D only: keep the bank as [E * out, in] and slice rows per expert
                    for name in ("weight", "weight_scale", "weight_s_rel", "weight_s_channel"):
                        t = state_dict.get(f"{prefix}{name}", None)
                        if t is not None and t.ndim > 1 and t.shape[0] == self.num_experts:
                            state_dict[f"{prefix}{name}"] = t.reshape(self.num_experts * t.shape[1], *t.shape[2:])
                    self._orig_shape = (self.num_experts * self.out_features, self.in_features)
                _load_quantized_module(self, super()._load_from_state_dict, state_dict, prefix, *args, load_extra_params=False)

            def expert_weight(self, i: int):
                """Expert i's weight (Tensor or per-expert QuantizedTensor view)."""
                if isinstance(self.weight, QuantizedTensor):
                    return self._expert_qt_from(self.weight, i)
                return self.weight[i]

            def bank_cacheable(self, input):
                """Use the resident bank when it fits the current VRAM watermark."""
                if self.weight.device == input.device:
                    return True
                return self._v is not None and not model_management.is_device_cpu(input.device) and not vbar_above_watermark(self)

            def _cast_bank(self, input):
                # A quantized bank stays quantized: the layouts dequantize one [out, in] matrix at a time, per expert.
                if isinstance(self.weight, QuantizedTensor):
                    return CastBiasWeightContext(self, input=None, dtype=self.weight.dtype, device=input.device, bias_dtype=input.dtype, offloadable=True, want_requant=True)
                return CastBiasWeightContext(self, input, offloadable=True)

            def _dequantize_bank(self, weight, dtype):
                # in expert chunks: the W4A8 dequantize kernel allocates over twice its output in temporaries
                flat = QuantizedTensor(weight._qdata, weight._layout_cls, dataclasses.replace(weight._params, orig_dtype=dtype))
                out = torch.empty((self.num_experts, self.out_features, self.in_features), dtype=dtype, device=weight.device)
                step = max(1, (256 << 20) // (self.out_features * self.in_features * out.element_size()))
                for first in range(0, self.num_experts, step):
                    last = min(first + step, self.num_experts)
                    out[first:last] = self._bank_rows(flat, first, last).dequantize().view(last - first, self.out_features, self.in_features)
                return out

            def _bank_rows(self, weight: QuantizedTensor, first: int, last: int) -> QuantizedTensor:
                """Experts [first, last) of a flat [E * out, in] bank as one QuantizedTensor."""
                params = weight._params
                rows = slice(first * self.out_features, last * self.out_features)
                per_row = {f.name: getattr(params, f.name)[rows] for f in dataclasses.fields(params) if torch.is_tensor(getattr(params, f.name)) and getattr(params, f.name).ndim >= 1 and getattr(params, f.name).shape[0] == weight._qdata.shape[0]}
                return QuantizedTensor(weight._qdata[rows], weight._layout_cls, dataclasses.replace(params, orig_shape=((last - first) * self.out_features, self.in_features), **per_row))

            @contextlib.contextmanager
            def bank_resident(self, input):
                """Cast the whole bank once; expert_linear inside reuses the cast.
                Not re-entrant — do not nest calls on the same instance.
                """
                with self._cast_bank(input) as (weight, bias):
                    if self._full_precision_mm and isinstance(weight, QuantizedTensor) and weight._qdata.ndim == 2:  # flat per-row banks; 3-D banks dequantize per expert
                        weight = self._dequantize_bank(weight, input.dtype)
                    self._resident_bank = (weight, bias)
                    try:
                        yield self
                    finally:
                        self._resident_bank = None

            def expert_linear(self, input: torch.Tensor, i: int) -> torch.Tensor:
                """Linear against expert i's weight (with optional bias)."""
                resident = getattr(self, "_resident_bank", None)
                if resident is not None:
                    weight, bias = resident
                    return self._expert_linear_impl(input, weight, bias, i)
                with self._cast_bank(input) as (weight, bias):
                    return self._expert_linear_impl(input, weight, bias, i)

            def _expert_linear_impl(self, input, weight, bias, i):
                if isinstance(weight, QuantizedTensor):
                    qw = self._expert_qt_from(weight, i)
                else:
                    qw = cast_to_input(weight.reshape(self.num_experts, self.out_features, self.in_features)[i], input, copy=False)
                b = cast_to_input(bias[i], input, copy=False) if bias is not None else None

                if isinstance(qw, QuantizedTensor):
                    use_fast = (
                        not self._full_precision_mm
                        and qw.layout_cls.supports_fast_matmul()
                        and input.dim() == 2
                    )
                    if use_fast:
                        qin = QuantizedTensor.from_float(input, self.layout_type) if QUANT_ALGOS[self.quant_format].get("quantize_input", True) else input
                        return torch.nn.functional.linear(qin, qw, b)
                    qw = cast_to_input(qw.dequantize(), input, copy=False)
                return torch.nn.functional.linear(input, qw, b)

            def _expert_qt_from(self, weight: QuantizedTensor, i: int) -> QuantizedTensor:
                """Build a per-expert QuantizedTensor by indexing into a resident bank."""
                if weight._qdata.ndim == 2:
                    return self._bank_rows(weight, i, i + 1)
                params = weight._params
                kwargs = {
                    "scale": params.scale[i] if params.scale.dim() else params.scale,
                    "orig_dtype": params.orig_dtype,
                    "orig_shape": (self.out_features, self.in_features),
                }
                if hasattr(params, "block_scale"): # NVFP4
                    kwargs["block_scale"] = params.block_scale[i]
                if hasattr(params, "quant_group_size"):
                    kwargs["quant_group_size"] = params.quant_group_size
                if hasattr(params, "convrot"):
                    kwargs["convrot"] = params.convrot
                if hasattr(params, "convrot_groupsize"):
                    kwargs["convrot_groupsize"] = params.convrot_groupsize
                if hasattr(params, "linear_dtype"):
                    kwargs["linear_dtype"] = params.linear_dtype
                return QuantizedTensor(weight._qdata[i], weight._layout_cls, type(params)(**kwargs))

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix, extra_quant_conf={"num_experts": self.num_experts})

        class Embedding(manual_cast.Embedding):
            _disabled_storage_formats = disabled_storage

            def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
                weight_key = f"{prefix}weight"
                layer_conf = state_dict.pop(f"{prefix}comfy_quant", None)
                if layer_conf is not None:
                    layer_conf = json.loads(layer_conf.numpy().tobytes())

                # FP8 and tensorwise INT8 support per-row dequantization.
                quant_format = layer_conf.get("format") if layer_conf is not None else None
                manually_loaded_keys = []

                if quant_format in ("float8_e4m3fn", "float8_e5m2", "int8_tensorwise") and weight_key in state_dict:
                    self.quant_format = quant_format
                    qconfig = QUANT_ALGOS[quant_format]
                    self.layout_type = qconfig["comfy_tensor_layout"]
                    layout_cls = get_layout_class(self.layout_type)
                    weight = state_dict.pop(weight_key)
                    manually_loaded_keys.append(weight_key)

                    scale_key = f"{prefix}weight_scale"
                    scale = state_dict.pop(scale_key, None)
                    if scale is not None:
                        scale = scale.float()
                        manually_loaded_keys.append(scale_key)

                    extra = {}
                    if quant_format == "int8_tensorwise" and layer_conf.get("convrot", False):
                        extra["convrot"] = True
                        extra["convrot_groupsize"] = int(layer_conf.get("convrot_groupsize", 256))
                    params = layout_cls.Params(
                        scale=scale if scale is not None else torch.ones((), dtype=torch.float32),
                        orig_dtype=MixedPrecisionOps._compute_dtype,
                        orig_shape=(self.num_embeddings, self.embedding_dim),
                        **extra,
                    )
                    quantized_weight = QuantizedTensor(weight.to(dtype=qconfig["storage_t"]), qconfig["comfy_tensor_layout"], params)
                    if quant_format in self._disabled_storage_formats:
                        self.layout_type = None
                        self.weight = torch.nn.Parameter(quantized_weight.dequantize().to(dtype=MixedPrecisionOps._compute_dtype), requires_grad=False)
                    else:
                        self.weight = torch.nn.Parameter(quantized_weight, requires_grad=False)
                elif layer_conf is not None:
                    # Unsupported format — restore the marker so it round-trips; fall through to default load.
                    state_dict[f"{prefix}comfy_quant"] = torch.tensor(
                        list(json.dumps(layer_conf).encode('utf-8')), dtype=torch.uint8)

                super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
                for k in manually_loaded_keys:
                    if k in missing_keys:
                        missing_keys.remove(k)

            def state_dict(self, *args, destination=None, prefix="", **kwargs):
                sd = destination if destination is not None else {}
                return _quantized_weight_state_dict(self, sd, prefix)

            def host_rows(self, input, out_dtype=None):
                weight = self.weight
                if not isinstance(weight, QuantizedTensor):
                    return super().host_rows(input, out_dtype=out_dtype)
                if self.quant_format != "int8_tensorwise":
                    # TODO: fp8 tables still stream through the cast buffer
                    return self.forward_comfy_cast_weights(input, out_dtype=out_dtype)
                idx = input.reshape(-1).cpu()
                params = weight._params
                scale = params.scale[idx] if params.scale.dim() >= 2 else params.scale  # per-row scale is [vocab, 1], as dequantize_embedding expects
                params = dataclasses.replace(params, scale=scale.to(input.device))
                rows = weight._qdata[idx].to(input.device)
                x = get_layout_class(self.layout_type).dequantize_embedding(rows, params, torch.arange(idx.numel(), device=input.device))
                x = x.view(*input.shape, x.shape[-1])
                return x if out_dtype is None else x.to(dtype=out_dtype)

            def forward_comfy_cast_weights(self, input, out_dtype=None):
                weight = self.weight

                # Optimized path: lookup in fp8/INT8, dequantize only selected rows.
                if isinstance(weight, QuantizedTensor) and len(self.weight_function) == 0:
                    qdata, _, cast_state = _cast_weight_bias(
                        self,
                        input,
                        device=input.device,
                        dtype=weight.dtype,
                        want_requant=True,
                    )
                    if isinstance(qdata, QuantizedTensor):
                        params = qdata._params
                        scale = params.scale
                        qdata = qdata._qdata
                    else:
                        params = weight._params
                        scale = None

                    if self.quant_format == "int8_tensorwise":
                        output = get_layout_class(self.layout_type).dequantize_embedding(qdata, params, input)
                        output = _release_weight_bias(self, output, cast_state)
                        return output if out_dtype is None else output.to(dtype=out_dtype)

                    x = torch.nn.functional.embedding(
                        input, qdata, self.padding_idx, self.max_norm,
                        self.norm_type, self.scale_grad_by_freq, self.sparse)
                    x = _release_weight_bias(self, x, cast_state)
                    target_dtype = out_dtype if out_dtype is not None else weight._params.orig_dtype
                    x = x.to(dtype=target_dtype)
                    if scale is not None:
                        x = x * scale.to(dtype=target_dtype)
                    return x

                # Fallback for non-quantized or weight_function (LoRA) case
                return super().forward_comfy_cast_weights(input, out_dtype=out_dtype)

    return MixedPrecisionOps

def _requantize_kwargs(weight) -> dict:
    """The quantization a patched weight is requantized with: the one it was loaded in. An INT8
    ConvRot weight stays Hadamard-rotated with per-output-channel scales, and a per-channel INT8
    weight keeps its per-channel scales, instead of falling back to one unrotated tensorwise scale."""
    if not isinstance(weight, QuantizedTensor) or weight._layout_cls != "TensorWiseINT8Layout":
        return {}
    params = weight._params
    if getattr(params, "convrot", False):
        return {"per_channel": True, "convrot": True, "convrot_groupsize": int(params.convrot_groupsize)}
    if params.scale.numel() > 1:
        return {"per_channel": True}
    return {}


def get_disabled_quant_formats(device=None):
    """Quantized formats whose fast matmul must be emulated on ``device``."""
    disabled = set()
    if not mixed_precision_quantization_available():
        disabled.update({"float8_e4m3fn", "float8_e5m2", "nvfp4", "int8_tensorwise", "svdquant_w4a4", "awq_w4a16"})
    if not int8_quantization_available():
        disabled.add("int8_tensorwise")
    svdq_layout = get_layout_class("TensorCoreSVDQuantW4A4Layout")
    if svdq_layout is None or not svdq_layout.supports_fast_matmul():
        # SVDQuant W4A4 needs sm_80+ int4 tensor cores (layout MIN_SM (8,0)).
        disabled.add("svdquant_w4a4")
    if not model_management.supports_nvfp4_compute(device):
        disabled.add("nvfp4")
    if not model_management.supports_mxfp8_compute(device):
        disabled.add("mxfp8")
    if not model_management.supports_fp8_compute(device):
        disabled.add("float8_e4m3fn")
        disabled.add("float8_e5m2")
    if not model_management.supports_int8_compute(device):
        # int8 storage stays enabled: the dequantizing math fallback works
        # on every device, halving weight memory either way.
        disabled.add("int8_tensorwise")
        disabled.add("convrot_w4a4")
        disabled.update(_GROUPED_INT8_FORMATS)
    return disabled


def get_disabled_storage_formats():
    """Quantized formats that cannot even be kept quantized in memory."""
    disabled_storage = set()
    if not mixed_precision_quantization_available():
        disabled_storage.update({"float8_e4m3fn", "float8_e5m2", "mxfp8", "nvfp4", "int8_tensorwise", "svdquant_w4a4", "awq_w4a16"})
    if not int8_quantization_available():
        disabled_storage.add("int8_tensorwise")
    if not args.fp8_storage:
        disabled_storage.update({"float8_e4m3fn", "float8_e5m2", "mxfp8"})
    return disabled_storage


@contextlib.contextmanager
def use_quantized_matmul(model, device):
    disabled = get_disabled_quant_formats(device)
    previous = []
    try:
        for module in model.modules():
            if isinstance(module, MixedPrecisionOp) and module.can_use_quantized_matmul(disabled):
                previous.append((module, module._full_precision_mm))
                module._full_precision_mm = False
        yield
    finally:
        for module, full_precision_mm in previous:
            module._full_precision_mm = full_precision_mm


def pick_operations(weight_dtype, compute_dtype, load_device=None, disable_fast_fp8=False, fp8_optimizations=False, model_config=None, inference_mode: Optional[bool] = None):
    if inference_mode is None:
        inference_mode = current_execution_context().inference_mode
    if model_config and hasattr(model_config, 'quant_config') and model_config.quant_config:
        logger.info("Using mixed precision operations")
        disabled = get_disabled_quant_formats(load_device)
        disabled_storage = get_disabled_storage_formats()
        logger.info("Native ops: {} {}".format(", ".join(QUANT_ALGOS.keys() - disabled), ", emulated ops: {}".format(", ".join(disabled)) if len(disabled) > 0 else ""))
        return mixed_precision_ops(model_config.quant_config, compute_dtype, disabled=disabled, disabled_storage=disabled_storage)

    fp8_compute = model_management.supports_fp8_compute(load_device)
    if (
        fp8_compute and
        (fp8_optimizations or PerformanceFeature.Fp8MatrixMultiplication in args.fast) and
        not disable_fast_fp8
    ):
        return fp8_ops

    if (
        PerformanceFeature.CublasOps in args.fast and
        CUBLAS_IS_AVAILABLE and
        weight_dtype == torch.float16 and
        (compute_dtype == torch.float16 or compute_dtype is None)
    ):
        logger.debug("Using cublas ops")
        return cublas_ops

    if compute_dtype is None or weight_dtype == compute_dtype:
        return disable_weight_init if inference_mode else skip_init

    return manual_cast

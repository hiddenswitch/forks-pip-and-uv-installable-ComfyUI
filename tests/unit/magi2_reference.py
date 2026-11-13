"""Imports SandAI's MAGI-2 Preview inference package as the parity reference.

Set MAGI2_REFERENCE_SRC to a checkout of https://github.com/SandAI-org/MAGI-2-preview (the `inference/`
package). Its CUDA-only kernels and compiler hooks are replaced by plain PyTorch stand-ins that
implement the documented kernel contracts: FlashAttention-3 varlen forward returning (out, lse),
the sink LSE correction, the flash multi-head MoE gather/GEMM/scatter, flash-attn's rotary
helper and FlashAttention-2 with log-sum-exp. Everything else is SandAI's code, unmodified.
"""

import contextlib
import importlib
import importlib.machinery
import os
import sys
import types

import torch
import torch.nn.functional as F

REFERENCE_SRC = os.environ.get("MAGI2_REFERENCE_SRC")


def _dense_attention(q, k, v, softmax_scale=None):
    """q [Tq, H, D], k/v [Tk, Hkv, D] -> out [Tq, H, D], lse [H, Tq], all fp32 math."""
    heads = q.shape[1]
    if k.shape[1] != heads:
        k = k.repeat_interleave(heads // k.shape[1], dim=1)
        v = v.repeat_interleave(heads // v.shape[1], dim=1)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * scale
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v.float())
    return out.to(q.dtype), lse


def _fa3_fwd(q, k, v, _k_new, _v_new, _qv, _out, cu_seqlens_q, cu_seqlens_k, *args):
    softmax_scale = args[12]
    outs, lses = [], []
    bounds_q, bounds_k = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for index in range(len(bounds_q) - 1):
        out, lse = _dense_attention(q[bounds_q[index]:bounds_q[index + 1]], k[bounds_k[index]:bounds_k[index + 1]],
                                    v[bounds_k[index]:bounds_k[index + 1]], softmax_scale)
        outs.append(out)
        lses.append(lse)
    return torch.cat(outs, dim=0), torch.cat(lses, dim=1)


def _correct_out_lse_with_sink(out, lse, sink, sink_layout="sh"):
    """Zero-value sink logits [num_sink, H] join each row's softmax denominator."""
    if sink is None:
        return out, lse
    old = lse.float().transpose(0, 1)
    new = torch.logaddexp(old, torch.logsumexp(sink.float(), dim=0).unsqueeze(0))
    return out * torch.exp(old - new).unsqueeze(-1).to(out.dtype), new.transpose(0, 1)


def _mh_moe_fwd(x, gather_ids, probs, expert_offsets, W_gate, W_up, W_down):
    """flash_mh_moe: per expert gate/up GEMM -> SwiGLU7 -> down GEMM, scaled by the route probability."""
    output = torch.zeros_like(x)
    experts_per_head = (expert_offsets.numel() - 1) // x.shape[1]
    bounds = expert_offsets.tolist()
    for expert in range(len(bounds) - 1):
        begin, end = bounds[expert], bounds[expert + 1]
        if begin == end:
            continue
        head = expert // experts_per_head
        tokens = gather_ids[begin:end].long()
        rows = x.index_select(0, tokens)[:, head].float()
        gate = (rows @ W_gate[expert].float()).clamp(max=7.0)
        up = (rows @ W_up[expert].float()).clamp(min=-7.0, max=7.0)
        hidden = (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(W_down.dtype)
        y = (hidden.float() @ W_down[expert].float()) * probs[begin:end, None].float()
        output[:, head].index_add_(0, tokens, y.to(x.dtype))
    return output


def _flash_attn_func(query, key, value, return_attn_probs=False, **kwargs):
    out, lse = _dense_attention(query[0], key[0], value[0])
    if return_attn_probs:
        return out.unsqueeze(0), lse.unsqueeze(0), None
    return out.unsqueeze(0)


def _rotary(x, cos, sin, interleaved=False, **kwargs):
    rotary_dim = cos.shape[-1] * 2
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    head, tail = x[..., :rotary_dim], x[..., rotary_dim:]
    x1, x2 = head.chunk(2, dim=-1)
    return torch.cat((head * cos + torch.cat((-x2, x1), dim=-1) * sin, tail), dim=-1)


class _UnfoldNd(torch.nn.Module):
    """unfoldNd.UnfoldNd for the unit kernels MAGI-2 uses: [N, C, T, H, W] -> [N, C, T*H*W]."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        if tuple(kernel_size) != (1, 1, 1) or tuple(stride) != (1, 1, 1):
            raise NotImplementedError("only unit patches")

    def forward(self, x):
        return x.flatten(2)


def _identity_decorator(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return lambda target: target


_REGISTERED_OPS = set()


def _register_custom_op(name, infer_output_meta_fn=None, is_subgraph_boundary=False):
    namespace, op = name.split("::")

    def decorator(function):
        if namespace == "magi2" and name not in _REGISTERED_OPS:
            _REGISTERED_OPS.add(name)
            setattr(_ops_namespace(namespace), op, function)
        return function
    return decorator


def _ops_namespace(namespace):
    holder = getattr(torch.ops, namespace)
    return holder


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    module.__dict__.update(attributes)
    return module


def _install_stubs():
    sink = types.SimpleNamespace(correct_out_lse_with_sink=staticmethod(_correct_out_lse_with_sink))
    stubs = {
        "magi_compiler": _module("magi_compiler", magi_compile=_identity_decorator, magi_register_custom_op=_register_custom_op),
        "magi_compiler.api": _module("magi_compiler.api", magi_compile=_identity_decorator, magi_register_custom_op=_register_custom_op),
        "magi_compiler.config": _module("magi_compiler.config", CompileConfig=object),
        "flash_attn_interface": _module("flash_attn_interface", flash_attn_3_cuda=types.SimpleNamespace(fwd=_fa3_fwd)),
        "magi_attn_extensions": _module("magi_attn_extensions"),
        "magi_attn_extensions.fa3_interface_with_sink": _module(
            "magi_attn_extensions.fa3_interface_with_sink", FA3VarlenFuncWithSink=sink,
            fa3_func_with_sink=None, fa3_varlen_func_with_sink=None),
        "flash_attn": _module("flash_attn"),
        "flash_attn.layers": _module("flash_attn.layers"),
        "flash_attn.layers.rotary": _module("flash_attn.layers.rotary", apply_rotary_emb=_rotary),
        "flash_attn.flash_attn_interface": _module("flash_attn.flash_attn_interface", flash_attn_func=_flash_attn_func),
        "unfoldNd": _module("unfoldNd", UnfoldNd=_UnfoldNd),
    }
    for name, module in stubs.items():
        sys.modules.setdefault(name, module)


@contextlib.contextmanager
def _eager():
    previous = torch._dynamo.config.disable
    torch._dynamo.config.disable = True
    try:
        yield
    finally:
        torch._dynamo.config.disable = previous


def load():
    """SandAI preview, refiner, data proxies, config, turbo VAE and sampler modules."""
    _install_stubs()
    if REFERENCE_SRC not in sys.path:
        sys.path.insert(0, REFERENCE_SRC)
    with _eager():
        preview = importlib.import_module("inference.model.magi2_preview")
        refiner = importlib.import_module("inference.model.magi2_refiner")
        preview_proxy = importlib.import_module("inference.pipeline.preview_data_proxy")
        refiner_proxy = importlib.import_module("inference.pipeline.refiner_data_proxy")
        config = importlib.import_module("inference.common.magi2_config")
        turbo = importlib.import_module("inference.model.turbo_vaed")
        sampler = importlib.import_module("inference.pipeline.sampler")
    preview._mh_moe_fwd = _mh_moe_fwd
    torch.ops.magi2.flex_flash_attn_func = lambda q, k, v, q_ranges, k_ranges, *args: refiner._custom_flex_flash_attn_func(q, k, v, q_ranges, k_ranges)
    return types.SimpleNamespace(preview=preview, refiner=refiner, preview_proxy=preview_proxy, refiner_proxy=refiner_proxy,
                                 config=config, turbo=turbo, sampler=sampler, eager=_eager)


__all__ = ["REFERENCE_SRC", "load", "F"]

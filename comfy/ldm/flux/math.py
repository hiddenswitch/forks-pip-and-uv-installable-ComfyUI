from einops import rearrange
from torch import Tensor
import torch

from ... import model_management
from ... import quant_ops
from ..modules.attention import AttentionTensorContainer
from ..modules.attention import optimized_attention


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None, transformer_options={}, preferred_attention=None) -> Tensor:
    if isinstance(q, AttentionTensorContainer):
        q, k, v = q.take(), k.take(), v.take()
    if pe is not None:
        q, k = apply_rope(q, k, pe)
    heads = q.shape[1]
    q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
    x = optimized_attention(q, k, v, heads, skip_reshape=True, mask=mask, transformer_options=transformer_options, preferred_attention=preferred_attention)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    if pos.device.type == "mps":
        return _rope_cpu(pos, dim, theta)
    if not model_management.supports_fp64(pos.device):
        return _rope(pos, dim, theta, torch.device("cpu"))
    return _rope(pos, dim, theta, pos.device)


@torch.compiler.disable
def _rope_cpu(pos: Tensor, dim: int, theta: int) -> Tensor:
    # PyTorch 2.13 Inductor can fuse CPU float64 work into an unsupported MPS kernel.
    return _rope(pos, dim, theta, torch.device("cpu"))


def _rope(pos: Tensor, dim: int, theta: int, device) -> Tensor:
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def _apply_rope1(x: Tensor, freqs_cis: Tensor):
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]

    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])

    return x_out.reshape(*x.shape).type_as(x)


def _apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    return apply_rope1(xq, freqs_cis), apply_rope1(xk, freqs_cis)


def apply_rope(xq, xk, freqs_cis):
    if model_management.in_training:
        return _apply_rope(xq, xk, freqs_cis)
    else:
        return quant_ops.ck.apply_rope(xq, xk, freqs_cis)


def apply_rope1(x, freqs_cis):
    if model_management.in_training:
        return _apply_rope1(x, freqs_cis)
    else:
        return quant_ops.ck.apply_rope1(x, freqs_cis)

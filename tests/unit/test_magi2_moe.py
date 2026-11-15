"""MAGI-2 routed experts against the flash_mh_moe kernel contract.

The kernel accumulates the gate and up GEMMs in fp32, applies SwiGLU7 in fp32 and rounds once to
the down projection's dtype, accumulates the down GEMM in fp32, scales it by the route probability
and rounds once more. Two correct implementations then differ only by fp32 summation order, so the
tolerance here is far below the error of rounding the gate and up outputs to bf16.

Each token picks one expert per head: with several, the bf16 scatter-add of their outputs rounds
in whatever order the adds land (the kernel adds atomically), which is not what this test is about.
"""

import torch

import comfy.ops as comfy_ops
from comfy.ldm.magi2.model import MultiHeadMoE, ieee_fp32_matmul

from . import magi2_reference

TOLERANCE = 1e-4


def contract_inputs(module, x, gate, up, down):
    """Route x [S, H * d] with the module and lay the experts out as the kernel takes them."""
    tokens = x.shape[0]
    x_heads = x.view(tokens, module.heads, module.head_dim)
    gather_ids, probs, counts = [], [], []
    for head in range(module.heads):
        head_probs, indices = module.route(x_heads[:, head].contiguous(), head)
        order = indices.flatten().argsort(stable=True)
        gather_ids.append(order // module.top_k)
        probs.append(head_probs.flatten()[order])
        counts.append(torch.bincount(indices.flatten(), minlength=module.num_experts))
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64), torch.cat(counts).cumsum(0)))
    return dict(x=x_heads, gather_ids=torch.cat(gather_ids).to(torch.int32), probs=torch.cat(probs), expert_offsets=offsets,
                W_gate=gate.transpose(1, 2), W_up=up.transpose(1, 2), W_down=down.transpose(1, 2))


def kernel_contract(x, gather_ids, probs, expert_offsets, W_gate, W_up, W_down, hidden_dtype, gate_up_dtype=torch.float32):
    """The kernel contract with SwiGLU7's output rounded to hidden_dtype (the kernel uses W_down's dtype).

    gate_up_dtype other than fp32 rounds the gate and up outputs before SwiGLU7, which the kernel does not.
    """
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
        gate = (rows @ W_gate[expert].float()).to(gate_up_dtype).float().clamp(max=7.0)
        up = (rows @ W_up[expert].float()).to(gate_up_dtype).float().clamp(min=-7.0, max=7.0)
        hidden = (gate * torch.sigmoid(1.702 * gate) * (up + 1.0)).to(hidden_dtype)
        y = (hidden.float() @ W_down[expert].float()) * probs[begin:end, None].float()
        output[:, head].index_add_(0, tokens, y.to(x.dtype))
    return output


def relative_error(actual, expected):
    return ((actual.double() - expected.double()).norm() / expected.double().norm()).item()


def test_routed_experts_follow_the_kernel_contract():
    torch.manual_seed(0)
    module = MultiHeadMoE(512, 2, 8, 1, 128, 4.9, dtype=torch.bfloat16, operations=comfy_ops.disable_weight_init)
    with torch.no_grad():
        module.router.weight.normal_(0, 256 ** -0.5)
        module.router.bias.uniform_(-0.1, 0.1)
        for bank in (module.gate_proj, module.up_proj, module.down_proj):
            bank.weight.uniform_(-(3.0 / bank.in_features) ** 0.5, (3.0 / bank.in_features) ** 0.5)
    x = torch.randn(1, 200, 512).to(torch.bfloat16)
    experts = lambda bank: bank.weight.view(-1, bank.out_features, bank.in_features)
    inputs = contract_inputs(module, x[0], experts(module.gate_proj), experts(module.up_proj), experts(module.down_proj))
    with ieee_fp32_matmul(), torch.no_grad():
        actual = module(x)[0]
        expected = magi2_reference._mh_moe_fwd(**inputs).reshape(200, 512)
        contract = kernel_contract(**inputs, hidden_dtype=torch.bfloat16).reshape(200, 512)
        rounded = kernel_contract(**inputs, hidden_dtype=torch.bfloat16, gate_up_dtype=torch.bfloat16).reshape(200, 512)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(contract, expected, rtol=0, atol=0)
    assert relative_error(actual, expected) < TOLERANCE
    assert relative_error(rounded, expected) > 10 * TOLERANCE


def test_ieee_fp32_matmul_restores_the_previous_precision():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        with ieee_fp32_matmul():
            with ieee_fp32_matmul():
                assert torch.get_float32_matmul_precision() == "highest"
            assert torch.get_float32_matmul_precision() == "highest"
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)

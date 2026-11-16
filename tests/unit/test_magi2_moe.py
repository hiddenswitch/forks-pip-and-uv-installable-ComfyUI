"""MAGI-2 routed experts against the flash_mh_moe kernel contract.

The kernel accumulates the gate and up GEMMs in fp32, applies SwiGLU7 in fp32 and rounds once to
the down projection's dtype, accumulates the down GEMM in fp32, scales it by the route probability
and rounds once more. Two correct implementations then differ only by fp32 summation order, so the
tolerance here is far below the error of rounding the gate and up outputs to bf16.

Each token picks one expert per head: with several, the bf16 scatter-add of their outputs rounds
in whatever order the adds land (the kernel adds atomically), which is not what this test is about.
"""

import pytest
import torch

import comfy.ops as comfy_ops
from comfy.ldm.magi2.model import MultiHeadMoE

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
    offsets = torch.cat((torch.zeros(1, dtype=torch.int64, device=x.device), torch.cat(counts).cumsum(0)))
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


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_routed_experts_follow_the_kernel_contract(device):
    torch.manual_seed(0)
    module = MultiHeadMoE(512, 2, 8, 1, 128, 4.9, dtype=torch.bfloat16, device=device, operations=comfy_ops.disable_weight_init)
    with torch.no_grad():
        module.router.weight.normal_(0, 256 ** -0.5)
        module.router.bias.uniform_(-0.1, 0.1)
        for bank in (module.gate_proj, module.up_proj, module.down_proj):
            bank.weight.uniform_(-(3.0 / bank.in_features) ** 0.5, (3.0 / bank.in_features) ** 0.5)
    x = torch.randn(1, 200, 512, device=device).to(torch.bfloat16)
    experts = lambda bank: bank.weight.view(-1, bank.out_features, bank.in_features)
    inputs = contract_inputs(module, x[0], experts(module.gate_proj), experts(module.up_proj), experts(module.down_proj))
    with torch.no_grad():
        actual = module(x)[0]
        expected = magi2_reference._mh_moe_fwd(**inputs).reshape(200, 512)
        contract = kernel_contract(**inputs, hidden_dtype=torch.bfloat16).reshape(200, 512)
        rounded = kernel_contract(**inputs, hidden_dtype=torch.bfloat16, gate_up_dtype=torch.bfloat16).reshape(200, 512)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(contract, expected, rtol=0, atol=0)
    assert relative_error(actual, expected) < TOLERANCE
    assert relative_error(rounded, expected) > 10 * TOLERANCE


def test_batched_route_all_matches_per_head_route():
    torch.manual_seed(7)
    module = MultiHeadMoE(512, 4, 8, 2, 128, 4.9, dtype=torch.bfloat16, operations=comfy_ops.disable_weight_init)
    with torch.no_grad():
        module.router.weight.normal_(0, 128 ** -0.5)
        module.router.bias.uniform_(-0.1, 0.1)
    x = torch.randn(32, 4, 128, dtype=torch.bfloat16)
    all_probs, all_indices = module.route_all(x)
    for head in range(module.heads):
        probs, indices = module.route(x[:, head].contiguous(), head)
        torch.testing.assert_close(all_probs[head], probs, rtol=1e-6, atol=1e-6)
        assert torch.equal(all_indices[head], indices)


def test_grouped_gate_up_rotates_shared_convrot_input_once(monkeypatch):
    import comfy.ldm.magi2.model as magi2_model

    calls = []
    real_rotate = magi2_model._rotate_activation

    def counting_rotate(x, h, group):
        calls.append(group)
        return real_rotate(x, h, group)

    monkeypatch.setattr(magi2_model, "_rotate_activation", counting_rotate)
    x = torch.randn(12, 64, dtype=torch.bfloat16)
    offsets = torch.tensor([4, 12], dtype=torch.int32)
    gate_bank = (torch.randn(2, 32, 64, dtype=torch.float32), 64)
    up_bank = (torch.randn(2, 32, 64, dtype=torch.float32), 64)
    gate, up = magi2_model._grouped_gate_up(x, gate_bank, up_bank, offsets)
    assert calls == [64]
    torch.testing.assert_close(gate, magi2_model._grouped_projection(x, gate_bank, offsets))
    torch.testing.assert_close(up, magi2_model._grouped_projection(x, up_bank, offsets))

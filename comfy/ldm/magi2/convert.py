"""SandAI MAGI-2 checkpoint keys -> the ComfyUI MAGI-2 layout.

Modality-grouped linears ([3 * out, in] in the release) become one linear per modality, the
expert banks become [experts, out, in] like every other ComfyUI linear, and the router keeps the
EMA selection bias the reference inference loads in place of the raw training bias.
"""

import re

MODALITY_LINEARS = ("attention.linear_qkv", "attention.linear_g", "attention.linear_proj", "mlp.up_gate_proj", "mlp.down_proj")
ALWAYS_MODALITY_LINEARS = ("mlp.modality_specific_shared_expert_fc1", "mlp.modality_specific_shared_expert_fc2")
NUM_MODALITIES = 3
_LAYER = re.compile(r"^block\.layers\.(\d+)\.(.+)$")
_EXPERT_BANKS = {"mlp.moe_mlp.W_gate": "mlp.moe_mlp.gate_proj.weight",
                 "mlp.moe_mlp.W_up": "mlp.moe_mlp.up_proj.weight",
                 "mlp.moe_mlp.W_down": "mlp.moe_mlp.down_proj.weight"}


def comfy_tensors(name, tensor, multimodal_layers):
    """The ComfyUI (key, tensor) pairs for one release tensor; the raw router bias maps to none."""
    match = _LAYER.match(name)
    if match is None:
        return [(name, tensor)]
    index, rest = int(match.group(1)), match.group(2)
    prefix = f"block.layers.{index}."
    module = rest.removesuffix(".weight")
    if rest.endswith(".weight") and (module in ALWAYS_MODALITY_LINEARS or (module in MODALITY_LINEARS and index in multimodal_layers)):
        return [(f"{prefix}{module}.{modality}.weight", chunk) for modality, chunk in enumerate(tensor.chunk(NUM_MODALITIES))]
    if rest in _EXPERT_BANKS:
        return [(prefix + _EXPERT_BANKS[rest], tensor.transpose(1, 2))]
    if rest == "mlp.moe_mlp.gate":
        return [(prefix + "mlp.moe_mlp.router.weight", tensor)]
    if rest == "mlp.moe_mlp.router.expert_bias_ema":
        return [(prefix + "mlp.moe_mlp.router.bias", tensor)]
    if rest == "mlp.moe_mlp.router.expert_bias":
        return []
    return [(name, tensor)]


def multimodal_layers_from_shapes(shapes, hidden_size):
    """Layers whose attention pre-norm holds one scale vector per modality."""
    layers = set()
    for name, shape in shapes.items():
        match = _LAYER.match(name)
        if match is not None and match.group(2) == "attention.pre_norm.weight" and shape[0] == NUM_MODALITIES * hidden_size:
            layers.add(int(match.group(1)))
    return layers

import json

import torch

import comfy.ops as comfy_ops
from comfy import model_detection
from comfy.ldm.magi2.model import Magi2Model
from comfy.ldm.magi2.refiner import Magi2RefinerModel

PREVIEW = dict(num_layers=5, hidden_size=256, head_dim=64, num_query_groups=2, video_in_channels=8, audio_in_channels=6,
               text_in_channels=24, intermediate_size=128, multimodal_layers=(0, 4), moe_layers=(1, 2, 3), num_streams=4,
               moe_heads=2, num_experts=4, expert_intermediate_size=32, shared_expert_intermediate_size=48,
               modality_shared_expert_intermediate_size=16)
REFINER = dict(num_layers=3, hidden_size=256, head_dim=64, num_query_groups=2, video_in_channels=8, audio_in_channels=6,
               text_in_channels=24, intermediate_size=96, multimodal_layers=(0, 2))


def _state(model):
    return {key: torch.empty(value.shape, dtype=value.dtype, device="meta") for key, value in model.state_dict().items()}


def test_detects_preview_config():
    model = Magi2Model(**PREVIEW, dtype=torch.bfloat16, device="meta", operations=comfy_ops.disable_weight_init)
    metadata = {"config": json.dumps({"transformer": {"top_k": 2, "route_scale": 2.5, "text_in_channels": 24}})}
    config = model_detection.detect_unet_config(_state(model), "", metadata=metadata)
    expected = {key: value for key, value in PREVIEW.items()}
    assert config.pop("image_model") == "magi2"
    assert config.pop("top_k") == 2 and config.pop("route_scale") == 2.5
    assert config == expected


def test_detects_refiner_config_with_prefix():
    model = Magi2RefinerModel(**REFINER, dtype=torch.bfloat16, device="meta", operations=comfy_ops.disable_weight_init)
    state = {"model.diffusion_model." + key: value for key, value in _state(model).items()}
    config = model_detection.detect_unet_config(state, "model.diffusion_model.", metadata={"config": json.dumps({"transformer": {"text_in_channels": 24}})})
    assert config.pop("image_model") == "magi2_refiner"
    assert config == REFINER


def test_magi2_detection_needs_every_signature_key():
    model = Magi2Model(**PREVIEW, dtype=torch.bfloat16, device="meta", operations=comfy_ops.disable_weight_init)
    state = _state(model)
    del state["block.layers.1.mlp.moe_mlp.router.weight"], state["block.layers.2.mlp.moe_mlp.router.weight"], state["block.layers.3.mlp.moe_mlp.router.weight"]
    assert model_detection.detect_unet_config(state, "")["moe_layers"] == ()
    del state["block.layers.0.attention.q_norm.weight"]
    assert model_detection.detect_unet_config(state, "") is None

from unittest.mock import patch

import pytest
import torch

from comfy import ops
from comfy.text_encoders import magi2, qwen35

transformers = pytest.importorskip("transformers")

TINY = dict(vocab_size=512, hidden_size=64, intermediate_size=96, num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=1, head_dim=32,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16)


def test_magi2_conditioning_is_the_hidden_state_three_layers_from_the_top():
    torch.manual_seed(0)
    config = transformers.Qwen3_5TextConfig(**TINY, linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
                                            rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                                                             "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    reference = transformers.Qwen3_5TextModel(config).float().eval()
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.05)

    class Built(Exception):
        pass

    def record(**kwargs):
        raise Built(kwargs)

    with patch.object(magi2.qwen35, "Qwen35ClipModel", record), pytest.raises(Built) as built:
        magi2.Magi2TEModel(device="meta", dtype=torch.float32, model_options={})
    assert {key: built.value.args[0][key] for key in ("layer", "layer_idx", "model_type")} == {"layer": "hidden", "layer_idx": -3, "model_type": "qwen35_27b"}
    model = qwen35.Qwen35ClipModel(layer="hidden", layer_idx=-3, model_type=magi2.MODEL_TYPE, dtype=torch.float32, device="cpu",
                                   model_options={"custom_operations": ops.disable_weight_init},
                                   textmodel_json_config={**TINY, "layer_types": config.layer_types, "mrope_section": [2, 1, 1], "lm_head": False})
    missing, unexpected = model.transformer.load_state_dict({"model." + key: value for key, value in reference.state_dict().items()}, strict=False)
    assert not unexpected and all(key.startswith("visual.") for key in missing)

    tokens = torch.randint(0, 512, (1, 11), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = reference(input_ids=tokens, output_hidden_states=True).hidden_states[-3]
        actual = model.encode_token_weights([[(int(token), 1.0) for token in tokens[0]]])[0]
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-4, atol=1e-4)

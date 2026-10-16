import torch

from comfy import ops
from comfy.text_encoders.qwen35 import Qwen35Config, Qwen35Transformer, _qwen35_layer_types


def test_qwen35_encodes_with_attention_mask():
    # MOSS-Image conditions on Qwen3.5 with the text encoder's attention mask
    config = Qwen35Config(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                          head_dim=64, linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                          mrope_section=[3, 3, 2], layer_types=_qwen35_layer_types(4))
    torch.manual_seed(0)
    model = Qwen35Transformer(config, dtype=torch.float32, ops=ops.manual_cast)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    model.requires_grad_(False)
    ids = torch.tensor([[5, 6, 7, 8]])
    masked, _ = model(ids, attention_mask=torch.ones(1, 4))
    unmasked, _ = model(ids)
    assert torch.equal(masked, unmasked)

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from comfy import supported_models
from comfy.tensor_parallel import loader as tensor_loader
from comfy.xdit.loader import _load_xdit_rank


@pytest.mark.parametrize("strategy", ["ulysses", "ring"])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("model_type,family", [(supported_models.Flux2, "flux2"), (supported_models.MiniMaxH3, "minimax_h3")])
@pytest.mark.parametrize("quantized", [False, True])
def test_sequence_parallel_rank_loads_complete_weights(monkeypatch, strategy, rank, model_type, family, quantized):
    weight = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8)
    key = "double_blocks.0.img_attn.qkv.weight"
    reader = Mock()
    reader.load_keys.return_value = {key: weight}
    model = Mock()
    configurations = []

    def get_model(config, state, prefix, device):
        configurations.append(config.custom_operations.xdit_sequence_parallel)
        return model

    monkeypatch.setattr(model_type, "get_model", get_model)
    monkeypatch.setattr(tensor_loader, "get_model_patcher_class", lambda disabled: (
        lambda model, **kwargs: SimpleNamespace(model=model, is_dynamic=lambda: False)
    ))
    operations = SimpleNamespace(rank=rank, world_size=2)
    config = model_type({"image_model": family, "hidden_size": 8})
    config.quant_config = {} if quantized else None
    result = _load_xdit_rank(
        reader, config, {}, "", {key: key},
        "model.safetensors", {}, True, torch.float32, torch.device("cpu"),
        operations, strategy,
    )

    assert result.model is model
    loaded = model.load_model_weights.call_args.args[0]
    torch.testing.assert_close(loaded[key], weight)
    assert configurations[0].operations is operations
    assert configurations[0].strategy == strategy

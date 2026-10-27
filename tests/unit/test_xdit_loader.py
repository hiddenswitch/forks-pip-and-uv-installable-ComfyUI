from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from comfy import supported_models
from comfy.tensor_parallel import loader as tensor_loader
from comfy.xdit.loader import _load_xdit_rank


@pytest.mark.parametrize("strategy", ["ulysses", "ring"])
@pytest.mark.parametrize("rank", [0, 1])
def test_sequence_parallel_rank_loads_complete_weights(monkeypatch, strategy, rank):
    weight = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8)
    key = "double_blocks.0.img_attn.qkv.weight"
    reader = Mock()
    reader.load_keys.return_value = {key: weight}
    model = Mock()
    configurations = []

    def get_model(config, state, prefix, device):
        configurations.append(config.custom_operations.xdit_sequence_parallel)
        return model

    monkeypatch.setattr(supported_models.Flux2, "get_model", get_model)
    monkeypatch.setattr(tensor_loader, "get_model_patcher_class", lambda disabled: (
        lambda model, **kwargs: SimpleNamespace(model=model, is_dynamic=lambda: False)
    ))
    operations = SimpleNamespace(rank=rank, world_size=2)
    result = _load_xdit_rank(
        reader, supported_models.Flux2({"image_model": "flux2", "hidden_size": 8}), {}, "", {key: key},
        "model.safetensors", {}, True, torch.float32, torch.device("cpu"),
        operations, strategy,
    )

    assert result.model is model
    loaded = model.load_model_weights.call_args.args[0]
    torch.testing.assert_close(loaded[key], weight)
    assert configurations[0].operations is operations
    assert configurations[0].strategy == strategy

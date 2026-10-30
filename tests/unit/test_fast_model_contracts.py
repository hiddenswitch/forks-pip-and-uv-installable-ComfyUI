"""Explicit per-model forward coverage and visible gaps, independent of workflows."""

import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock
import pytest
import torch
from comfy import model_base, supported_models
from comfy_execution.graph_utils import GraphBuilder
from tests.fast_dit_cases import forward_contract

CONTRACTS = json.loads(
    (Path(__file__).parents[1] / "fast_model_contracts.json").read_text()
)


def test_contract_inventory_matches_registry():
    root = Path(__file__).resolve().parents[2]
    assert (
        Path(supported_models.__file__).resolve() == root / "comfy/supported_models.py"
    )
    assert set(CONTRACTS) == {model.__name__ for model in supported_models.models}
    for entry in CONTRACTS.values():
        assert set(entry) in ({"test"}, {"test", "kwargs"}, {"gap"})
        assert next(iter(entry.values()))


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_model_forward_contract(name):
    entry = CONTRACTS[name]
    if "gap" in entry:
        pytest.skip(entry["gap"])
    module, function = entry["test"].split(":")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        getattr(importlib.import_module(module), function)(**entry.get("kwargs", {}))


def test_forward_contract_rejects_constant_output(monkeypatch):
    monkeypatch.setattr(
        model_base.BaseModel,
        "apply_model",
        lambda self, x, *args, **kwargs: torch.zeros_like(x),
    )
    with pytest.raises(AssertionError, match="conditioning did not affect"):
        forward_contract("SD3")


@pytest.mark.asyncio
async def test_empty_execution_check_rejects_missing_error():
    module = importlib.import_module("tests.execution.test_execution")
    client = AsyncMock()
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        await module.TestExecution().test_partial_execution_empty_list(
            client, GraphBuilder(prefix="missing-error")
        )


def fibo2_forward_contract():
    import torch
    from tests.unit.test_fibo2 import _ours_model, _inputs, patchify, comfy_context
    from comfy import model_detection

    model = _ours_model()
    latents, timestep, text = _inputs(text_len=9)
    packed = patchify(latents)
    with torch.no_grad():
        result = model(packed, timestep, comfy_context(text))
        changed = model(packed + 1, timestep, comfy_context(text))
    assert result.shape == packed.shape
    assert result.dtype == packed.dtype
    assert torch.isfinite(result).all()
    assert not torch.allclose(result, changed)
    state = {
        key: torch.empty(value.shape, device="meta")
        for key, value in model.state_dict().items()
    }
    assert model_detection.detect_unet_config(state, "")["image_model"] == "fibo2"

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from comfy import model_management, supported_models
from comfy_extras.nodes.nodes_multigpu import _force_supported_compute_dtype


@pytest.mark.parametrize("bf16_supported,expected", [(True, torch.bfloat16), (False, torch.float32)])
def test_select_device_respects_qwen_compute_dtypes(monkeypatch, bf16_supported, expected):
    monkeypatch.setattr(model_management, "should_use_fp16", lambda *a, **kw: True)
    monkeypatch.setattr(model_management, "should_use_bf16", lambda *a, **kw: bf16_supported)
    monkeypatch.setattr(model_management, "PRIORITIZE_FP16", True)
    patcher = SimpleNamespace(
        model=SimpleNamespace(model_config=supported_models.QwenImage),
        model_dtype=lambda: torch.float8_e4m3fn,
        set_model_compute_dtype=Mock(),
    )
    _force_supported_compute_dtype(patcher, torch.device("cuda:0"))
    patcher.set_model_compute_dtype.assert_called_once_with(expected)

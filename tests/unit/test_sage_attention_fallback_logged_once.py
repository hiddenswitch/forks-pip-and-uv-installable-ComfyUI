import logging
from unittest import mock

import torch

from comfy.ldm.modules import attention


def test_a_sage_attention_failure_is_logged_once_per_cause(caplog):
    """Ideogram 4's head_dim 256 failed sage attention on every attention call of every step (38k ERROR lines in one
    eval); the fallback to PyTorch attention is said once per distinct cause."""
    q = k = v = torch.randn(1, 128, 2 * 16)

    def failing(*args, **kwargs):
        raise ValueError("Unsupported head_dim: 16")

    with mock.patch.object(attention, "sageattn", failing), caplog.at_level(logging.WARNING):
        for _ in range(3):
            out = attention.attention_sage(q, k, v, heads=2)
    assert out.shape == (1, 128, 32)
    assert sum("sage attention" in r.getMessage() for r in caplog.records) == 1

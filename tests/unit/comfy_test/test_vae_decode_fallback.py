from unittest.mock import Mock

import pytest
import torch

from comfy import model_management
from comfy.sd import VAE


@pytest.mark.parametrize(("compression", "tile"), [(4, 64), (8, 64), (16, 32), (32, 16)])
def test_image_decode_fallback_limits_output_tile_size(monkeypatch, compression, tile):
    vae = VAE.__new__(VAE)
    vae.first_stage_model = Mock()
    vae.latent_dim = 2
    vae.extra_1d_channel = None
    vae.handles_tiling = False
    vae.disable_offload = False
    vae.upscale_ratio = compression
    vae.device = vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.float32
    vae.memory_used_decode = Mock(side_effect=torch.OutOfMemoryError())
    vae.decode_tiled_ = Mock(return_value=torch.zeros(1, 3, 16, 16))
    monkeypatch.setattr(model_management, "soft_empty_cache", lambda: None)

    latent = torch.zeros(1, 4, 2, 2)
    assert vae.decode(latent).shape == (1, 16, 16, 3)
    vae.decode_tiled_.assert_called_once_with(latent, tile_x=tile, tile_y=tile, overlap=tile // 4)

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


@pytest.mark.parametrize("free_gib, tiled", [(12, True), (120, False)])
def test_magi_managed_decode_tiles_only_when_full_clip_exceeds_memory(monkeypatch, free_gib, tiled):
    from comfy import managed_memory
    from comfy.ldm.magi2.turbo_vae import TurboVAEDecoder

    vae = VAE(no_init=True)
    vae.first_stage_model = TurboVAEDecoder.__new__(TurboVAEDecoder)
    torch.nn.Module.__init__(vae.first_stage_model)
    pixels = torch.zeros(1, 3, 1, 16, 16)
    full = Mock(return_value=pixels)
    monkeypatch.setattr(vae.first_stage_model, "decode", full)
    vae.latent_dim = 3
    vae.extra_1d_channel = None
    vae.handles_tiling = vae.disable_offload = False
    vae.upscale_ratio = (lambda t: 4 * t - 3, 16, 16)
    vae.device = vae.output_device = torch.device("cpu")
    vae.vae_dtype = torch.bfloat16
    vae.process_output = lambda x: x
    vae.patcher = Mock()
    vae.patcher.get_free_memory.return_value = free_gib * 1024**3
    vae.memory_used_decode = lambda shape, dtype: 3000 * min(shape[2], 8) * shape[3] * shape[4] * 256 * 2
    vae.decode_tiled_3d = Mock(return_value=pixels)
    monkeypatch.setattr(managed_memory, "is_active", lambda: True)
    monkeypatch.setattr(model_management, "load_models_gpu", Mock())
    monkeypatch.setattr(model_management, "soft_empty_cache", Mock())
    monkeypatch.setattr(model_management, "get_total_memory", lambda device: free_gib * 1024**3)
    latent = torch.zeros(1, dtype=torch.bfloat16).expand(1, 48, 63, 68, 120)
    assert vae.decode(latent).shape == (1, 1, 16, 16, 3)
    assert vae.decode_tiled_3d.called == tiled
    assert full.called != tiled

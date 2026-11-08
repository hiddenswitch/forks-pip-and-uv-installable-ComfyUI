from unittest.mock import MagicMock

import torch



from comfy import nested_tensor  # noqa: E402
from comfy.nodes import base_nodes  # noqa: E402


def test_vae_decode_tiled_unwraps_nested_tensor():
    video = torch.zeros(1, 4, 2, 8, 8)
    audio = torch.zeros(1, 2, 2, 40)
    samples = {"samples": nested_tensor.NestedTensor((video, audio))}

    vae = MagicMock()
    vae.temporal_compression_decode.return_value = None
    vae.spacial_compression_decode.return_value = 8
    vae.decode_tiled.return_value = torch.zeros(1, 3, 2, 8, 8)

    base_nodes.VAEDecodeTiled().decode(vae, samples, tile_size=512)

    decoded_arg = vae.decode_tiled.call_args[0][0]
    assert decoded_arg is video

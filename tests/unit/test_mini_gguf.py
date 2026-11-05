"""Native GGUF loading, quantization and inference at reduced parameter shapes."""
import ctypes.util
import os

import gguf
import pytest
import torch

from comfy import utils
from comfy.gguf import GGMLOps, GGMLTensor, dequantize_tensor
from tests.inference.mini_gguf import quantize
from tests.inference.mini_weights import miniature_models


@pytest.mark.parametrize('kind', [gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q5_K, gguf.GGMLQuantizationType.Q6_K])
def test_native_gguf_checkpoint_shrinks_and_runs(tmp_path, kind):
    if not (os.environ.get('COMFYUI_TEST_GGML_LIBRARY') or ctypes.util.find_library('ggml-base')):
        pytest.skip('requires native libggml-base; set COMFYUI_TEST_GGML_LIBRARY')
    path = str(tmp_path / 'weights.gguf')
    writer = gguf.GGUFWriter(path, 'qwen_image')
    writer.add_tensor('weight', quantize(torch.zeros(8, 512), kind).numpy(), raw_dtype=kind)
    writer.add_tensor('bias', torch.linspace(-0.1, 0.1, 8).numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    source = utils.load_torch_file(path)
    assert source['weight'].shape == (8, 512)
    assert dequantize_tensor(source['weight']).count_nonzero() == 0
    with miniature_models(sources=[path]) as loaded:
        layer = GGMLOps.Linear(256, 4)
        layer.load_state_dict(utils.load_torch_file(path), strict=True)
        assert loaded == ['Linear']
        assert isinstance(layer.weight, GGMLTensor)
        assert layer.weight.tensor_type == kind and layer.weight.shape == (4, 256)
        weight = dequantize_tensor(layer.weight, dtype=torch.float32)
        assert 0.8 < weight.std().item() * 256 ** 0.5 < 1.2
        x = torch.randn(3, 256, generator=torch.Generator().manual_seed(7))
        with torch.inference_mode():
            actual = layer(x)
        expected = torch.nn.functional.linear(x, weight, torch.linspace(-0.1, 0.1, 8)[:4])
        torch.testing.assert_close(actual, expected)
        assert torch.isfinite(actual).all() and actual.std() > 0.01

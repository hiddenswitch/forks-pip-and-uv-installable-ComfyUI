"""Numerical smoke coverage through the production checkpoint loader and Flux blocks."""
from pathlib import Path
import json
import struct

import pytest
import torch
from safetensors.torch import save_file

from comfy import ops, quant_ops, sd, utils
from tests.inference.mini_weights import miniature_models


def test_lora_checkpoint_changes_and_restores_real_model_weight(checkpoint_header, tmp_path):
    header = json.loads((Path(__file__).parents[1] / 'inference/fixtures/flux1-dev-header.json').read_text())
    out_features, in_features = header['img_in.weight']['shape']
    path = str(tmp_path / 'adapter.safetensors')
    prefix = 'diffusion_model.img_in'
    save_file({prefix + '.lora_up.weight': torch.zeros(out_features, 64),
               prefix + '.lora_down.weight': torch.zeros(64, in_features),
               prefix + '.alpha': torch.tensor(64.)}, path)
    with miniature_models(sources=[checkpoint_header, path], width=128):
        model = sd.load_diffusion_model(checkpoint_header, model_options={
            'dtype': torch.float32, 'load_device': torch.device('cpu'), 'offload_device': torch.device('cpu'),
        }, disable_dynamic=True)
        patched, _ = sd.load_lora_for_models(model, None, utils.load_torch_file(path), 0.5, 0)
        key = prefix + '.weight'
        before = model.model.diffusion_model.img_in.weight.detach().clone()
        adapter = patched.patches[key][0][1]
        up, down, alpha, *_ = adapter.weights
        assert up.shape == (128, 8) and down.shape == (8, in_features)
        patched.patch_weight_to_device(key, device_to=torch.device('cpu'))
        actual = model.model.diffusion_model.img_in.weight
        expected = before + 0.5 * (alpha / down.shape[0]) * (up @ down)
        torch.testing.assert_close(actual, expected)
        assert not torch.allclose(actual, before)
        patched.unpatch_model()
        torch.testing.assert_close(model.model.diffusion_model.img_in.weight, before)


@pytest.fixture
def checkpoint_header(tmp_path):
    # Unmodified header from FLUX.1-dev revision 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21.
    header = (Path(__file__).parents[1] / 'inference/fixtures/flux1-dev-header.json').read_bytes().strip()
    path = tmp_path / 'flux1-dev.safetensors'
    path.write_bytes(struct.pack('<Q', len(header)) + header)
    return str(path)


@pytest.mark.parametrize('hidden_size,spatial', [(128, (8, 8)), (256, (7, 9))])
def test_checkpoint_load_and_numerical_forward(checkpoint_header, hidden_size, spatial, caplog):
    reader = utils.load_torch_file
    generator = torch.Generator().manual_seed(12)
    with miniature_models(sources=[checkpoint_header], width=hidden_size):
        patcher = sd.load_diffusion_model(checkpoint_header, model_options={
            'dtype': torch.float32, 'load_device': torch.device('cpu'), 'offload_device': torch.device('cpu'),
        }, disable_dynamic=True)
        model = patcher.model
        assert model.diffusion_model.params.hidden_size == hidden_size
        assert len(model.diffusion_model.double_blocks) == len(model.diffusion_model.single_blocks) == 2
        assert sum(p.numel() for p in model.parameters()) < 10_000_000
        assert not any('missing' in r.message.lower() or 'left over keys' in r.message.lower() for r in caplog.records)
        x = torch.randn((1, 16, *spatial), generator=generator)
        conditioning = dict(c_crossattn=torch.randn((1, 4, 4096), generator=generator),
                            y=torch.randn((1, 768), generator=generator), guidance=torch.ones(1))
        with torch.inference_mode():
            result = model.apply_model(x, torch.ones(1), **conditioning)
            repeated = model.apply_model(x, torch.ones(1), **conditioning)
            conditioning['c_crossattn'] = -conditioning['c_crossattn']
            changed = model.apply_model(x, torch.ones(1), **conditioning)
        assert result.shape == x.shape and result.dtype == x.dtype
        assert torch.isfinite(result).all() and result.std() > 0.01
        torch.testing.assert_close(result, repeated)
        assert not torch.allclose(result, changed)
        del patcher, model
    assert utils.load_torch_file is reader


def test_reader_restored_on_failure(checkpoint_header):
    reader = utils.load_torch_file
    state_reader = torch.nn.Module.load_state_dict
    with pytest.raises(RuntimeError, match='consumer failure'):
        with miniature_models(sources=[checkpoint_header]):
            raise RuntimeError('consumer failure')
    assert utils.load_torch_file is reader
    assert torch.nn.Module.load_state_dict is state_reader


@pytest.mark.parametrize('dtype,descriptor', [
    (torch.float8_e4m3fn, b'{"format":"float8_e4m3fn"}'),
    (torch.int8, b'{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":16}'),
])
@pytest.mark.parametrize('kind', ['Linear', 'Embedding'])
def test_quantized_checkpoint_loads_real_resized_weight(tmp_path, dtype, descriptor, kind):
    path = str(tmp_path / 'quantized.safetensors')
    weight = torch.zeros(64, 128, dtype=dtype)
    scale = torch.full((64, 1), 0.01) if dtype == torch.int8 else torch.tensor(0.01)
    save_file({
        'weight': weight,
        **({'bias': torch.linspace(-0.1, 0.1, 64)} if kind == 'Linear' else {}),
        'weight_scale': scale,
        'comfy_quant': torch.tensor(list(descriptor), dtype=torch.uint8),
    }, path)
    with miniature_models(sources=[path]):
        state = utils.load_torch_file(path)
        layer = getattr(ops.mixed_precision_ops(compute_dtype=torch.float32), kind)(32, 32, device='cpu')
        layer.load_state_dict(state, strict=True)
        assert isinstance(layer.weight, quant_ops.QuantizedTensor)
        assert tuple(layer.weight.shape) == (32, 32)
        assert layer.quant_format == json.loads(descriptor)['format']
        weights = layer.weight.dequantize().float()
        assert 0.8 < weights.std().item() * 32 ** 0.5 < 1.2
        assert weights.mean().abs() < 0.03
        # Exactly representable FP8 inputs isolate checkpoint resizing from input rounding.
        x = torch.randint(-8, 9, (3, 32), generator=torch.Generator().manual_seed(12)).float() / 8
        if kind == 'Embedding':
            x = torch.tensor([[0, 3, 31], [2, 7, 4]])
        with torch.inference_mode():
            actual = layer(x)
            expected = (torch.nn.functional.linear(x, layer.weight.dequantize(), layer.bias)
                        if kind == 'Linear' else layer.weight.dequantize()[x])
        assert actual.shape == expected.shape and torch.isfinite(actual).all()
        assert actual.std() > 0.01
        # INT8's native linear also dynamically quantizes activations row-wise.
        tolerance = {'atol': 0.01, 'rtol': 0.01} if dtype == torch.int8 else {}
        torch.testing.assert_close(actual, expected, **tolerance)

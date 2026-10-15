import torch

from comfy import model_detection, supported_models
from comfy.model_downloader import KNOWN_UNET_MODELS
from comfy.nodes.base_nodes import CLIPLoader
from comfy.text_encoders.flux import Flux3Tokenizer


def test_flux3_detection_and_packaged_text_encoder():
    state = {
        'content_mode_blocks.image.0.attn_out.weight': torch.empty(256, 256, device='meta'),
        'single_stream_modulations.txt.lin.weight': torch.empty(768, 256, device='meta'),
        'txt_mode_blocks.0.qkv.weight': torch.empty(768, 256, device='meta'),
        'single_blocks.0.qkv.weight': torch.empty(768, 256, device='meta'),
        'ref_time_in.in_layer.weight': torch.empty(256, 256, device='meta'),
    }
    config = model_detection.detect_unet_config(state, '')
    assert config['image_model'] == 'flux3'
    assert config['flow_map'] and config['fused_qkv']
    assert config['num_heads'] == 2
    assert config['depth'] == config['depth_single_blocks'] == 1
    model = supported_models.Flux3(config)
    assert model.clip_target().tokenizer is Flux3Tokenizer
    assert model.get_model({}, device=torch.device('meta')).diffusion_model.flow_map
    assert 'flux3' in CLIPLoader.INPUT_TYPES()['required']['type'][0]


def test_flux3_int8_models_are_discoverable():
    files = {str(model): model for model in KNOWN_UNET_MODELS.data}
    for name in ('flux3_image_klein_int8_convrot.safetensors', 'flux3_image_klein_sd_int8_convrot.safetensors'):
        assert files[name].repo_id == 'black-forest-labs/flux-3-image-klein-comfy'
        assert files[name].revision == '849a7a417c54120ac12f249da656b84ec05fd7eb'

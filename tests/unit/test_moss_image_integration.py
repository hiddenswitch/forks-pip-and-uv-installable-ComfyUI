import torch

from comfy import lora, model_detection, supported_models
from comfy.model_downloader import KNOWN_CLIP_MODELS, KNOWN_UNET_MODELS
from comfy.nodes.base_nodes import CLIPLoader
from comfy.text_encoders.moss_image import MOSSImageTokenizer

MOSS_REVISION = '9df12d99a97c9818afc555125829f9ed40b87b24'


def _moss_state(fused_qkv):
    dim, ffn = 256, 512
    state = {
        'shared_block_modulation.weight': torch.empty(6 * dim, dim, device='meta'),
        'layers.0.proj.ffn_up_gate.weight': torch.empty(2 * ffn, dim, device='meta'),
        'layers.1.proj.ffn_up_gate.weight': torch.empty(2 * ffn, dim, device='meta'),
    }
    for i in range(2):
        if fused_qkv:
            state[f'layers.{i}.proj.qkv.weight'] = torch.empty(3 * dim, dim, device='meta')
        else:
            for p in ('q', 'k', 'v'):
                state[f'layers.{i}.proj.{p}.weight'] = torch.empty(dim, dim, device='meta')
    return state


def test_moss_image_detection_and_text_encoder():
    config = model_detection.detect_unet_config(_moss_state(False), '')
    assert config == {'image_model': 'moss_image', 'dim': 256, 'ffn_dim': 512, 'num_heads': 2, 'num_layers': 2, 'fused_qkv': False}
    assert model_detection.detect_unet_config(_moss_state(True), '')['fused_qkv']
    model = supported_models.MOSSImage(config)
    assert model.clip_target().tokenizer is MOSSImageTokenizer
    assert model.sampling_settings['shift'] == 5.0
    base = model.get_model({}, device=torch.device('meta'))
    assert len(base.diffusion_model.layers) == 2
    assert 'moss_image' in CLIPLoader.INPUT_TYPES()['required']['type'][0]


def test_moss_image_lora_keys_address_fused_qkv():
    config = model_detection.detect_unet_config(_moss_state(True), '')
    base = supported_models.MOSSImage(config).get_model({}, device=torch.device('meta'))
    key_map = lora.model_lora_keys_unet(base, {})
    assert key_map['diffusion_model.layers.0.proj.k'] == ('diffusion_model.layers.0.proj.qkv.weight', (0, 256, 256))
    assert key_map['lora_unet_layers_1_proj_v'] == ('diffusion_model.layers.1.proj.qkv.weight', (0, 512, 256))


def test_moss_image_models_are_discoverable():
    unets = {str(model): model for model in KNOWN_UNET_MODELS.data}
    for name in ('moss_image_base_bf16.safetensors', 'moss_image_sft_bf16.safetensors',
                 'moss_image_base_int8_convrot_attn.safetensors', 'moss_image_sft_int8_convrot_attn.safetensors'):
        assert unets[name].repo_id == 'Comfy-Org/MOSS-Image'
        assert unets[name].revision == MOSS_REVISION
    clips = {str(model): model for model in KNOWN_CLIP_MODELS.data}
    assert clips['qwen3.5_9b_bf16.safetensors'].repo_id == 'Comfy-Org/Qwen3.5'
    for name in ('qwen3.5_9b_int8_convrot.safetensors', 'qwen3.5_9b_w4a8_gptq.safetensors'):
        assert clips[name].repo_id == 'Comfy-Org/MOSS-Image'
        assert clips[name].revision == MOSS_REVISION

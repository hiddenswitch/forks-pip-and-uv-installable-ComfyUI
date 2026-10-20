from __future__ import annotations

import importlib.resources

from comfy import supported_models
from . import workflows


# This is representative inference coverage for supported loader/model-family
# surfaces. Some supported classes are architecture variants that intentionally
# map to a shared workflow fixture; new classes still need an explicit entry.
SUPPORTED_MODEL_WORKFLOW_COVERAGE = {
    "LotusD": ("flux-controlnet-1.json",),
    "Stable_Zero123": ("flux-redux-0.json",),
    "SD15_instructpix2pix": ("flux-controlnet-1.json",),
    "SD15": ("default-0.json", "sd-1.5-lora-0.json"),
    "SD20": ("default-0.json",),
    "SD21UnclipL": ("flux-redux-0.json",),
    "SD21UnclipH": ("flux-redux-0.json",),
    "SDXL_instructpix2pix": ("sdxl-union-controlnet-0.json",),
    "SDXLRefiner": ("sdxl-union-controlnet-0.json",),
    "SDXL": ("sdxl-union-controlnet-0.json",),
    "SSD1B": ("sdxl-union-controlnet-0.json",),
    "KOALA_700M": ("sdxl-union-controlnet-0.json",),
    "KOALA_1B": ("sdxl-union-controlnet-0.json",),
    "Segmind_Vega": ("sdxl-union-controlnet-0.json",),
    "SD_X4Upscaler": ("image-upscale-with-model-0.json",),
    "Stable_Cascade_C": ("sdxl-union-controlnet-0.json",),
    "Stable_Cascade_B": ("sdxl-union-controlnet-0.json",),
    "SV3D_u": ("ltxv-1.json",),
    "SV3D_p": ("ltxv-1.json",),
    "SD3": ("sd3-default-0.json", "sd3-multiprompt-0.json", "sd3-single-t5-0.json"),
    "StableAudio": ("audio-0.json",),
    "StableAudio3": ("audio-0.json",),
    "AuraFlow": ("auraflow-0.json",),
    "PixArtAlpha": ("auraflow-0.json",),
    "PixArtSigma": ("auraflow-0.json",),
    "HunyuanDiT": ("hunyuandit-0.json",),
    "HunyuanDiT1": ("hunyuandit-0.json",),
    "FluxInpaint": ("flux-inpainting-0.json",),
    "Flux": ("flux-0.json", "flux-controlnet-0.json", "flux-redux-0.json"),
    "LongCatImage": ("qwen-image-edit-0.json",),
    "FluxSchnell": ("flux-0.json",),
    "GenmoMochi": ("mochi-text-to-video-0.json",),
    "LTXV": ("ltxv-0.json", "ltxv-1.json"),
    "LTXAV": ("ltx-2-0.json",),
    "MiniMaxH3": ("minimax-h3-fl2va-0.json", "minimax-h3-ref2va-0.json", "minimax-h3-union-controlnet-int8-convrot-0.json"),
    "MiniMaxMusic3": ("minimax-music3-0.json",),
    "HunyuanVideo15_SR_Distilled": ("hunyuan-video-0.json",),
    "HunyuanVideo15": ("hunyuan-video-0.json",),
    "HunyuanImage21Refiner": ("hunyuan_image-0.json",),
    "HunyuanImage21": ("hunyuan_image-0.json",),
    "HunyuanVideoSkyreelsI2V": ("hunyuan-video-0.json",),
    "HunyuanVideoI2V": ("hunyuan-video-0.json",),
    "HunyuanVideo": ("hunyuan-video-0.json",),
    "CosmosT2V": ("cosmos-0.json",),
    "CosmosI2V": ("cosmos-1.json",),
    "CosmosT2IPredict2": ("cosmos-2-0.json",),
    "CosmosI2VPredict2": ("cosmos-2-0.json",),
    "ZImagePixelSpace": ("z_image-0.json",),
    "ZImage": ("z_image-0.json",),
    "Lumina2": ("lumina2-0.json",),
    "Lens": ("phi-4-0.json",),
    "WAN22_T2V": ("ltxv-0.json",),
    "WAN21_CausalAR_T2V": ("ltxv-0.json",),
    "WAN21_T2V": ("ltxv-0.json",),
    "WAN21_I2V": ("ltxv-1.json",),
    "WAN21_FunControl2V": ("flux-controlnet-0.json",),
    "WAN21_Vace": ("flux-inpainting-0.json",),
    "WAN21_Camera": ("ltxv-1.json",),
    "WAN22_Camera": ("ltxv-1.json",),
    "WAN22_S2V": ("ltx-2-0.json",),
    "WAN21_HuMo": ("ltx-2-0.json",),
    "WAN22_Animate": ("ltxv-1.json",),
    "WAN_Animate2": ("ltxv-1.json",),
    "WAN21_FlowRVS": ("ltxv-1.json",),
    "WAN21_SCAIL": ("image-upscale-with-model-0.json",),
    "WAN21_SCAIL2": ("wan21-scail2-0.json",),
    "WAN22_WanDancer": ("ltxv-1.json",),
    "Hunyuan3Dv2mini": ("hunyuan_image-0.json",),
    "Hunyuan3Dv2": ("hunyuan_image-0.json",),
    "Hunyuan3Dv2_1": ("hunyuan_image-0.json",),
    "HiDream": ("hidream-0.json", "hidream-1.json"),
    "HiDreamO1": ("hidream-o1-0.json",),
    # SenseNova U1.5 shares the HiDream-O1 pixel-space latent and reference nodes; no natively
    # loadable checkpoint is published yet, so it maps to the shared fixture.
    "SenseNovaU15": ("hidream-o1-0.json",),
    "Chroma": ("chroma-0.json",),
    "ChromaRadiance": ("chroma-0.json",),
    "ACEStep": ("audio-0.json",),
    "ACEStep15": ("audio-0.json",),
    "Omnigen2": ("omnigen2-0.json",),
    "MingImage": ("ming-image-0.json",),
    "QwenImage21": ("qwen-image21-int8-convrot-0.json",),
    "YuE2": ("yue2-int8-convrot-0.json",),
    "QwenImage": ("qwen-image-0.json", "qwen-image-1.json", "qwen-image-2.json", "qwen-image-edit-0.json", "qwen-image-multigpu-cfg-0.json"),
    "JoyImage": ("joyimage-0.json",),
    "Ideogram4": ("ideogram4-0.json",),
    "Boogu": ("boogu-0.json",),
    "MageFlow": ("mage-flow-0.json",),
    "Krea2": ("krea2-0.json",),
    "MOSSImage": ("moss-image-0.json",),
    "Fibo2": ("fibo2-0.json",),
    "Flux3": ("flux3-0.json",),
    "Flux2": ("flux2-0.json", "flux2-klein-0.json", "flux2-template-text-to-image-0.json"),
    "Kandinsky5Image": ("qwen-image-edit-0.json",),
    "Kandinsky5": ("qwen-image-0.json",),
    "Anima": ("auraflow-0.json",),
    "RT_DETR_v4": ("sam3-segment-0.json",),
    "DepthAnything3": ("depth-anything3-0.json",),
    "ErnieImage": ("ernie-image-turbo-0.json",),
    "SAM3": ("sam3-segment-0.json",),
    "SAM31": ("sam3-segment-0.json",),
    "PixelDiTT2I": ("pixeldit-0.json",),
    "PiD": ("pid-0.json",),
    "TripoSplat": ("triposplat-0.json",),
    "CogVideoX_Inpaint": ("ltxv-1.json",),
    "CogVideoX_I2V": ("ltxv-1.json",),
    "CogVideoX_T2V": ("mochi-text-to-video-0.json",),
    "SVD_img2vid": ("ltxv-1.json",),
    "SeedVR2": ("seedvr2-0.json",),
}


def test_supported_models_have_inference_workflow_coverage():
    supported_model_names = {model.__name__ for model in supported_models.models}

    missing = supported_model_names - SUPPORTED_MODEL_WORKFLOW_COVERAGE.keys()
    extra = SUPPORTED_MODEL_WORKFLOW_COVERAGE.keys() - supported_model_names

    assert not missing, f"supported model classes without inference coverage: {sorted(missing)}"
    assert not extra, f"coverage entries for unknown supported model classes: {sorted(extra)}"

    workflow_files = {
        f.name
        for f in importlib.resources.files(workflows).iterdir()
        if f.is_file() and f.name.endswith(".json")
    }
    missing_workflows = {
        workflow_name
        for coverage in SUPPORTED_MODEL_WORKFLOW_COVERAGE.values()
        for workflow_name in coverage
        if workflow_name not in workflow_files
    }
    assert not missing_workflows, f"coverage references missing inference workflows: {sorted(missing_workflows)}"

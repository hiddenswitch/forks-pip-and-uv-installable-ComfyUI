from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import time
from importlib.resources import files
from pathlib import Path

import pytest

from comfy.app.custom_node_manager import CustomNodeManager
from comfy.cmd.workflow_templates import _collect_class_types
from comfy.component_model.prompt_utils import replace_steps, replace_width, replace_height
from comfy.component_model.workflow_convert import is_ui_workflow, convert_ui_to_api
from tests.inference_benchmark import prepare_benchmark_prompt

from comfy.component_model.node_registry import get_spec
from .conftest import (
    add_node_site_to_path,
    install_all_nodes,
    make_base_dirs,
    build_config,
)

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(os.environ.get(
    "COMFY_TEST_CACHE_DIR",
    Path.home() / ".cache" / "comfy-test" / "custom_nodes",
))

_STUB_IMAGE_URI = "pkg://tests.custom_nodes.test_data/president_official_portrait_hires2-1-1024x1024.jpg"
_STUB_AUDIO_URI = "pkg://tests.custom_nodes.test_data/test_audio.wav"
_STUB_VIDEO_URI = "pkg://tests.custom_nodes.test_data/test_video.mp4"
_STUB_POSE_VIDEO_URI = "pkg://tests.custom_nodes.test_data/test_pose_video.mp4"

_IMAGE_TO_URL: dict[str, str] = {
    "LoadImage": "LoadImageFromURL",
    "LoadImageMask": "LoadImageFromURL",
    "LoadImageOutput": "LoadImageFromURL",
}
_AUDIO_TO_URL: dict[str, str] = {
    "LoadAudio": "LoadAudioFromURL",
    "VHS_LoadAudio": "LoadAudioFromURL",
}
# VHS_LoadVideo/VHS_LoadVideoPath output IMAGE frames, not VIDEO —
# substituting with LoadVideoFromURL causes type mismatches.
# Instead, we patch their "video" input field in _patch_vhs_video_inputs.
_VIDEO_TO_URL: dict[str, str] = {}

_VHS_VIDEO_CLASS_TYPES = frozenset({"VHS_LoadVideo", "VHS_LoadVideoPath"})

_EXTRA_STEPS_CLASS_TYPES = frozenset({
    "WanVideoSampler",
    "WanVideoSamplerAdvanced",
    "SamplerCustomAdvanced",
    "SamplerCustom",
    "KSamplerSelect",
    "RES4LYF_Sampler",
})

_EXTRA_LATENT_CLASS_TYPES = frozenset({
    "WanVideoEmptyLatent",
    "EmptyMochiLatentVideo",
    "EmptyHunyuanLatentVideo",
    "EmptyLTXVLatentVideo",
    "EmptyCosmosLatentVideo",
})

_VIDEO_FRAME_CLASS_TYPES = frozenset({
    "WanVideoEmptyLatent",
    "EmptyMochiLatentVideo",
    "EmptyHunyuanLatentVideo",
    "EmptyLTXVLatentVideo",
    "EmptyCosmosLatentVideo",
    "WanVideoSampler",
})

_MODEL_MISSING_PATTERNS: tuple[str, ...] = tuple()

# Workflows to skip: models removed from HuggingFace or never published.
_XFAIL_WORKFLOWS: dict[str, str] = {
    "ComfyUI-KJNodes/leapfusion_hunyuuanvideo_i2v_native_testing":
        "LeapFusion Hunyuan video example loads large Hunyuan video models and exceeds local GPU test timeout on 24GB VRAM",
    "ComfyUI-WanVideoWrapper/LongCatAvatar_audio_image_to_video_example_01":
        "LongCatAvatar example exceeds local GPU test timeout on 24GB VRAM",
    "ComfyUI-WanVideoWrapper/LongCat_TI2V_example_01":
        "LongCat TI2V example exceeds local GPU test timeout on 24GB VRAM",
    "ComfyUI-WanVideoWrapper/wanvideo_1_3B_control_lora_example_01":
        "WanVideo control LoRA example exceeds local GPU test timeout on 24GB VRAM",
    "ComfyUI-WanVideoWrapper/wanvideo_2_1_14B_I2V_FantasyPortrait_example_01":
        "FantasyPortrait upstream node crashes on missing face landmarks with local stub media",
    "ComfyUI-WanVideoWrapper/wanvideo_2_1_14B_I2V_FantasyTalking_example_01":
        "WanVideo FantasyTalking example exceeds local GPU test timeout on 24GB VRAM",
    "ComfyUI-WanVideoWrapper/wanvideo_2_1_14B_I2V_InfiniteTalk_example_03":
        "WanVideo InfiniteTalk example exceeds local GPU test timeout inside WanVideo sampling on 24GB VRAM",
    "ComfyUI-WanVideoWrapper/wanvideo_2_1_14B_I2V_SkyReelsV3_TalkingAvatar_example_01":
        "Upstream WanVideoWrapper latent preview crashes when last_node_id is None in embedded execution",
    "ComfyUI-WanVideoWrapper/wanvideo_2_1_14B_Fun_control_camera_example_01":
        "1.3B Fun Camera model removed from HuggingFace (only 14B exists)",
    "ComfyUI-WanVideoWrapper/wanvideo_2_1_14B_Fun_control_example_01":
        "WanVideo Fun control example runs 177 frames at 25 steps and exceeds local GPU test timeout on 24GB VRAM",
    "ComfyUI-WanVideoWrapper/wanvideo_2_1_14B_HuMo_example_01":
        "WanVideo HuMo example exceeds local GPU test timeout on 24GB VRAM",
    "ComfyUI-WanVideoWrapper/wanvideo_2_1_14B_OneToAllAnimation_pose_control_example_01":
        "WanVideo OneToAllAnimation pose-control example hangs after GPU work on local 24GB test rig",
    "ComfyUI-WanVideoWrapper/wanvideo_2_1_14B_Stand-In_reference_example_01":
        "Upstream ControlNet Aux MediaPipe face mesh graph fails to parse in local custom-node environment",
    "ComfyUI-WanVideoWrapper/wanvideo_2_1_14B_skyreels_a2_example_01":
        "WanVideo SkyReels A2 example exceeds local GPU test timeout on 24GB VRAM",
    "ComfyUI-WanVideoWrapper/wanvideo_2_2_5B_Ovi_image_to_video_audio_10_seconds_example_01":
        "Ovi model_960x960_10s.safetensors removed from HuggingFace",
    "ComfyUI-WanVideoWrapper/wanvideo_2_1_14B_pusa_I2V_example_01":
        "14B Pusa model exceeds 24GB VRAM",
    "ComfyUI-segment-anything-2/image_batch_bbox_segment":
        "Upstream SAM2 batch bbox example crashes when Florence returns no boxes for local stub media",
    "ComfyUI_UltimateSDUpscale/basic-usdu":
        "Workflow references flat 4x-UltraSharp.pth upscale model that is not present in the local custom-node model cache",
    "RES4LYF/chroma txt2img":
        "Workflow references Chroma and ae.sft model filenames that are not present in the local custom-node model cache",
    "RES4LYF/comparison ksampler vs csksampler chain workflows":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/flux faceswap sync pulid":
        "Workflow references missing PulidFluxInsightFaceLoader node that is not installed by the current custom-node set",
    "RES4LYF/flux faceswap sync":
        "Upstream RES4LYF workflow crashes on empty mask coordinates with local stub media",
    "RES4LYF/flux faceswap":
        "Upstream RES4LYF workflow crashes on empty mask coordinates with local stub media",
    "RES4LYF/flux inpaint area":
        "Upstream RES4LYF workflow crashes on empty mask coordinates with local stub media",
    "RES4LYF/flux inpaint bongmath":
        "Upstream RES4LYF workflow crashes on empty mask coordinates with local stub media",
    "RES4LYF/flux inpainting":
        "Workflow references colossusProjectFlux_v42AIO.safetensors that is not present in the local custom-node model cache",
    "RES4LYF/flux style antiblur":
        "Workflow references colossusProjectFlux_v42AIO.safetensors that is not present in the local custom-node model cache",
    "RES4LYF/flux upscale thumbnail large multistage":
        "Workflow references unavailable Flux/controlnet model filenames and stale RES4LYF option values",
    "RES4LYF/flux upscale thumbnail large":
        "Workflow references unavailable Flux/controlnet model filenames and stale RES4LYF option values",
    "RES4LYF/flux upscale thumbnail widescreen":
        "Workflow references unavailable Flux/controlnet model filenames and stale RES4LYF option values",
    "RES4LYF/hidream guide data projection":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream guide epsilon projection":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream guide flow":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream guide fully_pseudoimplicit":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream guide lure":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream guide pseudoimplicit":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream hires fix":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream regional 3 zones":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream style antiblur":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream style transfer":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream txt2img":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream unsampling data WF":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream unsampling data":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream unsampling pseudoimplicit":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/hidream unsampling":
        "Workflow references ae.sft VAE filename that is not present in the local custom-node model cache",
    "RES4LYF/intro to clownsampling":
        "Workflow references missing UltraCascade_Loader node that is not installed by the current custom-node set",
    "RES4LYF/sd35 medium unsampling data":
        "Workflow references SD3.5 VAE/CLIP filenames that are not present in the local custom-node model cache",
    "RES4LYF/sd35 medium unsampling":
        "Workflow references SD3.5 VAE/CLIP filenames that are not present in the local custom-node model cache",
    "RES4LYF/sdxl regional antiblur":
        "Workflow references _SDXL_/juggernautXL_v9Rundiffusionphoto2.safetensors that is not present in the local custom-node model cache",
    "RES4LYF/style transfer":
        "Workflow references SD3.5 VAE/CLIP filenames that are not present in the local custom-node model cache",
    "RES4LYF/ultracascade txt2img style transfer":
        "Workflow references missing UltraCascade_Loader node that is not installed by the current custom-node set",
    "RES4LYF/ultracascade txt2img":
        "Workflow references missing UltraCascade_Loader node that is not installed by the current custom-node set",
    "RES4LYF/wan vid2vid":
        "Workflow references a stale RES4LYF unsampler_override option value that is no longer accepted",
    "audio-separation-nodes-comfyui/Remix Song":
        "Workflow fails locally while PyAV muxes preview audio after remix generation",
    "ComfyUI_AudioTools/AudioTools_example":
        "Workflow references Fast Groups Muter (rgthree), which is not installed by the current custom-node set",
    "ComfyUI_IPAdapter_plus/IPAdapter_FaceIDv2_Kolors":
        "Workflow references Kolors/IPAdapter/ChatGLM model filenames that are not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_advanced":
        "Workflow references sd15/realisticVisionV51_v51VAE.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_clipvision_enhancer":
        "Workflow references sdxl/RealVisXL_V4.0.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_combine_embeds":
        "Workflow references sd15/realisticVisionV51_v51VAE.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_faceid":
        "Workflow references sd15/realisticVisionV51_v51VAE.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_ideal_faceid_config":
        "Workflow references sd15/realisticVisionV51_v51VAE.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_kolors":
        "Workflow references Kolors/IPAdapter/ChatGLM model filenames that are not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_negative_image":
        "Workflow references sd15/realisticVisionV51_v51VAE.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_noise_injection":
        "Workflow references sd15/realisticVisionV51_v51VAE.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_portrait":
        "Workflow references sdxl/juggernautXL_version8Rundiffusion.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_precise_composition":
        "Workflow references sdxl/ProteusV0.3.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_precise_weight_type":
        "Workflow references sdxl/ProteusV0.3.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_regional_conditioning":
        "Workflow references sd15/juggernaut_reborn.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_simple":
        "Workflow references sd15/realisticVisionV51_v51VAE.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_style_composition":
        "Workflow references sdxl/AlbedoBaseXL.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_tiled":
        "Workflow references sd15/realisticVisionV51_v51VAE.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_weight_types":
        "Workflow references sd15/realisticVisionV51_v51VAE.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_weighted_embeds":
        "Workflow references sd15/realisticVisionV51_v51VAE.safetensors that is not present in the local custom-node model cache",
    "ComfyUI_IPAdapter_plus/ipadapter_weights":
        "Workflow references sd15/realisticVisionV51_v51VAE.safetensors that is not present in the local custom-node model cache",
    "ComfyUI-Flux-Continuum/Flux+ 1.3_release":
        "Workflow references JWIntegerMul, which is not installed by the current custom-node set",
    "ComfyUI-Flux-Continuum/Flux+ 1.4.4_release":
        "Workflow references missing OutputGet/OutputGetString nodes that are not installed by the current custom-node set",
    "ComfyUI-Flux-Continuum/Flux+ 1.4.5_release":
        "Workflow references missing OutputGet/OutputGetString nodes that are not installed by the current custom-node set",
    "ComfyUI-Flux-Continuum/Flux+ 1.6.4_release":
        "Workflow references missing OutputGet/OutputGetString nodes that are not installed by the current custom-node set",
    "ComfyUI-Flux-Continuum/Flux+ 1.7.0_release":
        "Workflow references missing OutputGet/OutputGetString nodes that are not installed by the current custom-node set",
    "ComfyUI-Flux-Continuum/Flux+ 1.7.1_beta":
        "Workflow references missing OutputGet/OutputGetString nodes that are not installed by the current custom-node set",
    "ComfyUI-Flux-Continuum/Flux+ Light 1.0.0_release":
        "Workflow references missing OutputGet node that is not installed by the current custom-node set",
    "ComfyUI_LayerStyle/auto_adjust_v2_example":
        "Workflow references LayerStyle RMBG model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/auto_brightness_example":
        "Workflow references LayerStyle BiRefNet model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/blend_mode_v2_example":
        "Workflow uses stale LayerStyle ColorPicker widget values that no longer validate",
    "ComfyUI_LayerStyle/crop_by_mask_&_restore_crop_box_example":
        "Workflow references LayerStyle RMBG model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/distort_displace_example":
        "Workflow references missing LayerMask: SegmentAnythingUltra V2 node that is not installed by the current custom-node set",
    "ComfyUI_LayerStyle/extend_canvas_example":
        "Workflow uses a stale LayerStyle ExtendCanvas schema missing the required color input",
    "ComfyUI_LayerStyle/flux_kontext_image_scale_example":
        "Workflow references LayerStyle RMBG model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/hl_frequency_detail_restore_example":
        "Workflow references LayerStyle RMBG model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/image_to_mask_example":
        "Workflow references LayerStyle RMBG model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/icmask_example":
        "Workflow references missing LayerMask: SegmentAnythingUltra V2 node that is not installed by the current custom-node set",
    "ComfyUI_LayerStyle/image_mask_scale_as_example":
        "Workflow references LayerStyle RMBG model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/image_remove_alpha & image_combine_alpha_example":
        "Workflow references LayerStyle RMBG model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/image_tagger_save_example":
        "Workflow references missing LayerMask: LoadFlorence2Model node that is not installed by the current custom-node set",
    "ComfyUI_LayerStyle/layer_image_transform_example":
        "Workflow references missing LayerMask: SegmentAnythingUltra V2 node that is not installed by the current custom-node set",
    "ComfyUI_LayerStyle/layerstyle_all_nodes":
        "Workflow references LayerStyle RMBG/BiRefNet model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/mask_by_color_example":
        "Workflow references LayerStyle RMBG model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/mask_edge_ultra_detail_example":
        "Workflow references missing Image Remove Background Rembg (mtb) node that is not installed by the current custom-node set",
    "ComfyUI_LayerStyle/mask_edge_ultra_detail_v3_example":
        "Workflow references LayerStyle BiRefNet model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/mask_edge_ultra_detail_v2_example":
        "Workflow references missing LayerMask: SegmentAnythingUltra V2 node that is not installed by the current custom-node set",
    "ComfyUI_LayerStyle/pixel_spread_example":
        "Workflow references LayerStyle RMBG model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/queue_stop_example":
        "Workflow uses a stale LayerStyle QueueStop widget value that no longer validates",
    "ComfyUI_LayerStyle/rembg_ultra_example":
        "Workflow references LayerStyle RMBG model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/rounded_rectangle_example":
        "Workflow references LayerStyle BiRefNet model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/segformet_clothes_example":
        "Workflow references upstream mattmdjaga/segformer_b3_clothes model repo that is not available from HuggingFace",
    "ComfyUI_LayerStyle/segformet_fashion_example":
        "Workflow references upstream mattmdjaga/segformer_b3_fashion model repo that is not available from HuggingFace",
    "ComfyUI_LayerStyle/simple_text_example":
        "Workflow references Alibaba-PuHuiTi-Bold.ttf, which is not present in the local custom-node font set",
    "ComfyUI_LayerStyle/text_image_example":
        "Workflow references a non-installed LayerStyle TextImage font file",
    "ComfyUI_LayerStyle/title_example_workflow":
        "Workflow references LayerStyle RMBG model files that are not present in the local custom-node model cache",
    "ComfyUI_LayerStyle/ultra_v2_nodes_example":
        "Workflow references missing LayerMask: SegmentAnythingUltra V2 node that is not installed by the current custom-node set",
    "ComfyUI_Fill-ChatterBox/Chatterbox":
        "Workflow requires torchcodec for torchaudio.save in the local custom-node environment",
}




_STUB_VIDEO_FRAMES = 180  # must match test_video.mp4
_STUB_POSE_VIDEO_FRAMES = 300  # must match test_pose_video.mp4

# Workflows that need the longer pose video (full-body human with 300 frames).
_POSE_VIDEO_WORKFLOWS: frozenset[str] = frozenset({
    "OneToAllAnimation",
    "MTV_Crafter",
})


def _select_video_for_workflow(workflow_name: str) -> tuple[str, int]:
    """Return (video_uri, frame_count) appropriate for the workflow."""
    for keyword in _POSE_VIDEO_WORKFLOWS:
        if keyword in workflow_name:
            return _STUB_POSE_VIDEO_URI, _STUB_POSE_VIDEO_FRAMES
    return _STUB_VIDEO_URI, _STUB_VIDEO_FRAMES


def _video_filename_for_uri(video_uri: str) -> str:
    if video_uri == _STUB_POSE_VIDEO_URI:
        return "test_pose_video.mp4"
    return "test_video.mp4"


def _patch_vhs_video_inputs(workflow: dict, video_uri: str, video_frames: int) -> dict:
    """Replace hardcoded video filenames in VHS_LoadVideo nodes with test video.

    Also disconnects any links overriding ``frame_load_cap`` (so the widget
    value is used) and caps ``frame_load_cap=0`` (unlimited) to the test
    video's actual frame count, avoiding temporal-dimension mismatches.
    """
    if is_ui_workflow(workflow):
        nodes = workflow.get("nodes")
        if not isinstance(nodes, list):
            return workflow
        need_patch = any(
            isinstance(n, dict) and n.get("type", "") in _VHS_VIDEO_CLASS_TYPES
            for n in nodes
        )
        if not need_patch:
            return workflow
        workflow = copy.deepcopy(workflow)
        # Collect link IDs to remove (links overriding frame_load_cap)
        remove_links: set[int] = set()
        for node in workflow["nodes"]:
            if not isinstance(node, dict) or node.get("type", "") not in _VHS_VIDEO_CLASS_TYPES:
                continue
            wv = node.get("widgets_values")
            if isinstance(wv, list) and wv:
                wv[0] = video_uri
            elif isinstance(wv, dict) and "video" in wv:
                wv["video"] = video_uri
                # Cap unlimited frame_load_cap to test video length
                if wv.get("frame_load_cap", 0) == 0:
                    wv["frame_load_cap"] = video_frames
            # Remove links feeding into frame_load_cap
            for inp in node.get("inputs", []):
                if inp.get("name") == "frame_load_cap" and inp.get("link") is not None:
                    remove_links.add(inp["link"])
                    inp["link"] = None
        if remove_links:
            links = workflow.get("links", [])
            workflow["links"] = [lk for lk in links if lk[0] not in remove_links]
    else:
        need_patch = any(
            isinstance(n, dict) and n.get("class_type", "") in _VHS_VIDEO_CLASS_TYPES
            for n in workflow.values()
        )
        if not need_patch:
            return workflow
        workflow = copy.deepcopy(workflow)
        for node in workflow.values():
            if isinstance(node, dict) and node.get("class_type", "") in _VHS_VIDEO_CLASS_TYPES:
                inputs = node.get("inputs", {})
                if "video" in inputs:
                    inputs["video"] = video_uri
                if inputs.get("frame_load_cap", 0) == 0:
                    inputs["frame_load_cap"] = video_frames
                # Remove linked frame_load_cap (it would be [node_id, slot])
                if isinstance(inputs.get("frame_load_cap"), list):
                    del inputs["frame_load_cap"]
    return workflow


def _patch_core_load_video_inputs(workflow: dict, video_filename: str) -> dict:
    if is_ui_workflow(workflow):
        nodes = workflow.get("nodes")
        if not isinstance(nodes, list):
            return workflow
        if not any(isinstance(n, dict) and n.get("type") == "LoadVideo" for n in nodes):
            return workflow
        workflow = copy.deepcopy(workflow)
        for node in workflow["nodes"]:
            if not isinstance(node, dict) or node.get("type") != "LoadVideo":
                continue
            wv = node.get("widgets_values")
            if isinstance(wv, list) and wv:
                wv[0] = video_filename
            elif isinstance(wv, dict):
                wv["file"] = video_filename
        return workflow

    if not any(
        isinstance(n, dict) and n.get("class_type") == "LoadVideo"
        for n in workflow.values()
    ):
        return workflow
    workflow = copy.deepcopy(workflow)
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type") == "LoadVideo":
            node.setdefault("inputs", {})["file"] = video_filename
    return workflow


def _substitute_media_nodes(workflow: dict, workflow_name: str = "") -> dict:
    _ALL_MEDIA: dict[str, tuple[str, str]] = {}
    for src, dst in _IMAGE_TO_URL.items():
        _ALL_MEDIA[src] = (dst, _STUB_IMAGE_URI)
    for src, dst in _AUDIO_TO_URL.items():
        _ALL_MEDIA[src] = (dst, _STUB_AUDIO_URI)
    for src, dst in _VIDEO_TO_URL.items():
        _ALL_MEDIA[src] = (dst, _STUB_VIDEO_URI)

    if is_ui_workflow(workflow):
        workflow = _substitute_media_nodes_ui(workflow, _ALL_MEDIA)
    else:
        workflow = _substitute_media_nodes_api(workflow, _ALL_MEDIA)
    video_uri, video_frames = _select_video_for_workflow(workflow_name)
    workflow = _patch_vhs_video_inputs(workflow, video_uri, video_frames)
    video_filename = _video_filename_for_uri(video_uri)
    return _patch_core_load_video_inputs(workflow, video_filename)


def _substitute_media_nodes_api(
    workflow: dict,
    media_map: dict[str, tuple[str, str]],
) -> dict:
    node_ids = [
        nid for nid, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type", "") in media_map
    ]
    if not node_ids:
        return workflow
    workflow = copy.deepcopy(workflow)
    for nid in node_ids:
        node = workflow[nid]
        url_class, stub_uri = media_map[node["class_type"]]
        node["class_type"] = url_class
        node["inputs"] = {"value": stub_uri}
        node.pop("_meta", None)
    return workflow


def _substitute_media_nodes_ui(
    workflow: dict,
    media_map: dict[str, tuple[str, str]],
) -> dict:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return workflow

    patched_ids = [
        node.get("id")
        for node in nodes
        if isinstance(node, dict) and node.get("type", "") in media_map
    ]
    if not patched_ids:
        return workflow

    workflow = copy.deepcopy(workflow)
    for node in workflow["nodes"]:
        if not isinstance(node, dict) or node.get("id") not in patched_ids:
            continue
        node_type = node.get("type", "")
        url_class, stub_uri = media_map[node_type]
        node["type"] = url_class
        node["widgets_values"] = [stub_uri]
    return workflow


def _apply_cost_reduction_api(workflow: dict) -> dict:
    workflow = replace_steps(workflow, 2)
    workflow = replace_width(workflow, 256)
    workflow = replace_height(workflow, 256)

    modified = False
    for nid, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        if class_type in _EXTRA_STEPS_CLASS_TYPES and "steps" in inputs:
            if not modified:
                workflow = copy.deepcopy(workflow)
                modified = True
            workflow[nid]["inputs"]["steps"] = 2

        if class_type in _EXTRA_LATENT_CLASS_TYPES:
            if not modified:
                workflow = copy.deepcopy(workflow)
                modified = True
            if "width" in inputs:
                workflow[nid]["inputs"]["width"] = 256
            if "height" in inputs:
                workflow[nid]["inputs"]["height"] = 256

        if class_type in _VIDEO_FRAME_CLASS_TYPES:
            if not modified:
                workflow = copy.deepcopy(workflow)
                modified = True
            for field in ("num_frames", "length", "video_frames", "batch_size"):
                if field in inputs and isinstance(inputs[field], (int, float)):
                    workflow[nid]["inputs"][field] = min(int(inputs[field]), 2)

    return workflow


def _apply_cost_reduction_ui(workflow: dict) -> dict:
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return workflow

    modified = False
    for node in nodes:
        if not isinstance(node, dict):
            continue
        widgets = node.get("widgets_values")
        if not isinstance(widgets, (list, dict)):
            continue
        node_type = node.get("type", "")

        if isinstance(widgets, dict):
            for key in ("steps", "width", "height", "num_frames", "length"):
                if key in widgets and isinstance(widgets[key], (int, float)):
                    if not modified:
                        workflow = copy.deepcopy(workflow)
                        modified = True
                        nodes = workflow["nodes"]
                    new_val = 2 if key in ("steps", "num_frames", "length") else 256
                    for n in nodes:
                        if n.get("id") == node.get("id"):
                            if isinstance(n.get("widgets_values"), dict):
                                n["widgets_values"][key] = new_val
                            break
            continue

        all_steps_types = {"KSampler", "KSamplerAdvanced", "BasicScheduler",
                          "Flux2Scheduler", "LTXVScheduler"} | _EXTRA_STEPS_CLASS_TYPES
        all_latent_types = {"EmptyLatentImage", "EmptySD3LatentImage"} | _EXTRA_LATENT_CLASS_TYPES

        if node_type in all_steps_types or node_type in all_latent_types or node_type in _VIDEO_FRAME_CLASS_TYPES:
            for i, val in enumerate(widgets):
                if isinstance(val, int) and val > 256:
                    if not modified:
                        workflow = copy.deepcopy(workflow)
                        modified = True
                        nodes = workflow["nodes"]
                    for n in nodes:
                        if n.get("id") == node.get("id"):
                            wv = n.get("widgets_values")
                            if isinstance(wv, list) and i < len(wv):
                                wv[i] = min(val, 256)
                            break

    return workflow


def _apply_cost_reduction(workflow: dict) -> dict:
    if is_ui_workflow(workflow):
        return _apply_cost_reduction_ui(workflow)
    return _apply_cost_reduction_api(workflow)


def _is_model_missing_error(error_msg: str) -> bool:
    return any(pattern.lower() in error_msg.lower() for pattern in _MODEL_MISSING_PATTERNS)



# Node types to bypass (mode=4) in UI workflows before conversion.
_BYPASS_NODE_TYPES: frozenset[str] = frozenset({
    "Bookmark (rgthree)",
    "ImageDisplay",
    "WanVideoTorchCompileSettings",
})


def _bypass_nodes(workflow: dict) -> dict:
    """Set mode=4 (bypass) on specific node types in a UI workflow."""
    if not is_ui_workflow(workflow):
        return workflow
    nodes = workflow.get("nodes")
    if not isinstance(nodes, list):
        return workflow
    node_by_id = {
        node.get("id"): node
        for node in nodes
        if isinstance(node, dict) and node.get("id") is not None
    }
    links_by_id = {
        link[0]: link
        for link in workflow.get("links", [])
        if isinstance(link, list) and len(link) >= 4
    }

    def _preview_is_fed_by_muted_node(node: dict) -> bool:
        if node.get("type") != "PreviewImage":
            return False
        for inp in node.get("inputs", []):
            link_id = inp.get("link")
            link = links_by_id.get(link_id)
            if link is None:
                continue
            src = node_by_id.get(link[1])
            if isinstance(src, dict) and src.get("mode") == 2:
                return True
        return False

    need_patch = any(
        isinstance(n, dict)
        and (n.get("type", "") in _BYPASS_NODE_TYPES or _preview_is_fed_by_muted_node(n))
        for n in nodes
    )
    if not need_patch:
        return workflow
    workflow = copy.deepcopy(workflow)
    for node in workflow["nodes"]:
        if not isinstance(node, dict):
            continue
        if node.get("type", "") in _BYPASS_NODE_TYPES or _preview_is_fed_by_muted_node(node):
            node["mode"] = 4
    return workflow



def _collect_workflow_entries(base_dir):
    custom_nodes_root = str(base_dir / "custom_nodes")
    return CustomNodeManager.scan_example_workflows([custom_nodes_root])


def _get_shared_base_dir() -> Path:
    base_dir = _CACHE_DIR
    marker = base_dir / ".installed"

    if marker.exists():
        logger.info("Using cached custom node installation at %s", base_dir)
    else:
        logger.info("Installing all custom nodes into %s (first run)", base_dir)
        make_base_dirs(base_dir)
        installed = install_all_nodes(base_dir)
        logger.info("Installed %d custom nodes", len(installed))
        marker.write_text(json.dumps({
            "count": len(installed),
            "nodes": sorted(installed.keys()),
        }))

    add_node_site_to_path(base_dir)
    input_dir = base_dir / "input"
    data_dir = files("tests.custom_nodes.test_data")
    for filename in ("test_video.mp4", "test_pose_video.mp4", "test_audio.wav", "president_official_portrait_hires2-1-1024x1024.jpg"):
        destination = input_dir / filename
        if not destination.exists():
            shutil.copyfile(data_dir / filename, destination)
    return base_dir


# ---------------------------------------------------------------------------
# Collect (node_id, workflow_name, filepath) at import time for parametrize.
# Only scans the filesystem — no heavy imports or node loading.
# ---------------------------------------------------------------------------
def _collect_all_workflow_params() -> list[tuple[str, str, str]]:
    """Scan the cache dir for example workflow JSON files.

    Returns (node_id, workflow_name, filepath) triples.
    Populate the cache before collecting this suite.
    """
    custom_nodes_root = _CACHE_DIR / "custom_nodes"
    if not custom_nodes_root.is_dir():
        return []
    results = []
    for folder_name in CustomNodeManager.EXAMPLE_WORKFLOW_FOLDER_NAMES:
        for filepath in sorted(custom_nodes_root.glob(f"*/{folder_name}/*.json")):
            node_id = filepath.parent.parent.name
            workflow_name = filepath.stem
            results.append((node_id, workflow_name, str(filepath)))
    return results


_ALL_WORKFLOW_PARAMS = _collect_all_workflow_params()


def test_workflow_examples_are_available():
    assert _ALL_WORKFLOW_PARAMS, (
        f"No custom-node workflows found in {_CACHE_DIR / 'custom_nodes'}. "
        "Populate COMFY_TEST_CACHE_DIR before collecting this suite; "
        "an empty parameter list does not exercise any workflows."
    )

# Build the pytest parameter list: id string is "node_id/workflow_name"
_PARAM_IDS = [f"{node_id}/{wf}" for node_id, wf, _ in _ALL_WORKFLOW_PARAMS]

_shared_base_dir: Path | None = None


@pytest.fixture(scope="session")
def shared_base_dir():
    global _shared_base_dir
    if _shared_base_dir is None:
        _shared_base_dir = _get_shared_base_dir()
    return _shared_base_dir


class TestCustomNodeExecution:

    @pytest.mark.asyncio
    @pytest.mark.timeout(1200)
    @pytest.mark.parametrize(
        "node_id,workflow_name,workflow_path",
        _ALL_WORKFLOW_PARAMS,
        ids=_PARAM_IDS,
    )
    async def test_execute_workflow(self, node_id, workflow_name, workflow_path, shared_base_dir, inference_benchmark):
        from comfy.client.embedded_comfy_client import Comfy

        test_key = f"{node_id}/{workflow_name}"
        if test_key in _XFAIL_WORKFLOWS:
            pytest.xfail(_XFAIL_WORKFLOWS[test_key])

        spec = get_spec(node_id)
        if spec is not None and spec.xfail:
            pytest.xfail(spec.xfail_reason)

        base_dir = shared_base_dir

        with open(workflow_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pytest.skip(f"{node_id}/{workflow_name}: invalid JSON")

        if not isinstance(data, dict):
            pytest.skip(f"{node_id}/{workflow_name}: top-level is not a dict")

        data = _apply_cost_reduction(data)
        data = _substitute_media_nodes(data, workflow_name)
        data = _bypass_nodes(data)

        class_types = _collect_class_types(data)
        logger.info(
            "%s/%s: %d class_types: %s",
            node_id, workflow_name, len(class_types),
            sorted(class_types)[:10],
        )

        import comfy.cmd.main_pre  # noqa: F401
        real_base = str(Path(str(files("tests"))).parent.parent)
        config = build_config(base_dir, base_paths=[real_base])

        async with Comfy(configuration=config) as client:
            if is_ui_workflow(data):
                data = convert_ui_to_api(data)
            data = prepare_benchmark_prompt(_apply_cost_reduction_api(data))
            start = time.monotonic()
            outputs = await client.queue_prompt(data)
            elapsed = time.monotonic() - start
            logger.info(
                "%s/%s: executed in %.1fs, outputs: %s",
                node_id, workflow_name, elapsed, outputs,
            )

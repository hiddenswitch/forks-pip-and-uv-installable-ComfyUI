from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CustomNodeSpec:
    node_id: str
    repo_url: str
    display_name: str
    priority: str = "High"
    git_ref: Optional[str] = None
    needs_submodules: bool = False
    expected_node_types: list[str] = field(default_factory=list)
    extra_requirements: list[str] = field(default_factory=list)
    skip_requirements: frozenset[str] = field(default_factory=frozenset)
    depends_on: tuple[str, ...] = ()
    xfail: bool = False
    xfail_reason: str = ""
    inject_version: Optional[str] = None


DEFAULT_SKIP: frozenset[str] = frozenset({
    "torch", "torchvision", "torchaudio", "torchsde",
    "comfy", "comfyui",
})

CUSTOM_NODE_REGISTRY: list[CustomNodeSpec] = [
    CustomNodeSpec(
        node_id="kandinsky6",
        repo_url="https://github.com/kandinskylab/kandinsky-6",
        display_name="Kandinsky 6",
        expected_node_types=[
            "Kandinsky6BeautifyPrompt", "Kandinsky6Sampler", "Kandinsky6EmptyLatent",
            "Kandinsky6ImageToVideoAudio", "Kandinsky6RemoveReferenceLatent",
            "Kandinsky6AudioVAELoader", "Kandinsky6AudioVAEDecode",
            "Kandinsky6TextEncode", "Kandinsky6MagCache",
        ],
    ),
    CustomNodeSpec(
        node_id="kandinsky6-sr",
        repo_url="https://github.com/kandinskylab/kandinsky-6-sr",
        display_name="Kandinsky 6 SR",
        expected_node_types=["Kandinsky6SRVAELoader", "Kandinsky6LatentUpscalerLoader", "Kandinsky6VSRUpscale"],
    ),
    CustomNodeSpec(
        node_id="comfyui-hunyuanimage3",
        repo_url="https://github.com/PedroMarinhoDev/ComfyUI-HunyuanImage3",
        display_name="ComfyUI-HunyuanImage3",
        git_ref="84ad3a3e2a54472e69e195269729f774b242d120",
        inject_version="0.1.0",
        expected_node_types=[
            "HunyuanImage3ModelLoader", "HunyuanImage3TextEncode", "HunyuanImage3ImageEncode",
            "HunyuanImage3VAELoader", "HunyuanImage3PromptRewriting", "HunyuanImage3Resolutions",
            "HunyuanImage3Guidance", "HunyuanImage3EmptyLatent", "HunyuanImage3Spectrum",
        ],
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Prompt-Combinator",
        repo_url="https://github.com/lquesada/ComfyUI-Prompt-Combinator",
        display_name="ComfyUI-Prompt-Combinator",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-qwenmultiangle",
        repo_url="https://github.com/jtydhr88/ComfyUI-qwenmultiangle",
        display_name="ComfyUI-qwenmultiangle",
        xfail=True,
        xfail_reason="transformers auto-download; requires large Qwen model",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-SCAIL-Pose",
        repo_url="https://github.com/kijai/ComfyUI-SCAIL-Pose",
        display_name="ComfyUI-SCAIL-Pose",
        depends_on=("ComfyUI-WanVideoWrapper", "ComfyUI-KJNodes", "ComfyUI-VideoHelperSuite"),
    ),
    CustomNodeSpec(
        node_id="ComfyUI-SeedVR2_VideoUpscaler",
        repo_url="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler",
        display_name="ComfyUI-SeedVR2_VideoUpscaler",
        depends_on=("ComfyUI-VideoHelperSuite",),
    ),
    CustomNodeSpec(
        node_id="audio-separation-nodes-comfyui",
        repo_url="https://github.com/christian-byrne/audio-separation-nodes-comfyui",
        display_name="audio-separation-nodes-comfyui",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Inspire-Pack",
        repo_url="https://github.com/ltdrdata/ComfyUI-Inspire-Pack",
        display_name="ComfyUI-Inspire-Pack",
        extra_requirements=[
            "webcolors",
        ],
        depends_on=("ComfyUI-Impact-Pack",),
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Impact-Subpack",
        repo_url="https://github.com/ltdrdata/ComfyUI-Impact-Subpack",
        display_name="ComfyUI-Impact-Subpack",
        depends_on=("ComfyUI-Impact-Pack",),
    ),
    CustomNodeSpec(
        node_id="ComfyUI_IPAdapter_plus",
        repo_url="https://github.com/cubiq/ComfyUI_IPAdapter_plus",
        display_name="ComfyUI_IPAdapter_plus",
        depends_on=("ComfyUI_essentials",),
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Impact-Pack",
        repo_url="https://github.com/ltdrdata/ComfyUI-Impact-Pack",
        display_name="ComfyUI-Impact-Pack",
        depends_on=("ComfyUI-Impact-Subpack", "ComfyUI-Inspire-Pack"),
    ),
    CustomNodeSpec(
        node_id="ComfyUI_LayerStyle",
        repo_url="https://github.com/chflame163/ComfyUI_LayerStyle",
        display_name="ComfyUI_LayerStyle",
        depends_on=("ComfyUI-Impact-Pack", "ComfyUI-segment-anything-2"),
    ),
    CustomNodeSpec(
        node_id="ComfyUI-nunchaku",
        repo_url="https://github.com/nunchaku-tech/ComfyUI-nunchaku",
        display_name="ComfyUI-nunchaku",
        extra_requirements=[
            'nunchaku; sys_platform != "darwin"',
        ],
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Advanced-ControlNet",
        repo_url="https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet",
        display_name="ComfyUI-Advanced-ControlNet",
        depends_on=("comfyui_controlnet_aux",),
    ),
    CustomNodeSpec(
        node_id="ComfyUI_AudioTools",
        repo_url="https://github.com/lum3on/ComfyUI_AudioTools",
        display_name="ComfyUI_AudioTools",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-AnimateDiff-Evolved",
        repo_url="https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved",
        display_name="ComfyUI-AnimateDiff-Evolved",
        depends_on=("ComfyUI-Advanced-ControlNet",),
    ),
    CustomNodeSpec(
        node_id="ComfyUI-MelBandRoFormer",
        repo_url="https://github.com/kijai/ComfyUI-MelBandRoFormer",
        display_name="ComfyUI-MelBandRoFormer",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-segment-anything-2",
        repo_url="https://github.com/kijai/ComfyUI-segment-anything-2",
        display_name="ComfyUI-segment-anything-2",
    ),
    CustomNodeSpec(
        node_id="ComfyMath",
        repo_url="https://github.com/evanspearman/ComfyMath",
        display_name="ComfyMath",
        inject_version="0.1.0",
    ),
    CustomNodeSpec(
        node_id="ComfyUI_essentials",
        repo_url="https://github.com/cubiq/ComfyUI_essentials",
        display_name="ComfyUI_essentials",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Florence2",
        repo_url="https://github.com/kijai/ComfyUI-Florence2",
        display_name="ComfyUI-Florence2",
    ),
    CustomNodeSpec(
        node_id="rgthree-comfy",
        repo_url="https://github.com/rgthree/rgthree-comfy",
        display_name="rgthree-comfy",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Flux-Continuum",
        repo_url="https://github.com/robertvoy/ComfyUI-Flux-Continuum",
        display_name="ComfyUI-Flux-Continuum",
        depends_on=("rgthree-comfy", "ComfyUI_essentials"),
    ),
    CustomNodeSpec(
        node_id="ComfyUI-ConditioningKrea2Rebalance",
        repo_url="https://github.com/nova452/ComfyUI-ConditioningKrea2Rebalance",
        display_name="ComfyUI-ConditioningKrea2Rebalance",
        inject_version="0.1.0",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-WanVideoWrapper",
        repo_url="https://github.com/kijai/ComfyUI-WanVideoWrapper",
        display_name="ComfyUI-WanVideoWrapper",
        depends_on=("ComfyUI-KJNodes", "ComfyUI-VideoHelperSuite"),
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Frame-Interpolation",
        repo_url="https://github.com/Fannovel16/ComfyUI-Frame-Interpolation",
        display_name="ComfyUI-Frame-Interpolation",
        depends_on=("ComfyUI-VideoHelperSuite",),
    ),
    CustomNodeSpec(
        node_id="RES4LYF",
        repo_url="https://github.com/ClownsharkBatwing/RES4LYF",
        display_name="RES4LYF",
        inject_version="0.1.0",
    ),
    CustomNodeSpec(
        node_id="Comfyui-Resolution-Master",
        repo_url="https://github.com/Azornes/Comfyui-Resolution-Master",
        display_name="Comfyui-Resolution-Master",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-WanAnimatePreprocess",
        repo_url="https://github.com/kijai/ComfyUI-WanAnimatePreprocess",
        display_name="ComfyUI-WanAnimatePreprocess",
        depends_on=("ComfyUI-WanVideoWrapper", "ComfyUI-KJNodes", "ComfyUI-VideoHelperSuite"),
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Crystools",
        repo_url="https://github.com/crystian/ComfyUI-Crystools",
        display_name="ComfyUI-Crystools",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Detail-Daemon",
        repo_url="https://github.com/Jonseed/ComfyUI-Detail-Daemon",
        display_name="ComfyUI-Detail-Daemon",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-GGUF",
        repo_url="https://github.com/city96/ComfyUI-GGUF",
        display_name="ComfyUI-GGUF",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-INT8-Fast",
        repo_url="https://github.com/BobJohnson24/ComfyUI-INT8-Fast",
        display_name="ComfyUI-INT8-Fast",
        inject_version="0.1.0",
    ),
    CustomNodeSpec(
        node_id="ComfyUI_UltimateSDUpscale",
        repo_url="https://github.com/ssitu/ComfyUI_UltimateSDUpscale",
        display_name="ComfyUI_UltimateSDUpscale",
        needs_submodules=True,
    ),
    CustomNodeSpec(
        node_id="ComfyUI_Fill-Nodes",
        repo_url="https://github.com/filliptm/ComfyUI_Fill-Nodes",
        display_name="ComfyUI_Fill-Nodes",
        xfail=True,
        xfail_reason="API-based nodes require anthropic/openai API keys",
    ),
    CustomNodeSpec(
        node_id="ComfyUI_Fill-ChatterBox",
        repo_url="https://github.com/filliptm/ComfyUI_Fill-ChatterBox",
        display_name="ComfyUI_Fill-ChatterBox",
        priority="Mid",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-KJNodes",
        repo_url="https://github.com/kijai/ComfyUI-KJNodes",
        display_name="ComfyUI-KJNodes",
        priority="Mid",
        depends_on=("ComfyUI-VideoHelperSuite",),
    ),
    CustomNodeSpec(
        node_id="ComfyUI-NormalCrafterWrapper",
        repo_url="https://github.com/AIWarper/ComfyUI-NormalCrafterWrapper",
        display_name="ComfyUI-NormalCrafterWrapper",
        priority="Mid",
        depends_on=("ComfyUI-VideoHelperSuite",),
        inject_version="0.1.0",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-FlashVSR_Ultra_Fast",
        repo_url="https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast",
        display_name="ComfyUI-FlashVSR_Ultra_Fast",
        priority="Mid",
        depends_on=("ComfyUI-VideoHelperSuite",),
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Lotus",
        repo_url="https://github.com/kijai/ComfyUI-Lotus",
        display_name="ComfyUI-Lotus",
        priority="Mid",
        inject_version="0.1.0",
    ),
    CustomNodeSpec(
        node_id="Bjornulf_custom_nodes",
        repo_url="https://github.com/justUmen/Bjornulf_custom_nodes",
        display_name="Bjornulf_custom_nodes",
        priority="Mid",
        xfail=True,
        xfail_reason="TTS models auto-download at runtime; complex dependencies",
    ),
    CustomNodeSpec(
        node_id="comfyui_controlnet_aux",
        repo_url="https://github.com/Fannovel16/comfyui_controlnet_aux",
        display_name="comfyui_controlnet_aux",
        priority="Mid",
    ),
    CustomNodeSpec(
        node_id="ControlAltAI-Nodes",
        repo_url="https://github.com/gseth/ControlAltAI-Nodes",
        display_name="ControlAltAI-Nodes",
        priority="Mid",
        depends_on=("comfyui_controlnet_aux",),
    ),
    CustomNodeSpec(
        node_id="ComfyUI-DepthAnythingV2",
        repo_url="https://github.com/kijai/ComfyUI-DepthAnythingV2",
        display_name="ComfyUI-DepthAnythingV2",
        priority="Mid",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-VideoHelperSuite",
        repo_url="https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        display_name="ComfyUI-VideoHelperSuite",
        priority="Mid",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-WD14-Tagger",
        repo_url="https://github.com/pythongosssss/ComfyUI-WD14-Tagger",
        display_name="ComfyUI-WD14-Tagger",
        priority="Mid",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Custom-Scripts",
        repo_url="https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
        display_name="ComfyUI-Custom-Scripts",
        priority="Mid",
    ),
    CustomNodeSpec(
        node_id="ComfyUI-Kolors-MZ",
        repo_url="https://github.com/MinusZoneAI/ComfyUI-Kolors-MZ",
        display_name="ComfyUI-Kolors-MZ",
        priority="Mid",
    ),
]

_SPEC_BY_ID: dict[str, CustomNodeSpec] = {s.node_id: s for s in CUSTOM_NODE_REGISTRY}


def get_spec(node_id: str) -> CustomNodeSpec:
    return _SPEC_BY_ID[node_id]

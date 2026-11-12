"""MAGI-2 Preview DiT against SandAI's reference implementation (bf16, CUDA, tiny random weights).

Top-k routing is discontinuous: bf16 rounding differences between two correct implementations
flip expert choices for a few tokens and those flips grow through the following layers. So the
end-to-end comparison routes every token to every expert (the gather, GEMMs, SwiGLU7, routing
weights and scatter all still run), and sparse routing is checked layer by layer on the
reference's own layer inputs.
"""

import pytest
import torch

import comfy.ops as comfy_ops
from comfy.ldm.magi2.convert import comfy_tensors
from comfy.ldm.magi2.model import Magi2Model

from . import magi2_reference

pytestmark = pytest.mark.skipif(magi2_reference.REFERENCE_SRC is None or not torch.cuda.is_available(),
                                reason="needs MAGI2_REFERENCE_SRC and CUDA")

PREVIEW = dict(num_layers=4, hidden_size=128, head_dim=32, num_query_groups=4, video_in_channels=8, audio_in_channels=6,
               text_in_channels=24, multimodal_layers=(0, 3), moe_layers=(1, 2), moe_heads=2, num_experts=4,
               expert_intermediate_size=32, shared_expert_intermediate_size=32, modality_shared_expert_intermediate_size=32)
TOLERANCE = 2e-2


def _device():
    return torch.device("cuda", torch.cuda.device_count() - 1)


def _reference_preview(ref, device, top_k):
    cfg = ref.config
    config = cfg.ModelConfig(
        num_layers=PREVIEW["num_layers"], hidden_size=PREVIEW["hidden_size"], head_dim=PREVIEW["head_dim"],
        num_query_groups=PREVIEW["num_query_groups"], video_in_channels=PREVIEW["video_in_channels"],
        audio_in_channels=PREVIEW["audio_in_channels"], text_in_channels=PREVIEW["text_in_channels"],
        mm_layers=list(PREVIEW["multimodal_layers"]), num_heads_q=4, num_heads_kv=4,
        moe_config=cfg.MoEConfig(num_experts=4, top_k=top_k, moe_layers=list(PREVIEW["moe_layers"]), expert_intermediate_size=32,
                                 shared_expert_intermediate_size=32, modality_specific_expert_intermediate_size=32, num_heads=2))
    torch.manual_seed(0)
    model = ref.preview.Transformer(config)
    _randomize(model)
    for module in model.modules():
        if isinstance(module, ref.preview.CoreMultiHeadMoE):
            # inference routes with the EMA bias, which SandAI's loader copies over the raw one
            module.router.expert_bias.copy_(module.router.expert_bias_ema)
    return model.to(device).eval()


def _randomize(model):
    """Unit-variance activations (so bf16 noise stays small) and sink logits large enough to matter."""
    generator = torch.Generator().manual_seed(1)
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if name.endswith("rope.bands"):
            continue
        if name.endswith(".sinks"):
            bound = (1.0, 4.0)
        elif value.ndim >= 2:
            fan_in = value.shape[-2] if name.endswith(("W_gate", "W_up", "W_down")) or "mhc_phi" in name else value.shape[-1]
            limit = (3.0 / fan_in) ** 0.5
            bound = (-limit, limit)
        else:
            bound = (-0.3, 0.3)
        value.data.copy_(torch.empty(value.shape).uniform_(*bound, generator=generator))


def _comfy_preview(reference, device, top_k):
    state = {}
    for name, tensor in reference.state_dict().items():
        state.update(comfy_tensors(name, tensor, PREVIEW["multimodal_layers"]))
    model = Magi2Model(**PREVIEW, top_k=top_k, dtype=torch.bfloat16, device=device, operations=comfy_ops.disable_weight_init)
    model.load_state_dict({key: value.contiguous() for key, value in state.items()}, strict=True)
    return model.requires_grad_(False)


def _inputs(device, refs):
    generator = torch.Generator().manual_seed(2)
    rand = lambda *shape: torch.randn(*shape, generator=generator)
    inputs = dict(latent=rand(1, 8, 3, 4, 6), audio=rand(1, 7, 6), text=rand(1, 5, 24).to(torch.bfloat16),
                  null_text=rand(1, 3, 24).to(torch.bfloat16),
                  ref_image=rand(1, 1, 8, 1, 2, 3) if refs else None, ref_token=rand(1, 1, 24).to(torch.bfloat16) if refs else None)
    return {key: None if value is None else value.to(device) for key, value in inputs.items()}


def _reference_forward(ref, model, inputs, timestep, device):
    proxy = ref.preview_proxy.Magi2DataProxy(ref.config.DataProxyConfig())
    sampler = ref.sampler.Magi2PreviewSampler(model, proxy, device, torch.bfloat16)
    has_refs = inputs["ref_image"] is not None
    model_input = sampler.prepare_model_input(
        latent=inputs["latent"], audio_latent=inputs["audio"], txt_feat=inputs["text"], null_txt_feat=inputs["null_text"],
        ref_audio_feat=inputs["audio"].new_zeros(1, 0, 6), ref_image_feat=inputs["ref_image"],
        ref_image_feat_len=torch.tensor([[[2, 3]]], device=device) if has_refs else None,
        ref_image_special_token_embedding=inputs["ref_token"],
        t=torch.tensor(timestep, device=device), cfg_config=ref.preview_proxy.CFGConfig(use_ref_for_uncond=True))
    with torch.no_grad():
        return sampler.forward(model_input)


def _comfy_forward(model, inputs, context, timestep):
    refs = None
    if inputs["ref_image"] is not None:
        refs = [{"latent": inputs["ref_image"][0, 0].unsqueeze(0), "token": inputs["ref_token"][0, 0]}]
    # model_base hands the model sigma * 1000; the reference scheduler truncates it to an integer timestep
    video, audio = model([inputs["latent"], inputs["audio"].transpose(1, 2)],
                         torch.tensor([timestep + 0.4], device=inputs["latent"].device), context.float(), magi2_refs=refs)
    return video, audio.transpose(1, 2)


def _relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


@pytest.mark.parametrize("refs", [False, True])
def test_preview_matches_sandai_reference_end_to_end(refs):
    ref = magi2_reference.load()
    device = _device()
    with ref.eager():
        reference = _reference_preview(ref, device, top_k=4)
        inputs = _inputs(device, refs)
        expected_video, expected_audio = _reference_forward(ref, reference, inputs, 937, device)
    model = _comfy_preview(reference, device, top_k=4)
    for index, context in enumerate((inputs["text"], inputs["null_text"])):
        video, audio = _comfy_forward(model, inputs, context, 937)
        assert video.shape == expected_video[index:index + 1].shape
        assert audio.shape == expected_audio[index:index + 1].shape
        assert _relative_error(video, expected_video[index:index + 1]) < TOLERANCE
        assert _relative_error(audio, expected_audio[index:index + 1]) < TOLERANCE


def test_preview_sparse_routing_layers_match_sandai_reference():
    ref = magi2_reference.load()
    device = _device()
    captured = {}
    with ref.eager():
        reference = _reference_preview(ref, device, top_k=2)
        for index, layer in enumerate(reference.block.layers):
            def capture(module, args, output, index=index):
                dispatcher = args[3]
                captured[index] = (dispatcher._inv_permute(args[0]), dispatcher._inv_permute(output))
            layer.register_forward_hook(capture)
        inputs = _inputs(device, refs=False)
        _reference_forward(ref, reference, inputs, 937, device)
    model = _comfy_preview(reference, device, top_k=2)
    outputs = {}
    for index, layer in enumerate(model.block.layers):

        def teacher(module, args, index=index):
            return (captured[index][0][:args[0].shape[1]].unsqueeze(0).to(args[0].dtype),) + args[1:]

        def capture(module, args, output, index=index):
            outputs[index] = output[0]
        layer.register_forward_pre_hook(teacher)
        layer.register_forward_hook(capture)
    _comfy_forward(model, inputs, inputs["text"], 937)
    for index, output in outputs.items():
        expected = captured[index][1][:output.shape[0]]
        assert _relative_error(output, expected) < TOLERANCE, index

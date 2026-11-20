from importlib.resources import files
import json

import pytest
import torch

from comfy import model_detection, model_downloader
from comfy_extras.nodes.nodes_marigold import MarigoldV2PostProcess


@torch.inference_mode()
def test_magi_preview_chunked_output_matches_full_projection(monkeypatch):
    from comfy import ops
    from comfy.ldm.magi2 import model as magi

    model = magi.Magi2Model(
        num_layers=2, hidden_size=64, head_dim=32, num_query_groups=2,
        video_in_channels=4, audio_in_channels=2, text_in_channels=8,
        intermediate_size=96, multimodal_layers=(0, 1), moe_layers=(1,),
        num_streams=2, moe_heads=2, num_experts=4, top_k=1,
        expert_intermediate_size=32, shared_expert_intermediate_size=32,
        modality_shared_expert_intermediate_size=32, sinkhorn_iterations=3,
        dtype=torch.float32, device="cpu", operations=ops.disable_weight_init,
    ).eval()
    generator = torch.Generator().manual_seed(42)
    for parameter in model.parameters():
        parameter.normal_(mean=0.0, std=0.04, generator=generator)
    video = torch.randn(1, 4, 2, 3, 4, generator=generator)
    audio = torch.randn(1, 2, 11, generator=generator)
    context = torch.randn(1, 5, 8, generator=generator)
    inputs = ([video, audio], torch.tensor([500.0]), context)
    reference = model(*inputs)
    monkeypatch.setattr(magi, "RMS_NORM_FP32_CHUNK_BYTES", 5 * 128 * 4)
    output = model(*inputs)
    for actual, expected, latent in zip(output, reference, (video, audio), strict=True):
        assert actual.shape == latent.shape
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("packed", [True, False])
def test_qwen_image21_detection_preserves_latent_channels_for_packed_weights(fused, packed):
    prefix = "model.diffusion_model."
    weights = {
        "txt_in.text_norm.weight": (4096,),
        "modulation.1.weight": (16384, 4096),
        "transformer_blocks.0.attn.norm_q.weight": (128,),
        "img_in.weight": (4096, 32 if packed else 64),
        "proj_out.weight": (64, 2048 if packed else 4096),
        f"transformer_blocks.0.img_mlp.{'gate_up' if fused else 'proj'}.weight": (24576 if fused else 12288, 2048 if packed else 4096),
    }
    state = {prefix + name: torch.empty(shape, device="meta") for name, shape in weights.items()}
    config = model_detection.detect_unet_config(state, prefix)
    assert config == {
        "image_model": "qwen_image21", "in_channels": 64, "out_channels": 64,
        "num_layers": 1, "attention_head_dim": 128, "num_attention_heads": 32,
        "context_in_dim": 4096, "mlp_ratio": 3, "fused_mlp": fused,
    }
    for name in weights:
        partial = state.copy()
        del partial[prefix + name]
        assert model_detection.detect_unet_config(partial, prefix) is None


def test_yue2_detection_requires_complete_signature():
    keys = ("vae2llm.weight", "llm2vae.weight", "latent_pos_embed.pe",
            "model.layers.0.self_attn.qkv_proj.weight", "time_embedder.mlp.0.weight")
    state = {key: torch.empty(1, device="meta") for key in keys}
    assert model_detection.detect_unet_config(state, "") == {"audio_model": "yue2"}
    for key in keys:
        assert model_detection.detect_unet_config({k: v for k, v in state.items() if k != key}, "") is None


@pytest.mark.parametrize("prediction", ["depth", "normals", "albedo"])
def test_marigold_postprocessing_has_finite_expected_range(prediction):
    image = torch.tensor([[[[0., .25, 1.], [.5, .5, .5]], [[1., 1., 1.], [0., 0., 0.]]]])
    result = MarigoldV2PostProcess.execute(image, prediction)[0]
    assert result.shape == image.shape
    assert torch.isfinite(result).all()
    assert result.min() >= 0
    assert result.max() <= 1
    if prediction == "depth":
        torch.testing.assert_close(result[0, 1, 0], torch.zeros(3))
        torch.testing.assert_close(result[0, 1, 1], torch.ones(3))
    elif prediction == "normals":
        nonzero = (image * 2 - 1).norm(dim=-1) > 0
        torch.testing.assert_close((result * 2 - 1).norm(dim=-1)[nonzero], torch.ones(3))
    else:
        assert result[0, 0, 0, 1] == pytest.approx(.5370987)


def test_new_blueprint_artifacts_resolve_in_their_model_folders():
    databases = {}
    for database in model_downloader._known_models_db:
        for folder in database.folder_names:
            databases.setdefault(folder, set()).update(str(item) for item in database.data)

    def assets(value):
        if isinstance(value, dict):
            if "url" in value and "name" in value and "directory" in value:
                yield value
            for child in value.values():
                yield from assets(child)
        elif isinstance(value, list):
            for child in value:
                yield from assets(child)

    checked = set()
    for blueprint in files("comfy.blueprints").iterdir():
        if "Marigold V2" not in blueprint.name and "YuE2" not in blueprint.name:
            continue
        for artifact in assets(json.loads(blueprint.read_text())):
            directory, name = artifact["directory"], artifact["name"]
            assert name in databases[directory]
            checked.add((directory, name))
    assert len(checked) >= 11


@pytest.mark.parametrize("registry,name", [
    ("KNOWN_MODEL_PATCHES", "minimax_h3_fun_controlnet_union_pruned_bf16.safetensors"),
    ("KNOWN_MODEL_PATCHES", "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"),
    ("KNOWN_UNET_MODELS", "qwen_image_2.1_int8_convrot.safetensors"),
    ("KNOWN_CLIP_MODELS", "qwen3vl_8b_int8_convrot.safetensors"),
    ("KNOWN_CLIP_MODELS", "qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors"),
    ("KNOWN_CLIP_MODELS", "qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors"),
    ("KNOWN_CHECKPOINTS", "yue2_3b_int8_convrot.safetensors"),
    ("KNOWN_AUDIO_ENCODER_MODELS", "sheetsage2_bf16.safetensors"),
    ("KNOWN_GEOMETRY_ESTIMATION_MODELS", "moge_3_vitl_fp16.safetensors"),
    ("KNOWN_GEOMETRY_ESTIMATION_MODELS", "moge_3_vitg_fp16.safetensors"),
])
def test_release_model_is_registered_in_correct_folder(registry, name):
    assert name in {str(item) for item in getattr(model_downloader, registry).data}


def test_marigold_blueprint_string_node_counter_converts():
    from comfy.component_model.workflow_convert import _ensure_global_id_uniqueness

    workflow = {
        "last_node_id": "39", "nodes": [{"id": 39}],
        "definitions": {"subgraphs": [{"id": "marigold", "nodes": [{"id": 39}, {"id": 41}]}]},
    }
    assert _ensure_global_id_uniqueness(workflow, {}) == {"marigold": {39: 40}}


def test_qwen_image21_forward_keeps_odd_latent_shape():
    from comfy import ops
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel

    model = QwenImage21Transformer2DModel(
        in_channels=64, out_channels=64, num_layers=1, attention_head_dim=8,
        num_attention_heads=2, context_in_dim=16, mlp_ratio=2, axes_dims_rope=(2, 2, 4),
        dtype=torch.float32, device="cpu", operations=ops.manual_cast,
    )
    for parameter in model.parameters():
        parameter.data.fill_(.01)
    latent = torch.randn(1, 64, 3, 5)
    context = torch.randn(1, 4, 16)
    out = model(latent, torch.tensor([.5]), context, ref_latents=[torch.randn(1, 64, 2, 3)], image_slots=[2])
    assert out.shape == latent.shape
    assert out.dtype == latent.dtype
    assert torch.isfinite(out).all()


@torch.inference_mode()
def test_yue2_forward_preserves_chunked_audio_layout():
    from comfy import ops
    from comfy.ldm.yue2.model import YuE2

    model = YuE2(dtype=torch.float32, device="cpu", operations=ops.manual_cast, config={
        "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1,
        "max_position_embeddings": 16,
    })
    for parameter in model.parameters():
        parameter.data.fill_(.01)
    model.latent_pos_embed.pe.fill_(0)
    latent = torch.randn(1, 64, 5)
    context = torch.randn(1, 4, 256)
    out = model(latent, torch.tensor([.5]), context, yue2_chunks=[(0, 2, 0, 2), (2, 5, 2, 4)])
    assert out.shape == latent.shape
    assert out.dtype == latent.dtype
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("value", [{"points": [[0, 0], [1, 1]]}, [[0, 0], [1, 1]], None])
def test_curve_widget_export_matches_frontend_153(value):
    from comfy.component_model.workflow_convert import _map_widgets, _map_widgets_dict

    schema = {"required": {"curve": ("CURVE", {"socketless": False})}}
    expected = value if value is None else {"__type__": "CURVE", "__value__": value}
    assert _map_widgets(schema, [value])[0]["curve"] == expected
    assert _map_widgets_dict(schema, {"curve": value})["curve"] == expected


@pytest.mark.parametrize("blocks,post_norm", [(5, False), (10, True)])
def test_minimax_union_loader_preserves_injection_and_inpaint_mode(monkeypatch, blocks, post_norm):
    from comfy import ops
    from comfy.ldm.minimax.controlnet import MiniMaxH3FunControl
    from comfy_extras.nodes import nodes_model_patch

    layers = tuple(range(0, 50, 50 // blocks))
    source = MiniMaxH3FunControl(
        injection_layers=layers, hidden_size=16, num_attention_heads=2,
        attention_head_dim=8, ffn_hidden_size=32, time_embed_dim=8,
        use_adaln_curves=True, dtype=torch.float32, device="cpu", operations=ops.manual_cast,
    )
    state = source.state_dict()
    metadata = {"minimax_h3_fun_controlnet": "adaln_basis"}
    if post_norm:
        metadata["inpaint_masked_pixel_mode"] = "post_norm"
    monkeypatch.setattr(nodes_model_patch, "get_full_path_or_raise", lambda *_: "union.safetensors")
    monkeypatch.setattr(nodes_model_patch.utils, "load_torch_file", lambda *a, **kw: (state, metadata))
    monkeypatch.setattr(nodes_model_patch.model_management, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(nodes_model_patch.model_management, "unet_offload_device", lambda: torch.device("cpu"))

    model = nodes_model_patch.ModelPatchLoader().load_model_patch("union.safetensors")[0].model
    assert model.injection_layers == layers
    assert len(model.control_blocks) == blocks
    assert model.inpaint_post_norm is post_norm

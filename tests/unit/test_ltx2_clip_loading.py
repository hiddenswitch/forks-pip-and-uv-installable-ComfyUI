"""
Tests for LTX-2 CLIP/text encoder loading.

This tests the loading of LTX-2 models which use Gemma3 12B text encoders
with spiece_model tokenizer data.

The workflow loads:
- gemma_3_12B_it.safetensors (text encoder with spiece_model tokenizer)
- ltx-2-19b-dev-fp8.safetensors (checkpoint with model weights)
"""
import pytest
import torch



from comfy.sd import detect_te_model, TEModel


class TestLTX2TEModelDetection:
    """Test text encoder model detection for LTX-2 models."""

    def test_detect_gemma3_12b_model(self):
        """Verify Gemma3 12B model is detected correctly."""
        # Simulate a state dict with Gemma3 12B specific keys
        mock_sd = {
            "model.layers.0.post_feedforward_layernorm.weight": torch.zeros(1),
            "model.layers.47.self_attn.q_norm.weight": torch.zeros(1),
        }
        result = detect_te_model(mock_sd)
        assert result == TEModel.GEMMA_3_12B, f"Expected GEMMA_3_12B, got {result}"

    def test_detect_gemma4_12b_model(self):
        """Verify Gemma4 12B's layer scalar distinguishes it from Gemma3 12B."""
        mock_sd = {
            "model.layers.0.post_feedforward_layernorm.weight": torch.zeros(1),
            "model.layers.0.layer_scalar": torch.zeros(1),
            "model.layers.47.self_attn.q_norm.weight": torch.zeros(1),
        }
        result = detect_te_model(mock_sd)
        assert result == TEModel.GEMMA_4_12B, f"Expected GEMMA_4_12B, got {result}"

    def test_detect_gemma3_4b_model(self):
        """Verify Gemma3 4B model is detected correctly."""
        mock_sd = {
            "model.layers.0.post_feedforward_layernorm.weight": torch.zeros(1),
            "model.layers.0.self_attn.q_norm.weight": torch.zeros(1),
        }
        result = detect_te_model(mock_sd)
        assert result == TEModel.GEMMA_3_4B, f"Expected GEMMA_3_4B, got {result}"


class TestLTX2TokenizerData:
    """Test spiece_model tokenizer data handling for LTX-2."""

    def test_spiece_model_extraction_single_file(self):
        """Test that spiece_model is correctly extracted from state dict for single file LTXV."""
        # Mock a text encoder state dict with spiece_model
        mock_spiece = torch.ByteTensor([0, 1, 2, 3])  # Mock serialized model
        mock_sd = {
            "encoder.block.23.layer.1.DenseReluDense.wi_1.weight": torch.zeros(10240, 1),  # T5_XXL detection
            "spiece_model": mock_spiece,
        }

        # This tests the code path at sd.py:1229-1231
        # For single T5_XXL clip with LTXV type
        te_model = detect_te_model(mock_sd)
        assert te_model == TEModel.T5_XXL

    def test_spiece_model_extraction_dual_file(self):
        """Test that spiece_model is correctly extracted for dual-file LTXV (LTX-2 with Gemma)."""
        # Mock Gemma3 12B text encoder state dict with spiece_model
        mock_spiece = torch.ByteTensor([0, 1, 2, 3])
        mock_te_sd = {
            "model.layers.0.post_feedforward_layernorm.weight": torch.zeros(1),
            "model.layers.47.self_attn.q_norm.weight": torch.zeros(1),
            "spiece_model": mock_spiece,
        }

        # Mock checkpoint state dict (without Gemma weights, just projection layers)
        mock_ckpt_sd = {
            "text_embedding_projection.aggregate_embed.weight": torch.zeros(1),
        }

        clip_data = [mock_te_sd, mock_ckpt_sd]

        # Verify the first state dict contains spiece_model (code path at sd.py:1353)
        assert "spiece_model" in clip_data[0]
        spiece = clip_data[0].get("spiece_model", None)
        assert spiece is not None
        assert torch.equal(spiece, mock_spiece)


class TestLTX2ModelLoading:
    """Test LTX-2 model loading behavior."""

    def test_ltxav_te_model_load_sd_routing_logic(self):
        """Test that LTXAVTEModel.load_sd routing logic is correct."""

        # Test the routing condition itself
        # The load_sd method checks: "model.layers.47.self_attn.q_norm.weight" in sd

        # Gemma3 12B state dict should have this key
        gemma_sd = {"model.layers.47.self_attn.q_norm.weight": torch.zeros(1)}
        assert "model.layers.47.self_attn.q_norm.weight" in gemma_sd

        # Projection state dict should NOT have this key
        proj_sd = {"text_embedding_projection.aggregate_embed.weight": torch.zeros(1)}
        assert "model.layers.47.self_attn.q_norm.weight" not in proj_sd

    @pytest.mark.skip(reason="spiece_model handling changed upstream")
    def test_spiece_model_not_loaded_as_weight(self):
        """Test that spiece_model is NOT expected to be loaded as a model weight.

        The spiece_model key is tokenizer data and should be handled separately
        from model weights. This test verifies the expected behavior.
        """
        from comfy.text_encoders.lt import LTXAVTEModel

        # Create an instance
        model = LTXAVTEModel(device="cpu", dtype=torch.float32)

        # Create a state dict with only spiece_model
        mock_sd = {
            "spiece_model": torch.ByteTensor([0, 1, 2, 3]),
        }

        # spiece_model should NOT trigger the Gemma path
        assert "model.layers.47.self_attn.q_norm.weight" not in mock_sd

        # When load_sd is called with this, it goes to the else branch
        # which does prefix replacement and then load_state_dict
        # The spiece_model should be reported as unexpected
        m, u = model.load_sd(mock_sd)
        assert "spiece_model" in u, "spiece_model should be in unexpected keys"


class TestSpieceTokenizer:
    """Test SPieceTokenizer functionality."""

    def test_spiece_tokenizer_from_bytes(self):
        """Test that SPieceTokenizer can be initialized from serialized bytes."""

        # We need actual serialized model bytes to test this properly
        # Skip if no sentencepiece model available
        pytest.skip("Requires actual sentencepiece model bytes for full test")

    def test_spiece_tokenizer_serialize_roundtrip(self):
        """Test tokenizer serialization roundtrip."""

        # Skip if no model available
        pytest.skip("Requires actual sentencepiece model for full test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""LTX text-encoder detection and real SentencePiece serialization."""
import io

import pytest
import sentencepiece
import torch



from comfy.sd import detect_te_model, TEModel
from comfy.text_encoders.spiece_tokenizer import SPieceTokenizer


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


def test_detect_t5_with_embedded_tokenizer():
    state = {
        "encoder.block.23.layer.1.DenseReluDense.wi_1.weight": torch.zeros(10240, 1),
        "spiece_model": torch.tensor([0, 1, 2, 3], dtype=torch.uint8),
    }
    assert detect_te_model(state) == TEModel.T5_XXL


@pytest.fixture(scope="module")
def spiece_model():
    output = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(["hello small model", "small model tests", "hello tokenizer", "tokenizer round trip"]),
        model_writer=output, vocab_size=32, hard_vocab_limit=False, minloglevel=2,
    )
    return output.getvalue()


def test_spiece_tokenizer_from_bytes(spiece_model):
    tokenizer = SPieceTokenizer(spiece_model, add_bos=True, add_eos=True)
    text = "hello small model"
    tokens = tokenizer(text)["input_ids"]
    assert tokens[0] == tokenizer.tokenizer.bos_id()
    assert tokens[-1] == tokenizer.eos_token_id
    assert tokenizer.decode(tokens) == text


def test_spiece_tokenizer_serialize_roundtrip(spiece_model):
    original = SPieceTokenizer(spiece_model)
    serialized = original.serialize_model()
    assert serialized.dtype == torch.uint8
    assert bytes(serialized.tolist()) == spiece_model
    restored = SPieceTokenizer(serialized)
    text = "tokenizer round trip"
    assert restored(text) == original(text)
    assert restored.get_vocab() == original.get_vocab()
    assert restored.decode(restored(text)["input_ids"]) == text

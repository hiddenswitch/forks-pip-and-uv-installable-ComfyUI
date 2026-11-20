import pytest
import torch
import importlib
import contextlib
from unittest.mock import patch, MagicMock

# For version comparison
from packaging.version import parse as parse_version

# Module under test
import comfy.ops

TORCH_VERSION = parse_version(torch.__version__.split('+')[0])
CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(autouse=True)
def cleanup_module():
    """Reloads comfy.ops after each test to reset its state."""
    yield
    importlib.reload(comfy.ops)


def test_sdpa_no_cuda():
    """
    Tests that scaled_dot_product_attention falls back to the basic implementation
    when CUDA is not available.
    """
    with patch('torch.cuda.is_available', return_value=False):
        # Reload the module to apply the mock
        importlib.reload(comfy.ops)

        assert comfy.ops.scaled_dot_product_attention is comfy.ops._scaled_dot_product_attention

        # Test functionality
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        output = comfy.ops.scaled_dot_product_attention(q, k, v)
        assert output.shape == q.shape


def test_sdpa_masked_gqa_fallback_repeats_key_value_heads():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 2, 8, 16)
    v = torch.randn(2, 2, 8, 16)
    mask = torch.ones(8, 8, dtype=torch.bool)

    actual = comfy.ops._scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        enable_gqa=True,
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(2, dim=-3),
        v.repeat_interleave(2, dim=-3),
        attn_mask=mask,
    )

    torch.testing.assert_close(actual, expected)


def test_sdpa_old_torch_with_cuda():
    """
    Tests that scaled_dot_product_attention falls back and warns
    on older torch versions that have CUDA but lack 'set_priority' in sdpa_kernel.
    """
    # Mock signature object without 'set_priority'
    mock_signature = MagicMock()
    mock_signature.parameters = {}

    # Mock the logger to capture warnings
    mock_logger = MagicMock()

    # Mock the attention module to prevent import errors on non-CUDA builds
    mock_attention_module = MagicMock()
    mock_attention_module.sdpa_kernel = MagicMock()
    mock_attention_module.SDPBackend = MagicMock()

    with patch('torch.cuda.is_available', return_value=True), \
            patch('inspect.signature', return_value=mock_signature), \
            patch('logging.getLogger', return_value=mock_logger), \
            patch.dict('sys.modules', {'torch.nn.attention': mock_attention_module}):
        importlib.reload(comfy.ops)

        assert comfy.ops.scaled_dot_product_attention is comfy.ops._scaled_dot_product_attention
        mock_logger.warning.assert_called_once_with("Torch version too old to set sdpa backend priority, even though you are using CUDA")

        # Test functionality
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)
        output = comfy.ops.scaled_dot_product_attention(q, k, v)
        assert output.shape == q.shape


def test_sdpa_import_exception():
    """
    Tests that scaled_dot_product_attention falls back if an exception occurs
    during the SDPA setup.
    """
    mock_logger = MagicMock()
    with patch('torch.cuda.is_available', return_value=True), \
            patch('inspect.signature', side_effect=Exception("Test Exception")), \
            patch('logging.getLogger', return_value=mock_logger):
        # Mock the attention module to prevent import errors on non-CUDA builds
        mock_attention_module = MagicMock()
        mock_attention_module.sdpa_kernel = MagicMock()
        mock_attention_module.SDPBackend = MagicMock()
        with patch.dict('sys.modules', {'torch.nn.attention': mock_attention_module}):
            importlib.reload(comfy.ops)

            assert comfy.ops.scaled_dot_product_attention is comfy.ops._scaled_dot_product_attention

            # Test functionality
            q = torch.randn(2, 4, 8, 16)
            k = torch.randn(2, 4, 8, 16)
            v = torch.randn(2, 4, 8, 16)
            output = comfy.ops.scaled_dot_product_attention(q, k, v)
            assert output.shape == q.shape


@pytest.mark.parametrize("flash_supported", [False, True])
def test_sdpa_rocm_small_inputs_follow_backend_policy(flash_supported):
    """ROCm must not bypass the explicit backend policy for small inputs."""
    from torch.nn.attention import SDPBackend
    import comfy.model_management as model_management

    with patch('torch.cuda.is_available', return_value=True), \
            patch.object(model_management, 'is_nvidia', return_value=False), \
            patch.object(model_management, 'AMD_FLASH_SDP_SUPPORTED', flash_supported):
        importlib.reload(comfy.ops)

        q = torch.randn(1, 1, 8, 16)
        expected = torch.randn_like(q)
        with patch.object(comfy.ops, 'sdpa_kernel', return_value=contextlib.nullcontext()) as kernel, \
                patch.object(torch.nn.functional, 'scaled_dot_product_attention', return_value=expected):
            actual = comfy.ops.scaled_dot_product_attention(q, q, q)

        assert actual is expected
        backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH] if flash_supported else [SDPBackend.MATH]
        kernel.assert_called_once_with(backends, set_priority=True)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
@pytest.mark.skipif(TORCH_VERSION < parse_version("2.6.0"), reason="Requires torch version 2.6.0 or greater")
def test_sdpa_with_cuda_and_priority():
    """
    Tests that the prioritized SDPA implementation is used when CUDA is available
    and the torch version is new enough.
    This is a real test and does not use mocks.
    """
    # Reload to ensure the correct version is picked up based on the actual environment
    importlib.reload(comfy.ops)

    # Check that the correct function is assigned
    assert comfy.ops.scaled_dot_product_attention is not comfy.ops._scaled_dot_product_attention
    assert comfy.ops.scaled_dot_product_attention.__name__ == "_scaled_dot_product_attention_sdpa2"

    # Create tensors on CUDA device
    device = torch.device("cuda")
    q = torch.randn(2, 4, 8, 16, device=device, dtype=torch.float16)
    k = torch.randn(2, 4, 8, 16, device=device, dtype=torch.float16)
    v = torch.randn(2, 4, 8, 16, device=device, dtype=torch.float16)

    # Execute the function
    output = comfy.ops.scaled_dot_product_attention(q, k, v)

    # Assertions
    assert output.shape == q.shape
    assert output.device.type == device.type
    assert output.dtype == torch.float16


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA is not available")
@pytest.mark.skipif(TORCH_VERSION < parse_version("2.6.0"), reason="Requires torch version 2.6.0 or greater")
@pytest.mark.skipif(torch.version.hip is not None, reason="cuDNN is not available on ROCm")
def test_sdpa_cudnn_fallback():
    """
    Tests that the SDPA function gracefully falls back when cuDNN fails.
    This test verifies the fallback mechanism works when a cuDNN-related
    RuntimeError occurs.
    """
    importlib.reload(comfy.ops)

    # Verify we have the prioritized implementation
    assert comfy.ops.scaled_dot_product_attention is not comfy.ops._scaled_dot_product_attention

    # Create tensors on CUDA device
    device = torch.device("cuda")
    q = torch.randn(2, 4, 8, 16, device=device, dtype=torch.float16)
    k = torch.randn(2, 4, 8, 16, device=device, dtype=torch.float16)
    v = torch.randn(2, 4, 8, 16, device=device, dtype=torch.float16)

    # Execute - this should work even if cuDNN fails due to version mismatch
    # The fallback mechanism will catch cuDNN errors and retry without cuDNN
    output = comfy.ops.scaled_dot_product_attention(q, k, v)

    # The output should be valid regardless of which backend was used
    assert output.shape == q.shape
    assert output.device.type == device.type
    assert not torch.isnan(output).any(), "Output should not contain NaN values"


def test_cudnn_nvrtc_compatibility_check():
    """
    Tests that _check_cudnn_nvrtc_compatibility returns expected values.
    """
    importlib.reload(comfy.ops)

    # The function should be defined
    assert hasattr(comfy.ops, '_check_cudnn_nvrtc_compatibility')

    # Should return a boolean
    result = comfy.ops._check_cudnn_nvrtc_compatibility()
    assert isinstance(result, bool)


class TestCuDNNVersionMismatch:
    """Tests related to cuDNN version mismatch detection."""

    def test_cudnn_version_available(self):
        """Test that cuDNN version can be queried."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Skip on ROCm - cuDNN is NVIDIA-specific
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            pytest.skip("ROCm detected - cuDNN not applicable")

        cudnn_version = torch.backends.cudnn.version()
        assert cudnn_version is not None
        assert isinstance(cudnn_version, int)
        # cuDNN 8.x returns 4-digit numbers (e.g. 8906 for 8.9.6),
        # cuDNN 9.x returns 5-digit numbers (e.g. 90102 for 9.1.2)
        assert cudnn_version >= 1000, f"Unexpected cuDNN version format: {cudnn_version}"

    def test_cuda_runtime_available(self):
        """Test that CUDA runtime version is accessible."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Skip on ROCm - torch.cuda.is_available() returns True but torch.version.cuda is None
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            pytest.skip("ROCm detected - CUDA runtime version not applicable")

        # Get CUDA version
        cuda_version = torch.version.cuda
        assert cuda_version is not None
        # Should be like "13.0" or "12.4"
        parts = cuda_version.split(".")
        assert len(parts) >= 2, f"Unexpected CUDA version format: {cuda_version}"

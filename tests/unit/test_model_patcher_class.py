"""
Tests for ModelPatcher class selection based on dynamic VRAM configuration.

Verifies that:
1. get_model_patcher_class() returns the correct class based on aimdo state
2. CoreModelPatcher module property works for backwards compatibility
3. Module property is evaluated at access time, not import time
4. ModelPatcherDynamic behavior differs based on load device (CPU vs GPU)
"""
import pytest
import torch
from unittest.mock import patch


def get_test_devices():
    """Returns list of (device, device_name) tuples for parameterization."""
    devices = [(torch.device('cpu'), 'cpu')]
    if torch.cuda.is_available():
        devices.append((torch.device('cuda:0'), 'cuda'))
    return devices


class TestGetModelPatcherClass:
    """Tests for get_model_patcher_class() factory function."""

    def test_returns_model_patcher_without_aimdo(self):
        """Without aimdo allocator, returns standard ModelPatcher."""
        from comfy import memory_management
        from comfy.model_patcher import get_model_patcher_class, ModelPatcher

        with patch.object(memory_management, 'aimdo_allocator', None):
            result = get_model_patcher_class()
            assert result is ModelPatcher

    def test_returns_dynamic_patcher_with_aimdo(self):
        """With aimdo allocator set, returns ModelPatcherDynamic."""
        from comfy import memory_management
        from comfy.model_patcher import get_model_patcher_class, ModelPatcherDynamic

        with patch.object(memory_management, 'aimdo_allocator', object()):
            result = get_model_patcher_class()
            assert result is ModelPatcherDynamic


class TestCoreModelPatcherModuleProperty:
    """Tests for CoreModelPatcher module-level property."""

    def test_core_model_patcher_is_accessible(self):
        """CoreModelPatcher can be accessed from the module."""
        from comfy import model_patcher
        # Should not raise AttributeError
        assert hasattr(model_patcher, 'CoreModelPatcher')

    def test_core_model_patcher_returns_model_patcher_without_aimdo(self):
        """Without aimdo, CoreModelPatcher returns ModelPatcher."""
        from comfy import memory_management, model_patcher

        with patch.object(memory_management, 'aimdo_allocator', None):
            assert model_patcher.CoreModelPatcher is model_patcher.ModelPatcher

    def test_core_model_patcher_returns_dynamic_with_aimdo(self):
        """With aimdo, CoreModelPatcher returns ModelPatcherDynamic."""
        from comfy import memory_management, model_patcher

        with patch.object(memory_management, 'aimdo_allocator', object()):
            assert model_patcher.CoreModelPatcher is model_patcher.ModelPatcherDynamic

    def test_core_model_patcher_evaluated_at_access_time(self):
        """CoreModelPatcher is evaluated each time it's accessed."""
        from comfy import memory_management, model_patcher

        # Start without aimdo
        with patch.object(memory_management, 'aimdo_allocator', None):
            first = model_patcher.CoreModelPatcher
            assert first is model_patcher.ModelPatcher

        # Enable aimdo
        with patch.object(memory_management, 'aimdo_allocator', object()):
            second = model_patcher.CoreModelPatcher
            assert second is model_patcher.ModelPatcherDynamic

        # Back to no aimdo
        with patch.object(memory_management, 'aimdo_allocator', None):
            third = model_patcher.CoreModelPatcher
            assert third is model_patcher.ModelPatcher

    def test_direct_import_works(self):
        """from comfy.model_patcher import CoreModelPatcher works."""
        from comfy import memory_management

        with patch.object(memory_management, 'aimdo_allocator', None):
            # This import triggers the module property
            from comfy.model_patcher import CoreModelPatcher, ModelPatcher
            assert CoreModelPatcher is ModelPatcher


class TestModelPatcherIsDynamic:
    """Tests for is_dynamic() method on patcher classes."""

    @pytest.mark.parametrize("load_device,device_name", get_test_devices())
    def test_model_patcher_is_not_dynamic(self, load_device, device_name):
        """Standard ModelPatcher returns False for is_dynamic() on any device."""
        from comfy import memory_management
        from comfy.model_patcher import ModelPatcher

        model = torch.nn.Linear(10, 10)
        with patch.object(memory_management, 'aimdo_allocator', None):
            patcher = ModelPatcher(model, load_device=load_device, offload_device=torch.device('cpu'))
            assert patcher.is_dynamic() is False

    @pytest.mark.parametrize("load_device,device_name", get_test_devices())
    def test_model_patcher_dynamic_class_and_is_dynamic(self, load_device, device_name):
        """
        ModelPatcherDynamic behavior depends on load_device:
        - CPU: __new__ returns ModelPatcher, is_dynamic() is False
        - GPU: __new__ returns ModelPatcherDynamic, is_dynamic() is True
        """
        from comfy.model_patcher import ModelPatcher, ModelPatcherDynamic

        model = torch.nn.Linear(10, 10)
        patcher = ModelPatcherDynamic(model, load_device=load_device, offload_device=torch.device('cpu'))

        if load_device.type == 'cpu':
            # CPU: ModelPatcherDynamic.__new__ returns ModelPatcher
            assert type(patcher) is ModelPatcher
            assert patcher.is_dynamic() is False
        else:
            # GPU: ModelPatcherDynamic.__new__ returns actual ModelPatcherDynamic
            assert type(patcher) is ModelPatcherDynamic
            assert patcher.is_dynamic() is True


class TestModelPatcherParent:
    """Tests for parent tracking used by cloned/delegate patchers."""

    def test_parent_can_be_assigned_for_core_delegate(self):
        """Custom nodes can attach a dynamic parent to a core patcher delegate."""
        from comfy import memory_management
        from comfy.model_patcher import ModelPatcher

        model = torch.nn.Linear(10, 10)
        with patch.object(memory_management, 'aimdo_allocator', None):
            parent = ModelPatcher(model, load_device=torch.device('cpu'), offload_device=torch.device('cpu'))
            delegate = ModelPatcher(model, load_device=torch.device('cpu'), offload_device=torch.device('cpu'))

        delegate.parent = parent

        assert delegate.parent is parent

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for dynamic patcher")
    def test_core_delegate_can_parent_to_dynamic_patcher(self):
        """Regression for custom nodes replacing a dynamic patcher with a core delegate."""
        from comfy.model_patcher import ModelPatcher, ModelPatcherDynamic

        model = torch.nn.Linear(10, 10)
        parent = ModelPatcherDynamic(
            model,
            load_device=torch.device('cuda:0'),
            offload_device=torch.device('cpu'),
        )
        delegate = ModelPatcher(
            model,
            load_device=parent.load_device,
            offload_device=parent.offload_device,
            _force_core=True,
        )

        delegate.parent = parent

        assert parent.is_dynamic() is True
        assert delegate.is_dynamic() is False
        assert delegate.parent is parent


class TestModelPatcherFactoryCompatibility:
    """Tests for legacy direct ModelPatcher(...) construction."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for dynamic patcher")
    def test_direct_model_patcher_uses_factory_for_cuda_when_aimdo_enabled(self):
        from comfy import memory_management
        from comfy.model_patcher import ModelPatcher, ModelPatcherDynamic

        model = torch.nn.Linear(10, 10)
        with patch.object(memory_management, 'aimdo_allocator', object()):
            patcher = ModelPatcher(
                model,
                load_device=torch.device('cuda:0'),
                offload_device=torch.device('cpu'),
            )

        assert type(patcher) is ModelPatcherDynamic
        assert patcher.is_dynamic() is True

    def test_direct_model_patcher_keeps_cpu_on_core_when_aimdo_enabled(self):
        from comfy import memory_management
        from comfy.model_patcher import ModelPatcher

        model = torch.nn.Linear(10, 10)
        with patch.object(memory_management, 'aimdo_allocator', object()):
            patcher = ModelPatcher(
                model,
                load_device=torch.device('cpu'),
                offload_device=torch.device('cpu'),
            )

        assert type(patcher) is ModelPatcher
        assert patcher.is_dynamic() is False


def test_dynamic_patcher_keeps_large_weights_evictable():
    from comfy.model_patcher import dynamic_weight_requires_force_load

    assert dynamic_weight_requires_force_load(16 * 1024)
    assert not dynamic_weight_requires_force_load(20 * 1024 ** 3)
    assert dynamic_weight_requires_force_load(20 * 1024 ** 3, structurally_required=True)


def test_dynamic_vbar_block_ignores_force_loaded_modules():
    from comfy.model_patcher import extend_dynamic_vbar_block

    first = (object(), 1024, 256)
    later = (first[0], 2048, 512)

    assert extend_dynamic_vbar_block(None, None) is None
    assert extend_dynamic_vbar_block(None, first) is first
    assert extend_dynamic_vbar_block(first, None) is first
    assert extend_dynamic_vbar_block(first, later) == (first[0], 1024, 1536)


def test_attention_config_lora_isolated_from_original_and_weight_patches():
    import json

    from comfy.ldm.modules.attention import ComfyAttention
    from comfy.model_patcher import ModelPatcher

    model = torch.nn.Module()
    model.attention = ComfyAttention()
    model.linear = torch.nn.Linear(2, 2)
    patcher = ModelPatcher(model, load_device=torch.device('cpu'), offload_device=torch.device('cpu'))
    clone = patcher.clone()
    weights_uuid = clone.patches_uuid
    config = {'attention': 'unavailable-test-backend'}
    encoded = torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)

    assert clone.add_patches({'attention.config': ('config', (encoded,))}) == ['attention.config']
    assert clone.get_model_object('attention').config == config
    assert patcher.get_model_object('attention').config is None
    assert model.attention.config is None
    assert clone.patches_uuid == weights_uuid
    assert clone.patches == {}

    clone.add_patches({'linear.weight': ('diff', (torch.ones_like(model.linear.weight),))})
    assert clone.patches_uuid != weights_uuid
    assert 'linear.weight' in clone.patches

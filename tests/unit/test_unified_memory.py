from types import SimpleNamespace

import pytest
import torch

from comfy import memory_management
from comfy import model_management

GiB = 1024 ** 3


@pytest.fixture(autouse=True)
def _fresh_unified_cache():
    model_management._UNIFIED_MEMORY_DEVICES.clear()
    yield
    model_management._UNIFIED_MEMORY_DEVICES.clear()


def _cuda_properties(monkeypatch, integrated):
    monkeypatch.setattr(model_management.torch.cuda, "get_device_properties",
                        lambda device=None: SimpleNamespace(is_integrated=integrated))
    monkeypatch.setattr(model_management.torch.cuda, "current_device", lambda: 0)


def _system_memory(monkeypatch, total, available):
    monkeypatch.setattr(model_management.system_memory, "virtual_memory_total", lambda: total)
    monkeypatch.setattr(model_management.system_memory, "virtual_memory_available", lambda: available)


def _cuda_allocator(monkeypatch, free, total, reserved, active):
    monkeypatch.setattr(model_management.torch.cuda, "mem_get_info", lambda dev=None: (free, total))
    monkeypatch.setattr(model_management.torch.cuda, "memory_stats", lambda dev=None: {
        "reserved_bytes.all.current": reserved,
        "active_bytes.all.current": active,
    })
    for name in ("is_intel_xpu", "is_ascend_npu", "is_mlu"):
        monkeypatch.setattr(model_management, name, lambda: False)


def test_mps_is_unified_memory():
    assert model_management.is_unified_memory_device(torch.device("mps"))


def test_integrated_cuda_is_unified_memory(monkeypatch):
    _cuda_properties(monkeypatch, integrated=True)
    assert model_management.is_unified_memory_device(torch.device("cuda", 0))
    assert model_management.is_unified_memory_device(torch.device("cuda"))


def test_discrete_cuda_and_cpu_are_not_unified_memory(monkeypatch):
    _cuda_properties(monkeypatch, integrated=False)
    assert not model_management.is_unified_memory_device(torch.device("cuda", 0))
    assert not model_management.is_unified_memory_device(torch.device("cpu"))


def test_unified_memory_devices_get_the_shared_vram_state(monkeypatch):
    _cuda_properties(monkeypatch, integrated=True)
    gpu = model_management.CPUState.GPU
    for requested in model_management.VRAMState:
        assert model_management.vram_state_for_device(requested, gpu, torch.device("cuda", 0)) == model_management.VRAMState.SHARED
    assert model_management.vram_state_for_device(
        model_management.VRAMState.DISABLED, model_management.CPUState.MPS, torch.device("mps")
    ) == model_management.VRAMState.SHARED


def test_discrete_gpu_vram_state_is_unchanged(monkeypatch):
    _cuda_properties(monkeypatch, integrated=False)
    gpu = model_management.CPUState.GPU
    for requested in (model_management.VRAMState.NORMAL_VRAM, model_management.VRAMState.LOW_VRAM,
                      model_management.VRAMState.NO_VRAM, model_management.VRAMState.HIGH_VRAM):
        assert model_management.vram_state_for_device(requested, gpu, torch.device("cuda", 0)) == requested
    assert model_management.vram_state_for_device(
        model_management.VRAMState.NORMAL_VRAM, model_management.CPUState.CPU, torch.device("cpu")
    ) == model_management.VRAMState.DISABLED


def test_integrated_cuda_free_memory_comes_from_system_memory(monkeypatch):
    # GB10: cudaMemGetInfo reports MemFree (108 GiB) while MemAvailable is 102 GiB; device
    # allocations come out of the same RAM, so system memory is the budget.
    _cuda_properties(monkeypatch, integrated=True)
    _system_memory(monkeypatch, total=121 * GiB, available=102 * GiB)
    _cuda_allocator(monkeypatch, free=108 * GiB, total=121 * GiB, reserved=3 * GiB, active=1 * GiB)
    device = torch.device("cuda", 0)

    assert model_management.get_free_memory(device) == 102 * GiB + 2 * GiB
    assert model_management.get_free_memory(device, torch_free_too=True) == (104 * GiB, 2 * GiB)
    assert model_management.get_total_memory(device) == 121 * GiB
    assert model_management.get_total_memory(device, torch_total_too=True) == (121 * GiB, 3 * GiB)


def test_discrete_cuda_free_memory_is_unchanged(monkeypatch):
    _cuda_properties(monkeypatch, integrated=False)
    _system_memory(monkeypatch, total=128 * GiB, available=100 * GiB)
    _cuda_allocator(monkeypatch, free=20 * GiB, total=24 * GiB, reserved=3 * GiB, active=1 * GiB)
    device = torch.device("cuda", 0)

    assert model_management.get_free_memory(device, torch_free_too=True) == (22 * GiB, 2 * GiB)
    assert model_management.get_total_memory(device, torch_total_too=True) == (24 * GiB, 3 * GiB)


def test_mps_free_memory_counts_the_allocator_cache(monkeypatch):
    _system_memory(monkeypatch, total=512 * GiB, available=90 * GiB)
    monkeypatch.setattr(model_management.torch.mps, "driver_allocated_memory", lambda: 14 * GiB)
    monkeypatch.setattr(model_management.torch.mps, "current_allocated_memory", lambda: 12 * GiB)
    device = torch.device("mps")

    assert model_management.get_free_memory(device, torch_free_too=True) == (92 * GiB, 2 * GiB)
    assert model_management.get_total_memory(device, torch_total_too=True) == (512 * GiB, 14 * GiB)


def test_cpu_free_memory_is_unchanged(monkeypatch):
    _system_memory(monkeypatch, total=64 * GiB, available=40 * GiB)
    assert model_management.get_free_memory(torch.device("cpu"), torch_free_too=True) == (40 * GiB, 40 * GiB)
    assert model_management.get_total_memory(torch.device("cpu"), torch_total_too=True) == (64 * GiB, 64 * GiB)


def test_dynamic_vram_budget_uses_system_memory_on_integrated_cuda(monkeypatch):
    _cuda_properties(monkeypatch, integrated=True)
    _system_memory(monkeypatch, total=121 * GiB, available=60 * GiB)
    _cuda_allocator(monkeypatch, free=108 * GiB, total=121 * GiB, reserved=0, active=0)
    monkeypatch.setattr(memory_management, "aimdo_allocator", None)

    assert memory_management.dynamic_vram_available_memory(torch.device("cuda", 0)) == 60 * GiB


def test_unified_memory_reserves_what_the_os_cannot_reclaim(monkeypatch):
    # Device allocations on unified memory are unevictable RAM, like pinned pages, so the
    # pinned-memory reserve is held back from them as well.
    monkeypatch.setattr(model_management, "is_unified_memory_device", lambda device=None: True)
    monkeypatch.setattr(model_management, "pinned_memory_reserve", lambda: 12 * GiB)
    assert model_management.extra_reserved_memory() == model_management.EXTRA_RESERVED_VRAM + 12 * GiB

    monkeypatch.setattr(model_management, "is_unified_memory_device", lambda device=None: False)
    assert model_management.extra_reserved_memory() == model_management.EXTRA_RESERVED_VRAM


def test_unified_memory_offload_devices_are_the_load_devices(monkeypatch):
    device = torch.device("cuda", 0)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: device)
    monkeypatch.setattr(model_management, "is_unified_memory_device", lambda d=None: True)
    monkeypatch.setattr(model_management, "vram_state", model_management.VRAMState.SHARED)
    monkeypatch.setattr(memory_management, "aimdo_allocator", None)

    assert model_management.unet_offload_device() == device
    assert model_management.vae_offload_device() == model_management.vae_device()
    assert model_management.text_encoder_offload_device() == model_management.text_encoder_device()


def test_discrete_offload_devices_are_unchanged(monkeypatch):
    monkeypatch.setattr(model_management, "get_torch_device", lambda: torch.device("cuda", 0))
    monkeypatch.setattr(model_management, "is_unified_memory_device", lambda d=None: False)
    monkeypatch.setattr(model_management, "vram_state", model_management.VRAMState.NORMAL_VRAM)

    assert model_management.unet_offload_device() == torch.device("cpu")
    assert model_management.vae_offload_device() == torch.device("cpu")
    assert model_management.text_encoder_offload_device() == torch.device("cpu")


def test_integrated_cuda_keeps_the_text_encoder_on_the_gpu(monkeypatch):
    device = torch.device("cuda", 0)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: device)
    monkeypatch.setattr(model_management, "vram_state", model_management.VRAMState.SHARED)
    monkeypatch.setattr(model_management, "cpu_state", model_management.CPUState.GPU)
    monkeypatch.setattr(model_management, "should_use_fp16", lambda *a, **k: True)
    monkeypatch.setattr(memory_management, "aimdo_allocator", None)

    assert model_management.text_encoder_device() == device


def test_pinned_memory_is_disabled_on_unified_memory(monkeypatch):
    monkeypatch.setattr(model_management, "is_unified_memory_device", lambda d=None: True)
    assert model_management.pinned_memory_disabled_for_device()
    monkeypatch.setattr(model_management, "is_unified_memory_device", lambda d=None: False)
    assert not model_management.pinned_memory_disabled_for_device()


def test_model_parallel_offload_device_follows_unified_memory(monkeypatch):
    device = torch.device("cuda", 0)
    monkeypatch.setattr(memory_management, "aimdo_allocator", None)
    monkeypatch.setattr(model_management, "is_unified_memory_device", lambda d=None: True)
    assert model_management.offload_device_for(device) == device

    monkeypatch.setattr(memory_management, "aimdo_allocator", object())
    assert model_management.offload_device_for(device) == torch.device("cpu")

    monkeypatch.setattr(memory_management, "aimdo_allocator", None)
    monkeypatch.setattr(model_management, "is_unified_memory_device", lambda d=None: False)
    assert model_management.offload_device_for(device) == torch.device("cpu")


def test_pipeline_stage_is_built_on_a_unified_memory_device(monkeypatch):
    from comfy.pipeline_parallel import loader
    from comfy.pipeline_parallel.types import PipelineStagePlan

    device = torch.device("cuda", 0)
    monkeypatch.setattr(model_management, "offload_device_for", lambda load, offload=None: load)
    built = {}

    class StopAfterPlacement(Exception):
        pass

    class Config:
        def __init__(self, unet_config):
            self.unet_config = dict(unet_config)
            self.quant_config = None
            self.custom_operations = None
            self.optimizations = {}
            self.supported_inference_dtypes = (torch.bfloat16,)

        def set_inference_dtype(self, *args, **kwargs):
            pass

        def get_model(self, state, prefix, device=None):
            built["device"] = device
            raise StopAfterPlacement

    class Reader:
        def load_keys(self, keys):
            return {}

    monkeypatch.setattr(loader, "get_pipeline_stage_spec", lambda name: SimpleNamespace(owns_key=lambda *a: True))
    monkeypatch.setattr(model_management, "unet_manual_cast", lambda *a, **k: None)
    stage = PipelineStagePlan(0, device, 0, 1, 1, frozenset())
    with pytest.raises(StopAfterPlacement):
        loader._load_pipeline_stage(Reader(), Config({"image_model": "qwen_image"}), {}, "", {}, "unet.safetensors",
                                    SimpleNamespace(size=1), stage, {}, True, torch.bfloat16, torch.device("cpu"))
    assert built["device"] == device

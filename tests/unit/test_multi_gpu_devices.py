from types import SimpleNamespace

import torch

from comfy import model_management
from comfy.cli_args_types import Configuration


def _use_gpu_state(monkeypatch):
    monkeypatch.setattr(model_management, "cpu_state", model_management.CPUState.GPU)
    monkeypatch.setattr(model_management, "is_ascend_npu", lambda: False)
    monkeypatch.setattr(model_management, "is_mlu", lambda: False)


def test_cuda_multi_gpu_devices_are_enumerated(monkeypatch):
    _use_gpu_state(monkeypatch)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: True)
    monkeypatch.setattr(model_management, "is_amd", lambda: False)
    monkeypatch.setattr(model_management, "is_intel_xpu", lambda: False)
    monkeypatch.setattr(model_management.torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(model_management.torch.cuda, "current_device", lambda: 1)

    assert model_management.get_all_torch_devices() == [
        torch.device("cuda", 0),
        torch.device("cuda", 1),
        torch.device("cuda", 2),
    ]
    assert model_management.get_all_torch_devices(exclude_current=True) == [
        torch.device("cuda", 0),
        torch.device("cuda", 2),
    ]


def test_process_local_model_manager_owns_only_current_device(monkeypatch):
    monkeypatch.setattr(
        model_management,
        "args",
        Configuration(model_management_device_scope="local"),
    )
    monkeypatch.setattr(
        model_management,
        "get_torch_device",
        lambda: torch.device("cuda", 1),
    )

    assert model_management.get_model_management_devices() == [
        torch.device("cuda", 1),
    ]
    assert model_management.get_model_management_devices(exclude_current=True) == []


def test_shared_model_manager_owns_all_discovered_devices(monkeypatch):
    monkeypatch.setattr(
        model_management,
        "args",
        Configuration(model_management_device_scope="all"),
    )
    discovered = [torch.device("cuda", 0), torch.device("cuda", 1)]
    monkeypatch.setattr(
        model_management,
        "get_all_torch_devices",
        lambda exclude_current=False: discovered[1:] if exclude_current else discovered,
    )

    assert model_management.get_model_management_devices() == discovered
    assert model_management.get_model_management_devices(exclude_current=True) == discovered[1:]


def test_rocm_multi_gpu_devices_are_enumerated_through_cuda(monkeypatch):
    _use_gpu_state(monkeypatch)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: False)
    monkeypatch.setattr(model_management, "is_amd", lambda: True)
    monkeypatch.setattr(model_management, "is_intel_xpu", lambda: False)
    monkeypatch.setattr(model_management.torch.cuda, "device_count", lambda: 2)

    assert model_management.get_all_torch_devices() == [
        torch.device("cuda", 0),
        torch.device("cuda", 1),
    ]


def test_xpu_multi_gpu_devices_are_enumerated(monkeypatch):
    _use_gpu_state(monkeypatch)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: False)
    monkeypatch.setattr(model_management, "is_amd", lambda: False)
    monkeypatch.setattr(model_management, "is_intel_xpu", lambda: True)
    monkeypatch.setattr(
        model_management.torch,
        "xpu",
        SimpleNamespace(device_count=lambda: 2, current_device=lambda: 0),
        raising=False,
    )

    assert model_management.get_all_torch_devices() == [
        torch.device("xpu", 0),
        torch.device("xpu", 1),
    ]
    assert model_management.get_all_torch_devices(exclude_current=True) == [
        torch.device("xpu", 1),
    ]


def test_gpu_device_options_are_added_only_for_multiple_gpus(monkeypatch):
    monkeypatch.setattr(
        model_management,
        "get_all_torch_devices",
        lambda: [torch.device("cuda", 0), torch.device("cuda", 1)],
    )

    assert model_management.get_gpu_device_options() == [
        "default",
        "cpu",
        "gpu:0",
        "gpu:1",
    ]
    assert model_management.get_gpu_device_options_no_cpu() == [
        "default",
        "gpu:0",
        "gpu:1",
    ]


def test_gpu_device_options_hide_gpu_indices_for_single_gpu(monkeypatch):
    monkeypatch.setattr(
        model_management,
        "get_all_torch_devices",
        lambda: [torch.device("cuda", 0)],
    )

    assert model_management.get_gpu_device_options() == ["default", "cpu"]
    assert model_management.get_gpu_device_options_no_cpu() == ["default"]


def test_resolve_gpu_device_option_uses_enumerated_device_order(monkeypatch):
    devices = [
        torch.device("cuda", 0),
        torch.device("cuda", 1),
        torch.device("cuda", 2),
    ]
    monkeypatch.setattr(model_management, "get_all_torch_devices", lambda: devices)

    assert model_management.resolve_gpu_device_option("default") is None
    assert model_management.resolve_gpu_device_option(None) is None
    assert model_management.resolve_gpu_device_option("cpu") == torch.device("cpu")
    assert model_management.resolve_gpu_device_option("gpu:0") == torch.device("cuda", 0)
    assert model_management.resolve_gpu_device_option("gpu:2") == torch.device("cuda", 2)
    assert model_management.resolve_gpu_device_option("gpu:3") is None
    assert model_management.resolve_gpu_device_option("gpu:not-an-int") is None
    assert model_management.resolve_gpu_device_option("cuda:0") is None


def test_get_all_torch_devices_falls_back_to_current_device_for_unknown_gpu_backend(monkeypatch):
    _use_gpu_state(monkeypatch)
    fallback = torch.device("privateuseone", 0)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: False)
    monkeypatch.setattr(model_management, "is_amd", lambda: False)
    monkeypatch.setattr(model_management, "is_intel_xpu", lambda: False)
    monkeypatch.setattr(model_management, "get_torch_device", lambda: fallback)

    assert model_management.get_all_torch_devices() == [fallback]

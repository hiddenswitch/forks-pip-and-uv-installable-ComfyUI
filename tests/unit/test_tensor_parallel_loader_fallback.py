import logging
from types import SimpleNamespace

import torch

from comfy.cli_args_types import Configuration
from comfy.execution_context import context_configuration
from comfy.tensor_parallel import loader


def _stub_checkpoint(monkeypatch, image_model: str):
    monkeypatch.setattr(loader, "SafetensorsCheckpointReader", lambda path: SimpleNamespace(path=path))
    monkeypatch.setattr(loader, "_normalize_detection_state", lambda reader: ({}, {}, ""))
    monkeypatch.setattr(
        loader.model_detection,
        "model_config_from_unet",
        lambda state, prefix, metadata=None: SimpleNamespace(unet_config={"image_model": image_model}),
    )
    monkeypatch.setattr(loader.model_management, "get_torch_device", lambda: torch.device("cuda", 0))
    monkeypatch.setattr(
        loader.model_management,
        "get_all_torch_devices",
        lambda exclude_current=False: [torch.device("cuda", 0), torch.device("cuda", 1)],
    )


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_unsupported_family_falls_through_to_the_ordinary_loaders(monkeypatch):
    _stub_checkpoint(monkeypatch, "qwen_image")

    def _must_not_launch(*args, **kwargs):
        raise AssertionError("tensor-parallel ranks must not launch for an unsupported family")

    monkeypatch.setattr(loader, "load_diffusion_model_tensor_parallel", _must_not_launch)
    records = _Records()
    loader.logger.addHandler(records)
    try:
        with context_configuration(Configuration(tensor_parallel_size=2)):
            assert loader.try_load_diffusion_model_tensor_parallel("model.safetensors") is None
    finally:
        loader.logger.removeHandler(records)
    assert any("qwen_image" in message and "without tensor parallelism" in message for message in records.messages)


def test_supported_family_loads_tensor_parallel(monkeypatch):
    _stub_checkpoint(monkeypatch, "flux2")
    calls = []

    def _launch(unet_path, devices, model_options=None, disable_dynamic=False, external_runtime=None):
        calls.append((unet_path, tuple(devices)))
        return "loaded"

    monkeypatch.setattr(loader, "load_diffusion_model_tensor_parallel", _launch)
    with context_configuration(Configuration(tensor_parallel_size=2)):
        assert loader.try_load_diffusion_model_tensor_parallel("model.safetensors") == "loaded"
    assert calls == [("model.safetensors", (torch.device("cuda", 0), torch.device("cuda", 1)))]


def test_size_one_skips_the_checkpoint(monkeypatch):
    def _no_reader(path):
        raise AssertionError("size one must not open the checkpoint")

    monkeypatch.setattr(loader, "SafetensorsCheckpointReader", _no_reader)
    with context_configuration(Configuration(tensor_parallel_size=1)):
        assert loader.try_load_diffusion_model_tensor_parallel("model.safetensors") is None

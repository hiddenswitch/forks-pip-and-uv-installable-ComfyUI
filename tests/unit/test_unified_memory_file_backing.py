import pytest
import safetensors.torch
import torch

from comfy import memory_management
from comfy import model_management
from comfy import utils
from comfy.model_patcher import ModelPatcher

accelerator = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a device distinct from the CPU")


def _checkpoint(tmp_path):
    torch.manual_seed(3)
    module = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Linear(32, 16))
    path = tmp_path / "model.safetensors"
    safetensors.torch.save_file({k: v.contiguous() for k, v in module.state_dict().items()}, str(path))
    return module, path


def _fresh_module():
    return torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Linear(32, 16))


def _mapping_permissions(address):
    with open("/proc/self/maps") as maps:
        for line in maps:
            bounds, permissions = line.split()[:2]
            low, high = (int(value, 16) for value in bounds.split("-"))
            if low <= address < high:
                return permissions
    raise AssertionError(f"address {address:#x} is not mapped")


def test_file_backed_safetensors_match_the_checkpoint(tmp_path):
    reference, path = _checkpoint(tmp_path)
    state = utils.load_safetensors_file_backed(str(path))[0]

    for key, value in reference.state_dict().items():
        assert memory_management.is_file_backed(state[key])
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)


@pytest.mark.skipif(not utils.sys.platform.startswith("linux"), reason="reads /proc/self/maps")
def test_file_backed_safetensors_are_a_read_only_private_mapping(tmp_path):
    """A writable private mapping lets the CUDA driver break copy-on-write when it copies from it.

    On a GB10 every byte copied to the device from such a mapping became an anonymous page as
    well, so a load cost twice its weights; comfy-aimdo maps checkpoints PROT_READ for the same reason.
    """
    _, path = _checkpoint(tmp_path)
    state = utils.load_safetensors_file_backed(str(path))[0]
    assert _mapping_permissions(state["0.weight"].data_ptr()) == "r--p"


def test_include_keys_limits_the_file_backed_state(tmp_path):
    _, path = _checkpoint(tmp_path)
    state = utils.load_safetensors_file_backed(str(path), include_keys={"1.bias"})[0]
    assert list(state) == ["1.bias"]


def test_unified_memory_loads_use_the_file_backed_reader(tmp_path, monkeypatch):
    _, path = _checkpoint(tmp_path)
    monkeypatch.setattr(model_management, "offload_shares_device_memory", lambda device=None: True)
    state = utils.load_torch_file(str(path))
    assert all(memory_management.is_file_backed(v) for v in state.values())

    monkeypatch.setattr(model_management, "offload_shares_device_memory", lambda device=None: False)
    state = utils.load_torch_file(str(path))
    assert not any(memory_management.is_file_backed(v) for v in state.values())


def test_recorded_cpu_weights_become_views_of_the_file(tmp_path):
    reference, path = _checkpoint(tmp_path)
    state = utils.load_safetensors_file_backed(str(path))[0]
    module = _fresh_module()
    module.load_state_dict(state)

    memory_management.record_file_backing(module, state)

    for name, param in module.named_parameters():
        assert memory_management.is_file_backed(param)
        assert param.data_ptr() == state[name].data_ptr()
        torch.testing.assert_close(param, reference.state_dict()[name], rtol=0, atol=0)


def test_weights_that_differ_from_the_file_are_not_recorded(tmp_path):
    _, path = _checkpoint(tmp_path)
    state = utils.load_safetensors_file_backed(str(path))[0]
    module = _fresh_module()
    module.load_state_dict(state)
    with torch.no_grad():
        module[0].weight.mul_(2.0)

    memory_management.record_file_backing(module, state)

    assert not memory_management.is_file_backed(module[0].weight)
    assert "0.weight" not in module._comfy_file_backing
    assert memory_management.is_file_backed(module[0].bias)


@accelerator
def test_releasing_device_weights_restores_the_file_views(tmp_path):
    reference, path = _checkpoint(tmp_path)
    device = torch.device("cuda", torch.cuda.current_device())
    state = utils.load_safetensors_file_backed(str(path))[0]
    module = _fresh_module().to(device)
    module.load_state_dict(state)
    memory_management.record_file_backing(module, state)

    nbytes = sum(p.nbytes for p in module.parameters())
    assert memory_management.file_backed_device_bytes(module) == nbytes
    assert memory_management.release_file_backed_weights(module) == nbytes
    assert memory_management.file_backed_device_bytes(module) == 0
    for name, param in module.named_parameters():
        assert param.device.type == "cpu"
        assert memory_management.is_file_backed(param)
        torch.testing.assert_close(param, reference.state_dict()[name], rtol=0, atol=0)

    module.to(device)
    assert memory_management.file_backed_device_bytes(module) == nbytes


@accelerator
def test_unloading_on_unified_memory_releases_instead_of_copying(tmp_path, monkeypatch):
    _, path = _checkpoint(tmp_path)
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(model_management, "offload_shares_device_memory", lambda d=None: True)
    state = utils.load_safetensors_file_backed(str(path))[0]
    module = _fresh_module().to(device)
    module.load_state_dict(state)
    memory_management.record_file_backing(module, state)
    nbytes = sum(p.nbytes for p in module.parameters())
    patcher = ModelPatcher(module, load_device=device, offload_device=device, _force_core=True)
    loaded = model_management.LoadedModel(patcher)
    loaded.model_load()
    assert patcher.reclaimable_non_vbar_memory() == nbytes

    assert loaded.model_unload(memory_to_free=1) is True

    assert all(memory_management.is_file_backed(p) for p in module.parameters())
    assert patcher.reclaimable_non_vbar_memory() == 0
    loaded.model_load()
    assert all(p.device == device for p in module.parameters())


@accelerator
def test_discrete_unload_still_offloads_to_the_cpu(tmp_path, monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(model_management, "offload_shares_device_memory", lambda d=None: False)
    module = _fresh_module()
    patcher = ModelPatcher(module, load_device=device, offload_device=torch.device("cpu"), _force_core=True)
    loaded = model_management.LoadedModel(patcher)
    loaded.model_load()
    assert all(p.device == device for p in module.parameters())
    assert patcher.reclaimable_non_vbar_memory() == patcher.loaded_size()

    loaded.model_unload()

    assert all(p.device.type == "cpu" for p in module.parameters())
    assert not any(memory_management.is_file_backed(p) for p in module.parameters())


@accelerator
def test_release_finds_backing_recorded_on_a_submodule(tmp_path):
    # BaseModel records on its diffusion_model; the patcher releases from the BaseModel.
    reference, path = _checkpoint(tmp_path)
    device = torch.device("cuda", torch.cuda.current_device())
    state = utils.load_safetensors_file_backed(str(path))[0]
    root = torch.nn.Module()
    root.diffusion_model = _fresh_module().to(device)
    root.diffusion_model.load_state_dict(state)
    memory_management.record_file_backing(root.diffusion_model, state)

    nbytes = sum(p.nbytes for p in root.parameters())
    assert memory_management.file_backed_device_bytes(root) == nbytes
    assert memory_management.release_file_backed_weights(root) == nbytes
    assert all(memory_management.is_file_backed(p) for p in root.parameters())

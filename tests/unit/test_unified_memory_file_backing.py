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


@pytest.mark.parametrize("unified, amd, disable_mmap, expected", [
    (True, False, False, True),
    (False, True, False, True),
    (False, False, False, False),
    (False, True, True, False),
])
@pytest.mark.skipif(utils.sys.platform == "win32", reason="requires POSIX mmap")
def test_cpu_checkpoint_mapping_policy(tmp_path, monkeypatch, unified, amd, disable_mmap, expected):
    reference, path = _checkpoint(tmp_path)
    monkeypatch.setattr(memory_management, "aimdo_allocator", None)
    monkeypatch.setattr(model_management, "offload_shares_device_memory", lambda device=None: unified)
    monkeypatch.setattr(model_management, "is_amd", lambda: amd)
    monkeypatch.setattr(utils, "DISABLE_MMAP", disable_mmap)
    state = utils.load_torch_file(str(path))
    for key, value in state.items():
        assert memory_management.is_file_backed(value) == expected
        torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("disable_pins, disable_mmap, expected", [
    (True, False, True), (False, False, False), (True, True, False),
])
@pytest.mark.skipif(utils.sys.platform == "win32", reason="requires POSIX mmap")
def test_discrete_nvidia_weight_backing_without_pins(tmp_path, monkeypatch, disable_pins, disable_mmap, expected):
    reference, path = _checkpoint(tmp_path)
    monkeypatch.setattr(memory_management, "aimdo_allocator", None)
    monkeypatch.setattr(model_management, "offload_shares_device_memory", lambda: False)
    monkeypatch.setattr(model_management, "is_amd", lambda: False)
    monkeypatch.setattr(model_management, "is_nvidia", lambda: True)
    monkeypatch.setattr(model_management, "DISABLE_PINNED_MEMORY", disable_pins)
    monkeypatch.setattr(utils, "DISABLE_MMAP", disable_mmap)
    state = utils.load_torch_file(str(path))
    module = _fresh_module()
    module.load_state_dict(state)
    memory_management.record_file_backing(module, state)
    for name, param in module.named_parameters():
        assert (param.data_ptr() == state[name].data_ptr()) == expected
        torch.testing.assert_close(param, reference.state_dict()[name], rtol=0, atol=0)


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


def test_file_backed_slices_and_concatenations_do_not_clone_on_cpu(tmp_path):
    _, path = _checkpoint(tmp_path)
    state = utils.load_safetensors_file_backed(str(path))[0]
    weight = state["0.weight"]

    second_half = memory_management.slice_tensor(weight, (slice(16, 32), slice(None)))
    assert memory_management.is_file_backed(second_half)
    assert second_half.data_ptr() == weight[16:32].data_ptr()
    if utils.sys.platform.startswith("linux"):
        assert _mapping_permissions(second_half.data_ptr()) == "r--p"

    pieces = [weight.narrow(0, 0, 8), weight.narrow(0, 16, 8)]
    concat = memory_management.concatenate_tensors(pieces, dim=0)
    assert isinstance(concat, memory_management.FileBackedConcat)
    assert concat._materialized is None

    copied = concat.to(dtype=torch.float64)
    assert concat._materialized is None
    torch.testing.assert_close(copied, torch.cat(pieces, dim=0).to(dtype=torch.float64), rtol=0, atol=0)

    copied_with_device = concat.to(device="cpu", dtype=torch.float64)
    assert concat._materialized is None
    torch.testing.assert_close(copied_with_device, copied, rtol=0, atol=0)

    destination = torch.empty_like(torch.cat(pieces, dim=0))
    destination.copy_(concat)
    assert concat._materialized is None
    torch.testing.assert_close(destination, torch.cat(pieces, dim=0), rtol=0, atol=0)


@accelerator
def test_quantized_tensor_and_concat_weights_release_and_reload(tmp_path, monkeypatch):
    from comfy import ops
    from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(model_management, "offload_shares_device_memory", lambda d=None: True)

    qdata = torch.randint(-64, 64, (32, 64), dtype=torch.int8)
    scale = torch.linspace(0.1, 0.5, 32, dtype=torch.float32).unsqueeze(1)
    quant_meta = torch.tensor(list(b'{"format": "int8_tensorwise"}'), dtype=torch.uint8)
    path = tmp_path / "quant.safetensors"
    safetensors.torch.save_file(
        {"0.weight": qdata, "0.weight_scale": scale, "0.comfy_quant": quant_meta},
        str(path),
    )

    state = utils.load_safetensors_file_backed(str(path))[0]
    fused_weight = memory_management.concatenate_tensors(
        [state["0.weight"].narrow(0, 0, 16), state["0.weight"].narrow(0, 16, 16)],
        dim=0,
    )
    state["0.weight"] = fused_weight

    mp_ops = ops.mixed_precision_ops({"0": {"format": "int8_tensorwise"}}, torch.bfloat16)
    module = torch.nn.Sequential(mp_ops.Linear(64, 32, bias=False, device=device, dtype=torch.bfloat16))
    module.load_state_dict(state, strict=False)
    assert fused_weight._materialized is None
    memory_management.record_file_backing(module, state)
    assert fused_weight._materialized is None

    assert isinstance(module[0].weight, QuantizedTensor)
    assert isinstance(module[0].weight._params, TensorWiseINT8Layout.Params)
    expected_dequant = module[0].weight.dequantize().cpu()

    patcher = ModelPatcher(module, load_device=device, offload_device=device, _force_core=True)
    loaded = model_management.LoadedModel(patcher)
    loaded.model_load()
    assert patcher.reclaimable_non_vbar_memory() == qdata.nbytes

    assert loaded.model_unload(memory_to_free=1) is True
    assert memory_management.is_file_backed(module[0].weight)
    assert module[0].weight.device.type == "cpu"
    assert module[0].weight._qdata.device.type == "cpu"
    assert patcher.reclaimable_non_vbar_memory() == 0
    assert fused_weight._materialized is None

    loaded.model_load()
    assert module[0].weight.device == device
    assert module[0].weight._qdata.device == device
    assert fused_weight._materialized is None
    torch.testing.assert_close(module[0].weight.dequantize().cpu(), expected_dequant, rtol=0, atol=0)




@pytest.mark.parametrize("wrapped", [True, False])
def test_checkpoint_index_reads_only_requested_shards_and_keeps_their_paths(tmp_path, monkeypatch, wrapped):
    import json

    reference, shard = _checkpoint(tmp_path)
    blob = shard.rename(tmp_path / "weights.blob")
    index = {key: blob.name for key in reference.state_dict()}
    index["unused"] = "unstaged.safetensors"
    path = tmp_path / "model.safetensors.index.json"
    path.write_text(json.dumps({"weight_map": index} if wrapped else index))

    monkeypatch.setattr(memory_management, "aimdo_allocator", None)
    monkeypatch.setattr(utils, "_file_backed_loads", lambda: False)
    state, metadata = utils.load_torch_file(str(path), include_keys={"1.bias"}, return_metadata=True)
    assert list(state) == ["1.bias"] and metadata is None
    assert state["1.bias"].untyped_storage()._comfy_source_path == str(blob)
    torch.testing.assert_close(state["1.bias"], reference[1].bias, rtol=0, atol=0)


@accelerator
def test_discrete_unload_under_host_pressure_restores_only_matching_weights(tmp_path, monkeypatch):
    _, path = _checkpoint(tmp_path)
    device = torch.device("cuda", torch.cuda.current_device())
    state = utils.load_safetensors_file_backed(str(path))[0]
    module = _fresh_module()
    module.load_state_dict(state)
    memory_management.record_file_backing(module, state)
    patcher = ModelPatcher(module, load_device=device, offload_device=torch.device("cpu"), _force_core=True)
    loaded = model_management.LoadedModel(patcher)
    loaded.model_load()
    with torch.no_grad():
        module[0].bias.add_(1)
    changed = module[0].bias.detach().cpu().clone()
    monkeypatch.setattr(model_management.system_memory, "virtual_memory_available", lambda: 0)
    loaded.model_unload()

    assert module[0].weight.data_ptr() == state["0.weight"].data_ptr()
    assert module[0].bias.data_ptr() != state["0.bias"].data_ptr()
    torch.testing.assert_close(module[0].bias, changed, rtol=0, atol=0)
    loaded.model_load()
    torch.testing.assert_close(module[0].weight.cpu(), state["0.weight"], rtol=0, atol=0)
    torch.testing.assert_close(module[0].bias.cpu(), changed, rtol=0, atol=0)

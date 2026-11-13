import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import typer

from comfy import managed_memory
from comfy.cli_args_types import Configuration
from comfy.cmd import cli as cli_module
from comfy.component_model import setup
from comfy.component_model.configuration import MODEL_MANAGEMENT_ARGS


@pytest.fixture
def fresh_allocator_state(monkeypatch):
    monkeypatch.setattr(managed_memory, "_resolved", None)
    monkeypatch.setattr(managed_memory, "_active", False)
    monkeypatch.setattr(managed_memory, "_allocator", None)


class _FakeDriver:
    """libcuda with one device whose attributes the test chooses."""

    def __init__(self, integrated, concurrent_managed_access, count=1):
        self.attributes = {
            managed_memory.CU_DEVICE_ATTRIBUTE_INTEGRATED: int(integrated),
            managed_memory.CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS: int(concurrent_managed_access),
        }
        self.count = count

    def cuInit(self, flags):
        return 0

    def cuDeviceGetCount(self, count):
        count._obj.value = self.count
        return 0

    def cuDeviceGet(self, device, ordinal):
        device._obj.value = ordinal
        return 0

    def cuDeviceGetAttribute(self, value, attribute, device):
        value._obj.value = self.attributes[attribute]
        return 0


def _cuda_torch(monkeypatch, cuda="13.0", hip=None):
    import torch
    monkeypatch.setattr(torch.version, "cuda", cuda)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)


@pytest.mark.parametrize("integrated, concurrent, expected", [
    (True, True, True),
    (True, False, False),
    (False, True, False),
    (False, False, False),
])
def test_auto_resolves_on_for_nvidia_shared_memory_devices_only(monkeypatch, integrated, concurrent, expected):
    _cuda_torch(monkeypatch)
    monkeypatch.setattr(managed_memory, "_load_driver", lambda: _FakeDriver(integrated, concurrent))
    assert managed_memory.resolve(None) is expected


def test_auto_is_off_for_rocm_and_cpu_only_torch(monkeypatch):
    monkeypatch.setattr(managed_memory, "_load_driver", lambda: _FakeDriver(True, True))
    _cuda_torch(monkeypatch, cuda="6.4", hip="6.4")
    assert managed_memory.resolve(None) is False
    _cuda_torch(monkeypatch, cuda=None)
    assert managed_memory.resolve(None) is False


def test_auto_is_off_without_an_nvidia_driver(monkeypatch):
    _cuda_torch(monkeypatch)
    monkeypatch.setattr(managed_memory, "_load_driver", lambda: None)
    assert managed_memory.resolve(None) is False


def test_explicit_setting_overrides_detection(monkeypatch):
    monkeypatch.setattr(managed_memory, "nvidia_shared_memory_devices", lambda: pytest.fail("explicit values skip detection"))
    assert managed_memory.resolve(True) is True
    assert managed_memory.resolve(False) is False


def test_install_changes_the_allocator_only_when_resolved_on(monkeypatch, fresh_allocator_state):
    import torch
    monkeypatch.delenv("TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK", raising=False)
    installed = []
    monkeypatch.setattr(torch.cuda.memory, "change_current_allocator", installed.append)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(managed_memory, "_load_driver", lambda: object())
    monkeypatch.setattr(managed_memory, "_build_allocator", lambda: "managed-allocator")
    monkeypatch.setattr(managed_memory, "nvidia_shared_memory_devices", lambda: False)

    assert managed_memory.install(None) is False
    assert installed == []
    assert "TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK" not in managed_memory.os.environ

    monkeypatch.setattr(managed_memory, "_resolved", None)
    monkeypatch.setattr(managed_memory, "nvidia_shared_memory_devices", lambda: True)
    assert managed_memory.install(None) is True
    assert installed == ["managed-allocator"]
    assert managed_memory.is_active()
    assert managed_memory.os.environ["TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK"] == "0"

    assert managed_memory.install(None) is True
    assert installed == ["managed-allocator"]


def test_install_refuses_an_explicit_request_after_cuda_initialized(monkeypatch, fresh_allocator_state):
    import torch
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda.memory, "change_current_allocator", lambda a: pytest.fail("must not swap"))
    with pytest.raises(RuntimeError, match="before CUDA is initialized"):
        managed_memory.install(True)


def test_auto_stays_off_when_cuda_was_initialized_first(monkeypatch, fresh_allocator_state):
    import torch
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(managed_memory, "nvidia_shared_memory_devices", lambda: True)
    monkeypatch.setattr(torch.cuda.memory, "change_current_allocator", lambda a: pytest.fail("must not swap"))
    assert managed_memory.install(None) is False


def test_setup_resolves_cpu_configurations_off(monkeypatch):
    seen = []
    monkeypatch.setattr(managed_memory, "install", lambda configured: seen.append(configured) or False)
    setup.setup_unified_memory_allocator(Configuration(cpu=True))
    setup.setup_unified_memory_allocator(Configuration(torch_device="cpu"))
    setup.setup_unified_memory_allocator(Configuration())
    setup.setup_unified_memory_allocator(Configuration(unified_memory_allocator=True))
    assert seen == [False, False, None, True]


def test_auto_stays_off_for_nccl_model_parallel_runs(monkeypatch):
    seen = []
    monkeypatch.setattr(managed_memory, "install", lambda configured: seen.append(configured) or False)
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    setup.setup_unified_memory_allocator(Configuration(pipeline_parallel_size=2))
    setup.setup_unified_memory_allocator(Configuration(tensor_parallel_size=2))
    setup.setup_unified_memory_allocator(Configuration(pipeline_parallel_size=2, unified_memory_allocator=True))
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    setup.setup_unified_memory_allocator(Configuration())
    assert seen == [False, False, True, False]


def test_pre_torch_setup_installs_the_allocator_last(monkeypatch):
    calls = []
    for name in ("setup_environment", "setup_guess_settings", "setup_windows_multi_gpu_defaults",
                 "setup_cuda_devices", "setup_cuda_malloc", "setup_unified_memory_allocator"):
        monkeypatch.setattr(setup, name, lambda *a, _name=name: calls.append(_name))
    monkeypatch.setattr(setup, "setup_debug_hang", lambda config: None)
    setup.setup_pre_torch(Configuration())
    assert calls[-1] == "setup_unified_memory_allocator"
    assert calls.index("setup_cuda_devices") < calls.index("setup_unified_memory_allocator")


def test_the_cli_installs_the_allocator_before_the_distributed_device_is_selected(monkeypatch):
    calls = []
    monkeypatch.setattr(setup, "setup_pre_torch", lambda config: calls.append("pre_torch"))
    monkeypatch.setattr(setup, "setup_distributed_device", lambda distributed: calls.append("device"))
    monkeypatch.setattr(setup, "setup_distributed_runtime", lambda distributed: None)
    monkeypatch.setattr(setup, "setup_post_torch", lambda config: calls.append("post_torch"))
    monkeypatch.setattr(setup, "prepare_distributed_environment", lambda config: SimpleNamespace())
    monkeypatch.setattr(setup, "shutdown_tracing", lambda: None)
    with cli_module._application_runtime(Configuration()):
        pass
    assert calls.index("pre_torch") < calls.index("device")


def test_spawned_ranks_apply_their_configuration(monkeypatch):
    from comfy.execution_context import context_configuration
    from comfy.pipeline_parallel import distributed as pipeline_distributed
    from comfy.tensor_parallel import distributed as tensor_distributed

    seen = []
    monkeypatch.setattr(setup, "setup_unified_memory_allocator", lambda config: seen.append(config.unified_memory_allocator))
    with context_configuration(Configuration(unified_memory_allocator=True)):
        pipeline_distributed._apply_unified_memory_allocator()
        tensor_distributed._apply_unified_memory_allocator()
    assert seen == [True, True]


@pytest.mark.parametrize("arguments, environment, expected", [
    ([], {}, None),
    (["--unified-memory-allocator"], {}, True),
    (["--no-unified-memory-allocator"], {}, False),
    ([], {"COMFYUI_UNIFIED_MEMORY_ALLOCATOR": "1"}, True),
    ([], {"COMFYUI_UNIFIED_MEMORY_ALLOCATOR": "0"}, False),
])
def test_cli_and_environment_parsing(monkeypatch, arguments, environment, expected):
    monkeypatch.delenv("COMFYUI_UNIFIED_MEMORY_ALLOCATOR", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    command = typer.main.get_command(cli_module.app).commands["serve"]
    with command.make_context("serve", ["--no-guess-settings", *arguments]) as context:
        configuration = cli_module._build_config(context.params)
    assert configuration.unified_memory_allocator is expected


def test_setting_is_part_of_model_management_state():
    assert "unified_memory_allocator" in MODEL_MANAGEMENT_ARGS
    assert Configuration().unified_memory_allocator is None


def test_discrete_gpus_keep_the_default_allocator_and_dynamic_vram():
    """A discrete NVIDIA GPU resolves the setting off, so torch's caching allocator and DynamicVRAM stay."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    script = textwrap.dedent("""
        import torch
        from comfy import managed_memory
        from comfy.cli_args_types import Configuration
        from comfy.component_model.setup import setup_unified_memory_allocator
        active = setup_unified_memory_allocator(Configuration())
        torch.zeros(1, device="cuda")
        print(active, torch.cuda.memory_stats()["num_alloc_retries"] >= 0)
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=True)
    assert result.stdout.strip().splitlines()[-1] == "False True"


def test_managed_allocator_backs_cuda_tensors_with_managed_memory():
    """With the setting forced on, torch allocates cudaMallocManaged memory before anything else touches CUDA."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    script = textwrap.dedent("""
        import ctypes, torch
        from comfy import managed_memory
        assert managed_memory.install(True)
        x = torch.arange(1024, device="cuda", dtype=torch.float32)
        assert torch.empty(0, device="cuda").numel() == 0
        value = ctypes.c_int(0)
        CU_POINTER_ATTRIBUTE_IS_MANAGED = 8
        libcuda = ctypes.CDLL("libcuda.so.1")
        assert libcuda.cuPointerGetAttribute(ctypes.byref(value), CU_POINTER_ATTRIBUTE_IS_MANAGED, ctypes.c_uint64(x.data_ptr())) == 0
        print(int(value.value), float((x * 2).sum().item()))
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=True)
    assert result.stdout.strip().splitlines()[-1] == f"1 {float(sum(range(1024)) * 2)}"
    assert "cuMemAllocManaged" not in result.stderr


def test_size_classes():
    assert managed_memory.size_class(1) == 512
    assert managed_memory.size_class(513) == 1024
    assert managed_memory.size_class(1 << 20) == 1 << 20
    assert managed_memory.size_class((1 << 20) + 1) == 2 << 20
    assert managed_memory.size_class((5 << 20) + 7) == 6 << 20


class _RecordingDriver:
    def __init__(self):
        self.next = 0x1000
        self.events = []

    def cuMemAllocManaged(self, pointer, size, flags):
        self.next += size
        pointer._obj.value = self.next
        self.events.append(("alloc", size))
        return 0

    def cuMemAdvise(self, pointer, size, advice, device):
        return 0

    def cuCtxSynchronize(self):
        self.events.append(("sync",))
        return 0

    def cuMemFree_v2(self, pointer):
        self.events.append(("free", pointer))
        return 0


def test_freed_blocks_are_reused_only_after_a_context_synchronization(monkeypatch):
    driver = _RecordingDriver()
    monkeypatch.setattr(managed_memory, "_driver", driver)
    monkeypatch.setattr(managed_memory, "_cache", managed_memory._BlockCache())
    monkeypatch.setattr(managed_memory, "_active", True)

    first = managed_memory._managed_alloc(3 << 20, 0, None)
    managed_memory._managed_free(first, 3 << 20, 0, None)
    assert managed_memory.cached_bytes() == 4 << 20
    assert driver.events == [("alloc", 4 << 20)]

    again = managed_memory._managed_alloc(4 << 20, 0, None)
    assert again == first
    assert driver.events == [("alloc", 4 << 20), ("sync",)]
    assert managed_memory.cached_bytes() == 0

    other = managed_memory._managed_alloc(64, 0, None)
    assert other != first
    assert driver.events[-1] == ("alloc", 512)

    managed_memory._managed_free(again, 4 << 20, 0, None)
    managed_memory._managed_free(other, 64, 0, None)
    assert managed_memory.held_bytes() == (4 << 20) + 512
    assert managed_memory.release_cached() == (4 << 20) + 512
    assert managed_memory.held_bytes() == 0
    assert sorted(e for e in driver.events if e[0] == "free") == sorted([("free", first), ("free", other)])

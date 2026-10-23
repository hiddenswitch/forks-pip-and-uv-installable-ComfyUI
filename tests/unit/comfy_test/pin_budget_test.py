"""The host-memory pin budget.

Pinned (page-locked, CUDA host-registered) pages cannot be reclaimed, and the
kernel's Mlocked/VmPin counters do not show CUDA host registrations, so the only
thing standing between pinning and a host livelock or a cgroup OOM kill is this
budget. It must be derived from the effective memory limit of this process's
cgroup and from memory currently available to it (which reflects every other
process on the host, including QEMU VMs), minus a reserve, and both pin paths
(the dynamic VRAM host buffers in ``comfy.pinned_memory`` and the legacy
``model_management.pin_memory``) must honour the same budget, re-checked at pin
time.

The memory sources are injected: a fake cgroupfs tree, a fake
``/proc/self/cgroup`` and a fake ``psutil.virtual_memory`` whose
``available`` is what /proc/meminfo MemAvailable would report.
"""
import os
import sys
from types import SimpleNamespace

import pytest
import torch

import comfy.model_management as model_management
import comfy.pinned_memory as pinned_memory
import comfy.system_memory as system_memory
from comfy import memory_management
from comfy.cli_args import default_configuration
from comfy.execution_context import context_configuration

GIB = 1024 ** 3
MIB = 1024 ** 2
# appmana-001 as measured: 125 GiB RAM, no swap, QEMU VMs holding ~49 GiB.
HOST_TOTAL = 125 * GIB
VMS = 49 * GIB
CHUNK = GIB
MAX_CHUNKS = 160


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class FakeHost:
    """MemAvailable and cgroup usage that move as this process pins memory.

    ``other`` is memory held by every other process on the host (the VMs).
    ``own`` is this process's unpinned working set. Pinned bytes are counted
    from ``model_management.TOTAL_PINNED_MEMORY`` so each pin shrinks what the
    next budget check sees, exactly as a real registration of freshly allocated
    host memory does.
    """

    def __init__(self, root, total=HOST_TOTAL, other=0, own=0):
        self.root = root
        self.total = total
        self.other = other
        self.own = own
        self.cgroup_dir = None

    def available(self):
        return max(0, self.total - self.other - self.own - model_management.TOTAL_PINNED_MEMORY)

    def virtual_memory(self):
        return SimpleNamespace(total=self.total, available=self.available())

    def v2_cgroup(self, path, **files):
        self.cgroup_dir = self.root / path.strip("/")
        write(self.root.parent / "proc_self_cgroup", f"0::{path}\n")
        for name, value in files.items():
            write(self.cgroup_dir / name.replace("_", ".", 1), f"{value}\n")
        write(self.cgroup_dir / "memory.stat", "anon 1\ninactive_file 0\nactive_file 0\n")
        self.sync()

    def v1_cgroup(self, path, limit):
        self.cgroup_dir = self.root / "memory" / path.strip("/")
        write(self.root.parent / "proc_self_cgroup", f"4:memory:{path}\n")
        write(self.cgroup_dir / "memory.limit_in_bytes", f"{limit}\n")
        write(self.cgroup_dir / "memory.stat", "total_inactive_file 0\n")
        self.sync()

    def sync(self):
        if self.cgroup_dir is None:
            return
        usage = self.own + model_management.TOTAL_PINNED_MEMORY
        name = "memory.usage_in_bytes" if (self.cgroup_dir / "memory.limit_in_bytes").exists() else "memory.current"
        write(self.cgroup_dir / name, f"{usage}\n")


class FakeCudart:
    def __init__(self):
        self.registered = {}

    def cudaHostRegister(self, ptr, size, flags):
        self.registered[ptr] = size
        return 0

    def cudaHostUnregister(self, ptr):
        self.registered.pop(ptr, None)
        return 0


class FakeHostBuffer:
    def __init__(self):
        self.size = 0

    def extend(self, size, reallocate=False, register=True):
        self.size += int(size)
        return 1

    def truncate(self, size, do_unregister=True):
        self.size = int(size)


@pytest.fixture
def sparse_tensor(tmp_path):
    """A file-backed, never-touched CPU tensor large enough to slice every pin
    from, so pinning tens of GiB costs neither RAM nor commit charge."""
    if os.name == "nt":
        # NTFS files are not sparse unless flagged, so this would allocate
        # every byte on disk
        pytest.skip("needs a filesystem that creates sparse files by default")
    path = tmp_path / "sparse.bin"
    size = MAX_CHUNKS * CHUNK
    with open(path, "wb") as f:
        f.truncate(size)
    return torch.from_file(str(path), shared=True, size=size, dtype=torch.uint8)


@pytest.fixture
def host(tmp_path, monkeypatch):
    cgroupfs = tmp_path / "sys_fs_cgroup"
    cgroupfs.mkdir()
    fake = FakeHost(cgroupfs)
    monkeypatch.setattr(system_memory, "CGROUP_V2_ROOT", str(cgroupfs))
    monkeypatch.setattr(system_memory, "CGROUP_V1_MEMORY_ROOT", str(cgroupfs / "memory"))
    monkeypatch.setattr(system_memory, "PROC_SELF_CGROUP", str(tmp_path / "proc_self_cgroup"))
    write(tmp_path / "proc_self_cgroup", "0::/\n")
    monkeypatch.setattr(system_memory, "_cgroup_dirs", None)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(model_management, "WINDOWS", False)
    monkeypatch.setattr(system_memory.psutil, "virtual_memory", fake.virtual_memory)

    cudart = FakeCudart()
    monkeypatch.setattr(torch.cuda, "cudart", lambda: cudart)
    monkeypatch.setattr(model_management, "PINNED_MEMORY", {})
    monkeypatch.setattr(model_management, "TOTAL_PINNED_MEMORY", 0)
    # the ceiling the unfixed code computed on appmana-001; the live budget
    # must hold even under a permissive ceiling
    monkeypatch.setattr(model_management, "MAX_PINNED_MEMORY", 112341 * MIB)
    monkeypatch.setattr(model_management, "current_loaded_models", [])
    monkeypatch.setattr(memory_management, "extra_ram_release_callback", None)
    monkeypatch.setattr(memory_management, "RAM_CACHE_HEADROOM", 0)
    fake.cudart = cudart
    with context_configuration(default_configuration()):
        yield fake


def reserve_for(limit):
    return max(4 * GIB, limit // 10)


def legacy_pins(host):
    """Pin 1 GiB chunks through model_management.pin_memory until refused."""
    pinned = 0
    for i in range(MAX_CHUNKS):
        available_before = system_memory.virtual_memory_available()
        if not model_management.pin_memory(host.sparse[i * CHUNK:(i + 1) * CHUNK]):
            break
        assert CHUNK + host.expected_reserve <= available_before
        pinned += CHUNK
        host.sync()
    return pinned


def module_with_pin_state():
    state = {subset: (FakeHostBuffer(), [], [-1], [0], [0], {}) for subset in
             ("weights", "patches", "weights-loaded", "patches-loaded", "weights-fast", "patches-fast")}
    state["writable"] = False
    return SimpleNamespace(_pin_state=state)


def dynamic_pins(host):
    """Pin 1 GiB modules through the dynamic VRAM host-buffer path until refused."""
    pinned = 0
    for _ in range(MAX_CHUNKS):
        available_before = system_memory.virtual_memory_available()
        before = model_management.TOTAL_PINNED_MEMORY
        pinned_memory.pin_memory(module_with_pin_state(), subset="weights", size=CHUNK)
        if model_management.TOTAL_PINNED_MEMORY == before:
            break
        assert CHUNK + host.expected_reserve <= available_before
        pinned += CHUNK
        host.sync()
    return pinned


PIN_PATHS = [pytest.param(legacy_pins, id="legacy"), pytest.param(dynamic_pins, id="dynamic_vram")]


class TestReserveAndCeiling:
    @pytest.mark.parametrize("limit, reserve", [
        (HOST_TOTAL, int(12.5 * GIB)),
        (64 * GIB, int(6.4 * GIB)),
        (32 * GIB, 4 * GIB),
        (16 * GIB, 4 * GIB),
    ])
    def test_default_reserve_is_a_tenth_of_the_limit_but_at_least_4_gib(self, host, limit, reserve):
        host.total = limit
        assert model_management.pinned_memory_reserve() == reserve

    def test_ceiling_on_the_host_is_total_minus_reserve(self, host):
        assert model_management.compute_max_pinned_memory() == HOST_TOTAL - reserve_for(HOST_TOTAL)

    def test_ceiling_follows_cgroup_v2_memory_max(self, host):
        host.v2_cgroup("/kubepods.slice/pod/ctr", memory_max=32 * GIB, memory_current=0)
        assert model_management.compute_max_pinned_memory() == 28 * GIB

    def test_ceiling_follows_cgroup_v2_memory_max_of_an_ancestor(self, host):
        host.v2_cgroup("/pod/ctr", memory_max="max", memory_current=0)
        write(host.root / "pod" / "memory.max", f"{24 * GIB}\n")
        assert model_management.compute_max_pinned_memory() == 20 * GIB

    def test_ceiling_follows_cgroup_v2_memory_high(self, host):
        host.v2_cgroup("/user.slice/run-1.scope", memory_max="max", memory_high=16 * GIB, memory_current=0)
        assert model_management.compute_max_pinned_memory() == 12 * GIB

    def test_ceiling_follows_cgroup_v1_limit_in_bytes(self, host):
        host.v1_cgroup("/docker/abc", limit=24 * GIB)
        assert model_management.compute_max_pinned_memory() == 20 * GIB

    def test_windows_ceiling_is_capped_at_40_percent(self, host, monkeypatch):
        monkeypatch.setattr(model_management, "WINDOWS", True)
        assert model_management.compute_max_pinned_memory() == int(HOST_TOTAL * 0.40)

    def test_overrides(self, host):
        configuration = default_configuration()
        configuration.pinned_memory_reserve = 20.0
        with context_configuration(configuration):
            assert model_management.pinned_memory_reserve() == 20 * GIB
            assert model_management.compute_max_pinned_memory() == HOST_TOTAL - 20 * GIB
        configuration = default_configuration()
        configuration.max_pinned_memory = 8.0
        with context_configuration(configuration):
            assert model_management.compute_max_pinned_memory() == 8 * GIB

    def test_max_override_never_exceeds_the_effective_limit(self, host):
        host.v2_cgroup("/pod", memory_max=32 * GIB, memory_current=0)
        configuration = default_configuration()
        configuration.max_pinned_memory = 64.0
        with context_configuration(configuration):
            assert model_management.compute_max_pinned_memory() == 28 * GIB


class TestPinTimeBudget:
    @pytest.fixture(autouse=True)
    def pinnable(self, host, sparse_tensor, monkeypatch):
        monkeypatch.setattr(pinned_memory.comfy_aimdo.torch, "hostbuf_to_tensor", lambda hostbuf: sparse_tensor)
        host.sparse = sparse_tensor

    @pytest.mark.parametrize("pin", PIN_PATHS)
    def test_memory_used_by_other_processes_is_respected(self, host, pin):
        host.other = VMS
        host.expected_reserve = reserve_for(HOST_TOTAL)
        pinned = pin(host)
        assert pinned > 0
        assert pinned <= HOST_TOTAL - VMS - reserve_for(HOST_TOTAL)
        assert host.available() >= reserve_for(HOST_TOTAL)

    @pytest.mark.parametrize("pin", PIN_PATHS)
    def test_cgroup_memory_max_minus_current_is_respected(self, host, pin):
        # a 32 GiB pod on a host with plenty free: the pod limit binds
        host.own = 10 * GIB
        host.v2_cgroup("/kubepods.slice/pod/ctr", memory_max=32 * GIB)
        host.expected_reserve = reserve_for(32 * GIB)
        pinned = pin(host)
        assert pinned > 0
        assert 10 * GIB + pinned <= 32 * GIB - reserve_for(32 * GIB)

    @pytest.mark.parametrize("pin", PIN_PATHS)
    def test_cgroup_v1_limit_is_respected(self, host, pin):
        host.own = 4 * GIB
        host.v1_cgroup("/docker/abc", limit=24 * GIB)
        host.expected_reserve = reserve_for(24 * GIB)
        pinned = pin(host)
        assert pinned > 0
        assert 4 * GIB + pinned <= 24 * GIB - reserve_for(24 * GIB)

    def test_both_paths_share_one_budget(self, host):
        host.other = VMS
        host.expected_reserve = reserve_for(HOST_TOTAL)
        for i in range(30):
            assert model_management.pin_memory(host.sparse[i * CHUNK:(i + 1) * CHUNK])
            host.sync()
        dynamic = dynamic_pins(host)
        assert 30 * GIB + dynamic <= HOST_TOTAL - VMS - reserve_for(HOST_TOTAL)
        assert model_management.TOTAL_PINNED_MEMORY == 30 * GIB + dynamic

    @pytest.mark.parametrize("pin", PIN_PATHS)
    def test_availability_is_rechecked_at_pin_time(self, host, pin):
        host.expected_reserve = reserve_for(HOST_TOTAL)
        host.other = 100 * GIB
        first = pin(host)
        assert first == (HOST_TOTAL - 100 * GIB - reserve_for(HOST_TOTAL)) // CHUNK * CHUNK
        # a VM stops: memory returns, and pinning may continue
        host.other = VMS
        second = pin(host)
        assert second > 0
        assert first + second <= HOST_TOTAL - VMS - reserve_for(HOST_TOTAL)
        # a VM starts: nothing more is pinned
        host.other = 110 * GIB
        assert pin(host) == 0

    @pytest.mark.parametrize("pin", PIN_PATHS)
    def test_high_ram_does_not_bypass_the_budget(self, host, pin):
        configuration = default_configuration()
        configuration.high_ram = True
        host.other = VMS
        host.expected_reserve = reserve_for(HOST_TOTAL)
        with context_configuration(configuration):
            pinned = pin(host)
        assert 0 < pinned <= HOST_TOTAL - VMS - reserve_for(HOST_TOTAL)

    def test_ensure_pin_budget_uses_the_reserve(self, host):
        host.other = VMS
        available = HOST_TOTAL - VMS
        reserve = reserve_for(HOST_TOTAL)
        assert model_management.ensure_pin_budget(available - reserve)
        assert not model_management.ensure_pin_budget(available - reserve + 1)

"""Pinned host memory stays within budget inside a memory-limited cgroup.

The test runs a child process in a transient systemd scope with MemoryMax set
(the same cgroup v2 memory.max a Kubernetes pod limit becomes). Inside it:

* another process holds anonymous memory, standing in for the other workloads
  (VMs, other containers of the pod) the budget must not ignore;
* a GPU ballast leaves too little VRAM for the model, so its weights must be
  offloaded and pinned in host memory;
* the Qwen-Image 2.1 DiT (14.2 GB) is loaded through the dynamic VRAM path or
  the legacy ModelPatcher.

Every pin that registers new host memory is recorded along with the memory
left for the cgroup right after it, measured independently of the code under
test: min(MemAvailable, memory.max - (memory.current - inactive_file)). The
reserve is the documented default, max(4 GiB, memory.max / 10).
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

GIB = 1024 ** 3
# (memory.max, memory held by the other process). The other process starts
# after the weights are loaded and leaves less room than the model would pin
# (dynamic: ~6.8 GiB of streamed weights; legacy: ~2.1 GiB of offloaded ones),
# so the pins must stop at the reserve. Loading peaks higher on the legacy
# patcher, so its cgroup is larger.
SCENARIOS = {
    "dynamic": (24 * GIB, 13 * GIB),
    "legacy": (32 * GIB, 23 * GIB),
}
FREE_VRAM_FOR_MODEL = 6 * GIB
MODEL = "qwen_image_2.1_int8_convrot.safetensors"

BALLAST = (
    "import sys, numpy\n"
    "held = numpy.ones(int(sys.argv[1]), dtype=numpy.uint8)\n"
    "print('ready', flush=True)\n"
    "sys.stdin.read()\n"
)


def expected_reserve(limit):
    return max(4 * GIB, limit // 10)


def _own_cgroup():
    with open("/proc/self/cgroup", encoding="utf-8") as f:
        for line in f:
            hierarchy, controllers, path = line.rstrip("\n").split(":", 2)
            if hierarchy == "0" and controllers == "":
                return "/sys/fs/cgroup" + path
    raise RuntimeError("not on the cgroup v2 unified hierarchy")


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _keyed(path, key):
    for line in _read(path).splitlines():
        name, value = line.split()[:2]
        if name.rstrip(":") == key:
            return int(value)
    raise KeyError(key)


def _headroom(cgroup, limit):
    mem_available = _keyed("/proc/meminfo", "MemAvailable") * 1024
    working_set = int(_read(os.path.join(cgroup, "memory.current"))) - _keyed(os.path.join(cgroup, "memory.stat"), "inactive_file")
    return min(mem_available, limit - working_set)


def _child(mode, out_path):
    cgroup = _own_cgroup()
    limit = int(_read(os.path.join(cgroup, "memory.max")))
    assert limit == SCENARIOS[mode][0]

    import torch
    from comfy.cli_args import args
    # on a fast NVMe dynamic VRAM streams weights from disk instead of pinning
    # them; the host that livelocked took the pinned path
    args.disable_fast_disk = True
    from comfy import memory_management
    from comfy import model_management
    from comfy import pinned_memory
    from comfy import sd
    from comfy.model_downloader import get_or_download

    free, _ = torch.cuda.mem_get_info()
    vram_ballast = torch.empty(max(0, free - FREE_VRAM_FOR_MODEL), dtype=torch.uint8, device=model_management.get_torch_device())
    if mode == "dynamic":
        from comfy import aimdo_integration  # noqa: F401  (what setup_post_torch does)
        assert memory_management.aimdo_enabled

    pins = []
    peak = [0]

    def record(size):
        peak[0] = max(peak[0], model_management.TOTAL_PINNED_MEMORY)
        pins.append({"size": size, "headroom_after": _headroom(cgroup, limit), "total_pinned": model_management.TOTAL_PINNED_MEMORY})

    dynamic_pin = pinned_memory.pin_memory

    def pin_host_buffer(module, subset="weights", size=None):
        stack = module._pin_state[subset][1]
        before = len(stack)
        result = dynamic_pin(module, subset=subset, size=size)
        if len(stack) > before:
            record(module._pins[subset]["pin"].nbytes)
        return result

    legacy_pin = model_management.pin_memory

    def pin_tensor(tensor, evict_active=True):
        result = legacy_pin(tensor, evict_active=evict_active)
        if result:
            record(tensor.nbytes)
        return result

    pinned_memory.pin_memory = pin_host_buffer
    model_management.pin_memory = pin_tensor

    path = get_or_download("diffusion_models", MODEL)
    patcher = sd.load_diffusion_model(path, disable_dynamic=mode == "legacy")
    # the other workload grows after the weights are loaded and before they
    # are placed and pinned, as a VM or a sibling container would
    other = subprocess.Popen([sys.executable, "-c", BALLAST, str(SCENARIOS[mode][1])], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert other.stdout.readline().strip() == "ready"
    model_management.load_models_gpu([patcher])
    # dynamic VRAM pins host buffers as weights stream through a forward pass
    device = model_management.get_torch_device()
    with torch.inference_mode():
        for _ in range(2):
            patcher.model.apply_model(
                torch.randn(1, 64, 16, 16, device=device),
                torch.tensor([0.5], device=device),
                c_crossattn=torch.randn(1, 16, 4096, device=device),
            )
    torch.cuda.synchronize()

    events = {line.split()[0]: int(line.split()[1]) for line in _read(os.path.join(cgroup, "memory.events")).splitlines()}
    result = {
        "mode": mode,
        "is_dynamic": patcher.is_dynamic(),
        "memory_max": limit,
        "memory_peak": int(_read(os.path.join(cgroup, "memory.peak"))),
        "events": events,
        "max_pinned_memory": model_management.MAX_PINNED_MEMORY,
        "peak_pinned": peak[0],
        "pins": pins,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    del vram_ballast
    other.stdin.close()
    other.wait()


def _systemd_scope_available():
    if not sys.platform.startswith("linux") or shutil.which("systemd-run") is None:
        return False
    probe = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=1G", "true"], capture_output=True)
    return probe.returncode == 0


@pytest.mark.inference
@pytest.mark.requires_weights
@pytest.mark.requires_accelerator
@pytest.mark.parametrize("mode", ["dynamic", "legacy"])
def test_pins_stay_within_budget_under_cgroup_memory_max(mode, tmp_path, has_gpu):
    import torch
    if not has_gpu or not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    if not _systemd_scope_available():
        pytest.skip("requires systemd-run --user --scope with memory delegation")
    out = tmp_path / "result.json"
    command = [
        "systemd-run", "--user", "--scope", "--quiet",
        "-p", f"MemoryMax={SCENARIOS[mode][0]}", "-p", "MemorySwapMax=0",
        sys.executable, os.path.abspath(__file__), mode, str(out),
    ]
    completed = subprocess.run(command, env=dict(os.environ), timeout=1800)
    assert completed.returncode == 0, "the child died: killed by the cgroup OOM killer or failed to load"
    result = json.loads(out.read_text())
    reserve = expected_reserve(result["memory_max"])

    # the process never exceeded memory.max
    assert result["events"]["oom"] == 0
    assert result["events"]["oom_kill"] == 0
    assert result["memory_peak"] <= result["memory_max"]
    # the ceiling and the peak of registered bytes respect the cgroup limit
    assert result["max_pinned_memory"] <= result["memory_max"] - reserve
    assert result["peak_pinned"] <= result["memory_max"] - reserve
    # no pin ate into the reserve
    violations = [pin for pin in result["pins"] if pin["headroom_after"] < reserve]
    assert violations == [], f"{len(violations)} of {len(result['pins'])} pins left less than the {reserve} byte reserve; lowest {min(p['headroom_after'] for p in violations)}"
    if mode == "dynamic":
        assert result["is_dynamic"]
        # non-vacuous: the model did not fit, so host buffers were pinned
        assert result["pins"], "nothing was pinned; the scenario did not exercise the budget"


if __name__ == "__main__":
    _child(sys.argv[1], sys.argv[2])

import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from comfy import model_management
from comfy.model_patcher import ModelPatcherDynamic


def test_failed_truncate_keeps_pin_retryable(monkeypatch):
    tensor = torch.ones(4)
    entry = [0, 0, object()]
    pin = {"pin": tensor, "registered": True, "balancer_entry": entry}
    module = SimpleNamespace(_pins={"weights": pin})
    host = Mock()
    host.truncate.side_effect = [RuntimeError("HostBuffer.truncate failed"), None]
    stack = [(module, 0)]
    size = tensor.nbytes
    subset = (host, stack, [0], [size])
    patcher = SimpleNamespace(model=SimpleNamespace(dynamic_pins={"cpu": {"weights": subset}}), load_device="cpu")
    monkeypatch.setattr(model_management, "TOTAL_PINNED_MEMORY", size)

    with pytest.raises(RuntimeError, match="HostBuffer.truncate failed"):
        ModelPatcherDynamic.partially_unload_ram(patcher, size, ["weights"])
    assert stack == [(module, 0)] and pin["pin"] is tensor and entry[-1] is not None
    assert ModelPatcherDynamic.partially_unload_ram(patcher, size, ["weights"]) == size
    assert not stack and "pin" not in pin and entry[-1] is None
    assert subset[3] == [0] and model_management.TOTAL_PINNED_MEMORY == 0


def test_cached_native_buffer_released_before_library_teardown():
    code = textwrap.dedent("""
        import atexit
        from types import SimpleNamespace
        from comfy.cli_args import args
        args.cpu = True
        import comfy_aimdo.vram_buffer as native

        released = []
        def unload_library():
            native.lib = None
            print("released", released)
        atexit.register(unload_library)

        from comfy import model_management
        native.lib = SimpleNamespace(vrambuf_destroy=lambda ptr: released.append(ptr) or True)
        buffer = native.VRAMBuffer.__new__(native.VRAMBuffer)
        buffer._ptr = 42
        model_management.STREAM_AIMDO_CAST_BUFFERS[None] = buffer
        del buffer
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert "released [42]" in result.stdout, (result.stdout, result.stderr)
    assert "Exception ignored" not in result.stderr


def test_failed_final_unpin_still_detaches_once(caplog):
    patcher = SimpleNamespace(
        model=SimpleNamespace(dynamic_patchers=set()),
        unpin_all_weights=Mock(side_effect=RuntimeError("HostBuffer.truncate failed")),
        detach=Mock(),
    )
    patcher.model.dynamic_patchers.add(id(patcher))
    ModelPatcherDynamic.__del__(patcher)
    ModelPatcherDynamic.__del__(patcher)
    patcher.unpin_all_weights.assert_called_once_with()
    patcher.detach.assert_called_once_with(unpatch_all=False)
    assert "HostBuffer.truncate failed" in caplog.text

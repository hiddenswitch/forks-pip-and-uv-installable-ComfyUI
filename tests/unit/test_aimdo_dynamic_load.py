"""Regression tests for dynamic VRAM (comfy-aimdo) weight loading.

The dynamic patcher (ModelPatcherDynamic) loads weights with
``load_state_dict(assign=True)`` so the vbar streamer can take ownership of the
tensors. ``assign=True`` *replaces* each param with the incoming tensor instead
of ``copy_``-casting it into the fp8 placeholder. For a mixed-precision
checkpoint (e.g. ``flux2_dev_fp8mixed``, whose ``_quantization_metadata`` only
lists the fp8 layers) this **preserves** the higher-precision (bf16) layers the
author stored for quality — which is the whole point of mixed precision. These
tests pin that: dynamic load must NOT coerce a bf16 weight down to the model's
fp8 storage dtype, and quantized/fp8-stored weights must keep their native
layout across the vbar so their scale survives.
"""
import contextlib
import os
import types
import unittest
from unittest import mock

import torch

import comfy.memory_management as memory_management
import comfy.model_base as model_base
import comfy.model_management as model_management
import comfy.ops as ops
from comfy.model_patcher import ModelPatcher


def _fake_basemodel_self(diffusion_model):
    """Minimal stand-in exposing only what BaseModel.load_model_weights touches."""
    cfg = types.SimpleNamespace(process_unet_state_dict=lambda sd: sd)
    return types.SimpleNamespace(
        diffusion_model=diffusion_model,
        model_config=cfg,
        get_dtype=lambda: torch.float8_e4m3fn,
        manual_cast_dtype=torch.bfloat16,
    )


def _tiny_fp8_linear():
    """A Linear whose weight param is an fp8 placeholder, as an fp8 model builds."""
    layer = torch.nn.Linear(10, 20, bias=False)
    layer.weight = torch.nn.Parameter(
        torch.zeros(20, 10, dtype=torch.float8_e4m3fn), requires_grad=False
    )
    return layer


@contextlib.contextmanager
def limit_free_vram(free_bytes):
    """Force the dynamic patcher's resident-budget check to see a small amount of
    free VRAM, so a model that would otherwise stay resident is streamed instead.

    This is the unit-test analogue of the runtime levers for capping reported
    VRAM (NVIDIA MPS ``CUDA_MPS_PINNED_DEVICE_MEM_LIMIT``, or a ballast tensor):
    ModelPatcherDynamic.load() decides resident-vs-stream from
    ``model_management.get_free_memory``, so patching it deterministically
    exercises the vbar streaming path without needing a 24 GB+ model.
    """
    def fake_get_free_memory(dev=None, torch_free_too=False):
        return (free_bytes, free_bytes) if torch_free_too else free_bytes

    with mock.patch.object(model_management, "get_free_memory", side_effect=fake_get_free_memory):
        yield


class TestMixedPrecisionStoragePreserved(unittest.TestCase):
    """A mixed checkpoint deliberately keeps some layers in higher precision
    (bf16) for quality; loading them into the model's fp8 storage placeholder
    must not silently down-convert them. This must hold for BOTH the legacy
    ``copy_`` path (assign=False) and the dynamic VRAM ``assign`` path
    (assign=True). Layers stored at equal-or-lower precision still load normally
    (so an fp8 layer upcasts to bf16 when the model dtype is bf16)."""

    def _load(self, model, src, assign):
        model_base.BaseModel.load_model_weights(
            _fake_basemodel_self(model), {"weight": src.clone()}, assign=assign
        )

    def test_higher_precision_bf16_layer_preserved_both_paths(self):
        src = torch.randn(20, 10, dtype=torch.bfloat16)
        for assign in (True, False):
            with self.subTest(assign=assign):
                model = _tiny_fp8_linear()
                self._load(model, src, assign)
                self.assertEqual(model.weight.dtype, torch.bfloat16)
                torch.testing.assert_close(model.weight, src)

    def test_fp8_stored_layer_stays_fp8_both_paths(self):
        src = torch.randn(20, 10).to(torch.float8_e4m3fn)
        for assign in (True, False):
            with self.subTest(assign=assign):
                model = _tiny_fp8_linear()
                self._load(model, src, assign)
                self.assertEqual(model.weight.dtype, torch.float8_e4m3fn)
                torch.testing.assert_close(model.weight.float(), src.float())

    def test_lower_precision_weight_upcasts_to_param_dtype(self):
        """fp8-stored weight into a bf16 placeholder (e.g. --fp8_storage off):
        not retargeted, so it upcasts to bf16 — never preserve a *lower*
        precision than the model asked for."""
        layer = torch.nn.Linear(10, 20, bias=False).to(torch.bfloat16)
        src = torch.randn(20, 10).to(torch.float8_e4m3fn)
        for assign in (False,):  # copy_ path upcasts; assign would replace as fp8
            with self.subTest(assign=assign):
                self._load(layer, src, assign)
                self.assertEqual(layer.weight.dtype, torch.bfloat16)

    def test_matching_dtype_untouched_both_paths(self):
        src = torch.randn(20, 10, dtype=torch.bfloat16)
        for assign in (True, False):
            with self.subTest(assign=assign):
                layer = torch.nn.Linear(10, 20, bias=False).to(torch.bfloat16)
                self._load(layer, src, assign)
                self.assertEqual(layer.weight.dtype, torch.bfloat16)
                torch.testing.assert_close(layer.weight, src)


class TestStreamsInNativeDtype(unittest.TestCase):
    """The vbar must stream scale-carrying weights in their native low-precision
    layout, not densely materialize them mid-stream (which drops the per-tensor
    scale and diverges from the resident result)."""

    def test_plain_fp8_weight_streams_native(self):
        self.assertTrue(ops._streams_in_native_dtype(torch.zeros(4, 4, dtype=torch.float8_e4m3fn)))

    def test_bf16_weight_does_not_stream_native(self):
        self.assertFalse(ops._streams_in_native_dtype(torch.zeros(4, 4, dtype=torch.bfloat16)))

    def test_fp32_weight_does_not_stream_native(self):
        self.assertFalse(ops._streams_in_native_dtype(torch.zeros(4, 4, dtype=torch.float32)))

    def test_none_is_not_native(self):
        self.assertFalse(ops._streams_in_native_dtype(None))

    def test_vbar_cast_does_not_treat_empty_prefetch_slot_as_prefetched(self):
        layer = ops.manual_cast.Linear(4, 4, bias=False, device="cpu")
        layer._v = object()
        self.assertIsNone(layer._prefetch)

        def prepare_prefetch(modules, *args, **kwargs):
            modules[0]._prefetch = {"signature": None}
            return None

        with (
            mock.patch.object(ops, "cast_modules_with_vbar", side_effect=prepare_prefetch) as cast,
            mock.patch.object(ops, "resolve_cast_module_with_vbar", return_value=(layer.weight, None)),
            mock.patch.object(model_management, "device_supports_non_blocking", return_value=False),
        ):
            weight, bias, _ = ops.cast_bias_weight(
                layer,
                dtype=layer.weight.dtype,
                device=torch.device("cuda:0"),
                bias_dtype=layer.weight.dtype,
                offloadable=True,
            )

        cast.assert_called_once()
        self.assertIs(weight, layer.weight)
        self.assertIsNone(bias)
        self.assertIsNone(layer._prefetch)

    @unittest.skipUnless(
        ops.mixed_precision_quantization_available(),
        "requires comfy_kitchen-backed quantized tensors",
    )
    def test_quantized_tensor_streams_native(self):
        from comfy.quant_ops import QuantizedTensor

        qt = QuantizedTensor.from_float(
            torch.randn(20, 10, dtype=torch.bfloat16),
            "TensorCoreFP8E4M3Layout",
            scale="recalculate",
        )
        self.assertTrue(ops._streams_in_native_dtype(qt))

    @unittest.skipUnless(
        ops.int8_quantization_available(),
        "requires comfy_kitchen-backed INT8 tensors",
    )
    def test_int8_loading_list_uses_physical_not_logical_weight_size(self):
        from comfy.quant_ops import QuantizedTensor

        model = torch.nn.Module()
        model.layer = ops.manual_cast.Linear(
            16, 16, bias=False, dtype=torch.bfloat16, device="cpu"
        )
        weight = QuantizedTensor.from_float(
            torch.randn(16, 16, dtype=torch.bfloat16),
            "TensorWiseINT8Layout",
            scale="recalculate",
        )
        model.layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        model.manual_cast_dtype = torch.bfloat16
        patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))

        [item] = patcher._load_list(for_dynamic=True)

        self.assertEqual(item.module_size, weight.nbytes)
        self.assertEqual(item.module_offload_mem, weight.nbytes)
        self.assertLess(weight.nbytes, weight.numel() * weight.element_size())


def _aimdo_runtime_available():
    try:
        import comfy.aimdo_integration  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available() and memory_management.aimdo_enabled


@unittest.skipUnless(
    os.environ.get("COMFY_TEST_FP8_MODEL") and _aimdo_runtime_available(),
    "set COMFY_TEST_FP8_MODEL to an fp8 diffusion checkpoint and run on a CUDA "
    "box with comfy-aimdo to exercise end-to-end dynamic-VRAM streaming quality",
)
class TestDynamicStreamingQuality(unittest.TestCase):
    """End-to-end quality check: a model forced to stream under dynamic VRAM
    must produce the same forward output as the resident reference. The
    ``limit_free_vram`` helper provides the deterministic low-VRAM condition that
    drives the vbar streaming path. Catches both the load-dtype regression and
    any scale corruption in the streaming geometry."""

    def _run(self, seed=7):
        import comfy.sd as sd

        path = os.environ["COMFY_TEST_FP8_MODEL"]
        patcher = sd.load_diffusion_model(path)
        model_management.load_models_gpu([patcher], force_full_load=True)
        model = patcher.model
        device = model_management.get_torch_device()
        cfg = model.model_config.unet_config
        torch.manual_seed(seed)
        x = torch.randn(1, cfg.get("in_channels", 128), 32, 32, device=device, dtype=torch.bfloat16)
        t = torch.tensor([1.0], device=device)
        ctx = torch.randn(1, 64, cfg.get("context_in_dim", 4096), device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model.apply_model(x, t, c_crossattn=ctx)
        return out.float().cpu()

    def test_streamed_matches_resident(self):
        resident = self._run()
        # ~256 MB of free VRAM forces the dynamic patcher to stream every block.
        with limit_free_vram(256 * 1024 * 1024):
            streamed = self._run()
        torch.testing.assert_close(streamed, resident, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()


class TestBakedPatchesSurviveReload(unittest.TestCase):
    """A patch baked into a quantized weight (dequantize, patch, requantize) stays baked across
    dynamic reloads under the same patches; a reload under other patches restores the original."""

    def _patcher(self, uuid):
        from comfy.model_patcher import ModelPatcherDynamic
        model = torch.nn.Module()
        model.lin = torch.nn.Linear(2, 2)
        model.model_loaded_weight_memory = 0
        model.current_weight_patches_uuid = "lora"
        model.dynamic_baked_keys = {"lin.weight"}
        original = model.lin.weight.detach().clone()
        baked = torch.nn.Parameter(original + 1, requires_grad=False)
        model.lin.weight = baked
        patcher = types.SimpleNamespace(model=model, patches_uuid=uuid, backup={
            "lin.weight": types.SimpleNamespace(weight=original, inplace_update=False)}, backup_buffers={})
        return ModelPatcherDynamic.restore_loaded_backups, patcher, original, baked

    def test_a_reload_with_the_same_patches_keeps_the_baked_weight(self):
        restore, patcher, original, baked = self._patcher("lora")
        restore(patcher, keep_baked=True)
        self.assertIs(patcher.model.lin.weight, baked)
        self.assertEqual(patcher.model.dynamic_baked_keys, {"lin.weight"})
        self.assertIn("lin.weight", patcher.backup)

    def test_other_patches_or_an_unload_restore_the_original(self):
        for uuid, keep in (("base", True), ("lora", False)):
            restore, patcher, original, baked = self._patcher(uuid)
            restore(patcher, keep_baked=keep)
            torch.testing.assert_close(patcher.model.lin.weight, original)
            self.assertEqual(patcher.model.dynamic_baked_keys, set())
            self.assertEqual(patcher.backup, {})


class TestResidentReloadSkipsStaging(unittest.TestCase):
    """load_models_gpu reloads every model of every prompt. A dynamic model still resident on its device
    under the same patches is not staged again (the module pass of load); it is only made active again.
    Anything that changed what load set up stages it again."""

    def _patcher(self):
        from comfy.model_patcher import ModelPatcherDynamic
        model = torch.nn.Module()
        model.lin = torch.nn.Linear(4, 4)
        patcher = ModelPatcherDynamic(model, torch.device("cuda", 0), torch.device("cpu"))
        calls = []

        def fake_load(self, device_to, dirty=False):
            calls.append(dirty)
            self.model.device = device_to
            self.model.current_weight_patches_uuid = self.patches_uuid
            self.model.model_loaded_weight_memory = 1024
            pin_state = self.model.dynamic_pins[self.load_device]
            pin_state["active"], pin_state["failed"] = True, False

        return patcher, calls, mock.patch.object(ModelPatcherDynamic, "load", fake_load)

    def test_a_resident_model_is_activated_not_staged(self):
        patcher, calls, fake = self._patcher()
        device = torch.device("cuda", 0)
        with fake:
            patcher.partially_load(device)
            pin_state = patcher.model.dynamic_pins[device]
            pin_state["active"], pin_state["failed"] = False, True  # reset_cast_buffers after the prompt
            patcher.partially_load(device)
        self.assertEqual(calls, [False])
        self.assertTrue(pin_state["active"])
        self.assertFalse(pin_state["failed"])

    def test_changed_patches_or_unloaded_weights_stage_again(self):
        device = torch.device("cuda", 0)
        for change in ("patches", "unloaded", "hooks"):
            patcher, calls, fake = self._patcher()
            with fake:
                patcher.partially_load(device)
                if change == "patches":
                    patcher.patches_uuid = "another"
                elif change == "unloaded":
                    patcher.model.model_loaded_weight_memory = 0  # restore_loaded_backups under pressure
                else:
                    patcher.forced_hooks = mock.MagicMock()
                patcher.partially_load(device)
            self.assertEqual(len(calls), 2, change)


class TestPatchedWeightsStageTheirOwnSize(unittest.TestCase):
    """A LoRA-patched weight holds its own (unpatched) bytes in the VBAR, as an unpatched one does; the patch math's
    scratch is a transient budget (the materialization spec), not VBAR space. With the scratch in the VBAR a 14 GB
    Qwen Image 2.1 under a LoRA staged 40 GB on a 24 GB card and streamed every step."""

    def test_a_patched_weight_allocates_its_own_bytes(self):
        import comfy.ops as comfy_ops
        from comfy import memory_management as mm
        from comfy.model_patcher import ModelPatcherDynamic
        model = torch.nn.Module()
        model.lin = comfy_ops.manual_cast.Linear(128, 128, bias=False, dtype=torch.bfloat16)
        model.model_loaded_weight_memory = 0
        patcher = ModelPatcherDynamic(model, torch.device("cuda", 0), torch.device("cpu"))
        patcher.add_patches({"lin.weight": ("lora", (torch.zeros(128, 4), torch.zeros(4, 128), None, None))}, 1.0)
        allocs = []
        vbar = types.SimpleNamespace(alloc=lambda size: allocs.append(size) or ("vbar", 0, size), prioritize=lambda: None)
        with mock.patch.object(ModelPatcherDynamic, "_vbar_get", lambda self, create=False: vbar), \
                mock.patch("comfy.model_patcher.move_weight_functions", lambda m, device: 0):
            patcher.load(torch.device("cuda", 0))
        self.assertEqual(allocs, [mm.vram_aligned_size(model.lin.weight)])

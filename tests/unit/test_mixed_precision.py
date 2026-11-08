from types import SimpleNamespace
from unittest import mock
import json
import unittest
import unittest.mock

import torch

from comfy.cli_args_types import Configuration


from comfy import model_management
from comfy import ops
from comfy import utils
from comfy.model_base import _format_quantized_storage_summary
from comfy.quant_ops import QUANT_ALGOS
from comfy.quant_ops import QuantizedTensor
from comfy.supported_models import ZImage


class SimpleModel(torch.nn.Module):
    def __init__(self, operations=ops.disable_weight_init):
        super().__init__()
        self.layer1 = operations.Linear(10, 20, device="cpu", dtype=torch.bfloat16)
        self.layer2 = operations.Linear(20, 30, device="cpu", dtype=torch.bfloat16)
        self.layer3 = operations.Linear(30, 40, device="cpu", dtype=torch.bfloat16)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)
        x = self.layer3(x)
        return x


class TestMixedPrecisionOps(unittest.TestCase):
    def test_forced_fp16_z_image_weights_use_supported_compute_dtype(self):
        with (
            mock.patch.object(model_management, "args", Configuration(fp16_unet=True)),
            mock.patch.object(model_management, "should_use_fp16", return_value=True),
            mock.patch.object(model_management, "should_use_bf16", return_value=True),
        ):
            storage_dtype = model_management.unet_dtype(
                device=torch.device("cpu"), supported_dtypes=ZImage.supported_inference_dtypes,
            )
            compute_dtype = model_management.unet_manual_cast(
                storage_dtype, torch.device("cpu"), ZImage.supported_inference_dtypes,
            )
        self.assertEqual(storage_dtype, torch.float16)
        self.assertEqual(compute_dtype, torch.bfloat16)

        layer = ops.manual_cast.Linear(1, 1, bias=False, device="cpu", dtype=storage_dtype)
        layer.load_state_dict({"weight": torch.tensor([[100.0]], dtype=storage_dtype)})
        result = layer(torch.tensor([[1000.0]], dtype=compute_dtype))
        self.assertEqual(layer.weight.dtype, torch.float16)
        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(result.float(), torch.tensor([[100000.0]]), rtol=0.004, atol=0)

    def test_manual_cast_respects_model_and_device_dtype_support(self):
        cases = [
            (torch.float16, (torch.bfloat16, torch.float32), False, torch.float32),
            (torch.bfloat16, (torch.float32,), True, torch.float32),
            (torch.float16, (torch.float16, torch.float32), True, None),
            (torch.bfloat16, (torch.bfloat16, torch.float32), True, None),
        ]
        for storage, supported, bf16_available, expected in cases:
            with (
                self.subTest(storage=storage, supported=supported, bf16_available=bf16_available),
                mock.patch.object(model_management, "should_use_fp16", return_value=True),
                mock.patch.object(model_management, "should_use_bf16", return_value=bf16_available),
            ):
                self.assertEqual(
                    model_management.unet_manual_cast(storage, torch.device("cpu"), supported),
                    expected,
                )

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_quant_layout_class_fallbacks_exist(self):
        self.assertIsNotNone(ops.get_layout_class("TensorCoreFP8E4M3Layout"))
        self.assertIsNotNone(ops.get_layout_class("TensorCoreFP8E5M2Layout"))
        self.assertIsNotNone(ops.get_layout_class("TensorCoreNVFP4Layout"))

    def test_all_layers_standard(self):
        """Test that model with no quantization works normally"""
        # Create model
        model = SimpleModel(operations=ops.mixed_precision_ops({}))

        # Initialize weights manually
        model.layer1.weight = torch.nn.Parameter(torch.randn(20, 10, dtype=torch.bfloat16))
        model.layer1.bias = torch.nn.Parameter(torch.randn(20, dtype=torch.bfloat16))
        model.layer2.weight = torch.nn.Parameter(torch.randn(30, 20, dtype=torch.bfloat16))
        model.layer2.bias = torch.nn.Parameter(torch.randn(30, dtype=torch.bfloat16))
        model.layer3.weight = torch.nn.Parameter(torch.randn(40, 30, dtype=torch.bfloat16))
        model.layer3.bias = torch.nn.Parameter(torch.randn(40, dtype=torch.bfloat16))

        # Initialize weight_function and bias_function
        for layer in [model.layer1, model.layer2, model.layer3]:
            layer.weight_function = []
            layer.bias_function = []

        # Forward pass
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))
        self.assertEqual(output.dtype, torch.bfloat16)

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_mixed_precision_load(self):
        """Test loading a mixed precision model from state dict"""
        # Configure mixed precision: layer1 is FP8, layer2 and layer3 are standard
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            },
            "layer3": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        # Create state dict with mixed precision
        fp8_weight1 = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        fp8_weight3 = torch.randn(40, 30, dtype=torch.float32).to(torch.float8_e4m3fn)

        state_dict = {
            # Layer 1: FP8 E4M3FN
            "layer1.weight": fp8_weight1,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),

            # Layer 2: Standard BF16
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),

            # Layer 3: FP8 E4M3FN
            "layer3.weight": fp8_weight3,
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
            "layer3.weight_scale": torch.tensor(1.5, dtype=torch.float32),
        }

        state_dict, _ = utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})
        # Create model and load state dict (strict=False because custom loading pops keys)
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict, strict=False)

        # Verify weights are wrapped in QuantizedTensor
        self.assertIsInstance(model.layer1.weight, QuantizedTensor)
        self.assertEqual(model.layer1.weight._layout_cls, "TensorCoreFP8E4M3Layout")

        # Layer 2 should NOT be quantized
        self.assertNotIsInstance(model.layer2.weight, QuantizedTensor)

        # Layer 3 should be quantized
        self.assertIsInstance(model.layer3.weight, QuantizedTensor)
        self.assertEqual(model.layer3.weight._layout_cls, "TensorCoreFP8E4M3Layout")

        # Verify scales were loaded
        self.assertEqual(model.layer1.weight._params.scale.item(), 2.0)
        self.assertEqual(model.layer3.weight._params.scale.item(), 1.5)

        # Forward pass
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        with torch.inference_mode():
            output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_cast_bias_weight_preserves_quantized_weight_when_requant_requested(self):
        """Quantized matmul must not expand the full weight just because fp8 storage dtype differs from compute dtype."""
        layer = ops.mixed_precision_ops({}).Linear(10, 20, device="cpu", dtype=torch.bfloat16)
        layer.weight = QuantizedTensor.from_float(
            torch.randn(20, 10, dtype=torch.bfloat16),
            "TensorCoreFP8E4M3Layout",
            scale="recalculate",
        )
        layer.bias = torch.nn.Parameter(torch.randn(20, dtype=torch.bfloat16), requires_grad=False)
        layer.weight_function = []
        layer.bias_function = []
        input_tensor = QuantizedTensor.from_float(
            torch.randn(5, 10, dtype=torch.bfloat16),
            "TensorCoreFP8E4M3Layout",
            scale="recalculate",
        )

        weight, bias = ops.cast_bias_weight(layer, input_tensor, want_requant=True)

        self.assertIsInstance(weight, QuantizedTensor)
        self.assertEqual(weight._layout_cls, "TensorCoreFP8E4M3Layout")
        self.assertEqual(bias.dtype, torch.bfloat16)

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_convrot_int8_embedding_keeps_quantized_storage_for_row_dequantization(self):
        """ConvRot embeddings must reach the row-dequantization kernel as INT8, not compute dtype."""
        embedding = ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16).Embedding(
            8,
            256,
            device="cpu",
            dtype=torch.bfloat16,
        )
        quant_config = {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
        }
        state_dict = {
            "weight": torch.randint(-127, 128, (8, 256), dtype=torch.int8),
            "weight_scale": torch.ones(8, 1, dtype=torch.float32),
            "comfy_quant": torch.tensor(
                list(json.dumps(quant_config).encode("utf-8")),
                dtype=torch.uint8,
            ),
        }
        embedding.load_state_dict(state_dict, strict=True)

        self.assertIsInstance(embedding.weight, QuantizedTensor)
        output = embedding(torch.tensor([[0, 3, 7]], dtype=torch.long))

        self.assertEqual(embedding.weight._qdata.dtype, torch.int8)
        self.assertEqual(output.shape, (1, 3, 256))
        self.assertEqual(output.dtype, torch.bfloat16)

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_force_cast_does_not_disable_quantized_inference(self):
        """Manual cast flags should not turn fp8 mixed-precision inference into full-weight dequantization."""
        layer = ops.mixed_precision_ops({}).Linear(10, 20, device="cpu", dtype=torch.bfloat16)
        layer.quant_format = "float8_e4m3fn"
        layer.layout_type = "TensorCoreFP8E4M3Layout"
        layer.weight = torch.nn.Parameter(
            QuantizedTensor.from_float(
                torch.randn(20, 10, dtype=torch.bfloat16),
                "TensorCoreFP8E4M3Layout",
                scale="recalculate",
            ),
            requires_grad=False,
        )
        layer.bias = torch.nn.Parameter(torch.randn(20, dtype=torch.bfloat16), requires_grad=False)
        layer.comfy_force_cast_weights = True
        layer.weight_function = []
        layer.bias_function = []

        calls = []
        def wrapped_forward(input, compute_dtype=None, want_requant=False):
            calls.append((type(input), want_requant))
            return torch.empty(5, 20, dtype=torch.bfloat16)

        layer.forward_comfy_cast_weights = wrapped_forward
        with mock.patch("comfy.ops._quantized_layout_supports_fast_matmul", return_value=False):
            layer(torch.randn(5, 10, dtype=torch.bfloat16))

        self.assertTrue(calls)
        self.assertIs(calls[0][0], QuantizedTensor)
        self.assertTrue(calls[0][1])

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_vbar_auto_policy_keeps_quantized_weight_without_fast_matmul_probe(self):
        layer = ops.mixed_precision_ops({}).Linear(10, 20, device="cpu", dtype=torch.bfloat16)
        layer.quant_format = "float8_e4m3fn"
        layer.layout_type = "TensorCoreFP8E4M3Layout"
        layer.weight = torch.nn.Parameter(
            QuantizedTensor.from_float(
                torch.randn(20, 10, dtype=torch.bfloat16),
                "TensorCoreFP8E4M3Layout",
                scale="recalculate",
            ),
            requires_grad=False,
        )

        with mock.patch("comfy.ops._quantized_layout_supports_fast_matmul", return_value=False):
            self.assertTrue(ops.should_keep_quantized_vbar(layer, layer.weight))

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_quantized_lora_patch_bakes_back_into_weight(self):
        """LoRA-style patches on quantized layers should bake and requantize, not become runtime adapters."""
        from comfy.model_patcher import ModelPatcher
        from comfy.model_patcher import should_bake_lowvram_patch

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = ops.mixed_precision_ops({}).Linear(10, 20, device="cpu", dtype=torch.bfloat16)
                self.layer.quant_format = "float8_e4m3fn"
                self.layer.layout_type = "TensorCoreFP8E4M3Layout"
                self.layer.weight = torch.nn.Parameter(
                    QuantizedTensor.from_float(
                        torch.randn(20, 10, dtype=torch.bfloat16),
                        "TensorCoreFP8E4M3Layout",
                        scale="recalculate",
                    ),
                    requires_grad=False,
                )

        model = TinyModel()
        self.assertTrue(should_bake_lowvram_patch(model.layer, model.layer.weight))
        patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        patcher.add_patches({"layer.weight": ("diff", (torch.randn(20, 10, dtype=torch.bfloat16) * 0.01,))})

        patcher.patch_weight_to_device("layer.weight", device_to=torch.device("cpu"))

        self.assertIsInstance(model.layer.weight, QuantizedTensor)
        self.assertEqual(model.layer.weight._layout_cls, "TensorCoreFP8E4M3Layout")
        self.assertEqual(model.layer.weight_function, [])

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_lora_patch_on_int8_convrot_layer_changes_its_output_by_the_lora(self):
        """A LoRA on an INT8 ConvRot layer bakes into the requantized weight: the layer's output moves
        by the LoRA's x @ (up @ down)^T, to within one requantization of the weight."""
        from comfy.model_patcher import ModelPatcher

        torch.manual_seed(0)
        layer = ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16).Linear(256, 64, device="cpu", dtype=torch.bfloat16)
        layer.load_state_dict({
            "weight": torch.randint(-127, 128, (64, 256), dtype=torch.int8),
            "weight_scale": torch.full((64, 1), 0.01, dtype=torch.float32),
            "bias": torch.zeros(64, dtype=torch.bfloat16),
            "comfy_quant": torch.tensor(list(json.dumps(
                {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode("utf-8")), dtype=torch.uint8),
        }, strict=False)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = layer

        model = Model()
        x = torch.randn(8, 256, dtype=torch.bfloat16)
        with torch.inference_mode():
            before = model.layer(x).float()
        # a LoRA several quantization steps (0.01) large; one requantization is about half a step
        up = torch.randn(64, 4) * 0.2
        down = torch.randn(4, 256) * 0.2
        patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        from comfy import lora as comfy_lora
        patcher.add_patches(comfy_lora.load_lora({"layer.lora_up.weight": up, "layer.lora_down.weight": down},
                                                 {"layer": "layer.weight"}))

        patcher.patch_weight_to_device("layer.weight", device_to=torch.device("cpu"))

        self.assertIsInstance(model.layer.weight, QuantizedTensor)
        with torch.inference_mode():
            after = model.layer(x).float()
        lora = x.float() @ (up @ down).T
        self.assertLess(float((after - before - lora).norm() / lora.norm()), 0.1)

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_disabled_fp8_compute_preserves_scaled_quantized_weight(self):
        """Disabled fp8 kernels must still dequantize with the stored scale."""
        for quant_format, weight_dtype in (
            ("float8_e4m3fn", torch.float8_e4m3fn),
            ("float8_e5m2", torch.float8_e5m2),
        ):
            with self.subTest(quant_format=quant_format):
                layer_quant_config = {
                    "layer1": {
                        "format": quant_format,
                        "params": {}
                    }
                }
                fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(weight_dtype)
                state_dict = {
                    "layer1.weight": fp8_weight,
                    "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
                    "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
                    "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
                    "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
                    "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
                    "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
                }
                state_dict, _ = utils.convert_old_quants(
                    state_dict,
                    metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
                )

                model = SimpleModel(operations=ops.mixed_precision_ops({}, disabled={quant_format}))
                model.load_state_dict(state_dict, strict=False)

                self.assertTrue(model.layer1._full_precision_mm)
                self.assertIsInstance(model.layer1.weight, QuantizedTensor)
                self.assertEqual(model.layer1.weight._params.scale.item(), 2.0)
                self.assertEqual(model.layer1.weight._qdata.dtype, weight_dtype)

                input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
                with torch.inference_mode():
                    output = model(input_tensor)

                self.assertEqual(output.shape, (5, 40))

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_quantized_layer_keeps_fp8_resident_storage_with_bf16_compute(self):
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {},
            }
        }
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(1.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }
        state_dict, _ = utils.convert_old_quants(
            state_dict,
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )

        model = SimpleModel(operations=ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16, disabled={"float8_e4m3fn"}))
        model.load_state_dict(state_dict, strict=False)

        self.assertTrue(model.layer1._full_precision_mm)
        self.assertIsInstance(model.layer1.weight, QuantizedTensor)
        self.assertEqual(model.layer1.weight.dtype, torch.bfloat16)
        self.assertEqual(model.layer1.weight._qdata.dtype, torch.float8_e4m3fn)
        self.assertEqual(
            _format_quantized_storage_summary(model),
            "torch.float8_e4m3fn=1",
        )

        with torch.inference_mode():
            output = model(torch.randn(5, 10, dtype=torch.bfloat16))

        self.assertEqual(output.shape, (5, 40))
        self.assertEqual(output.dtype, torch.bfloat16)

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_quantized_layer_can_disable_fp8_resident_storage_for_benchmarking(self):
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {},
            }
        }
        state_dict = {
            "layer1.weight": torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn),
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(1.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }
        state_dict, _ = utils.convert_old_quants(
            state_dict,
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )

        model = SimpleModel(
            operations=ops.mixed_precision_ops(
                {},
                compute_dtype=torch.bfloat16,
                disabled_storage={"float8_e4m3fn"},
            )
        )
        model.load_state_dict(state_dict, strict=False)

        self.assertNotIsInstance(model.layer1.weight, QuantizedTensor)
        self.assertEqual(model.layer1.weight.dtype, torch.bfloat16)
        self.assertEqual(_format_quantized_storage_summary(model), "")

        with torch.inference_mode():
            output = model(torch.randn(5, 10, dtype=torch.bfloat16))

        self.assertEqual(output.shape, (5, 40))
        self.assertEqual(output.dtype, torch.bfloat16)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "requires torch fp8 dtype")
    def test_native_fp8_weight_storage_used_when_device_feature_test_passes(self):
        """Native fp8 checkpoints stay resident as fp8 when the device supports upcasted ops."""
        with (
            mock.patch.object(model_management, "args", Configuration(fp8_storage=True)),
            mock.patch.object(model_management, "supports_fp8_storage", return_value=True),
            mock.patch.object(model_management, "should_use_fp16", return_value=False),
            mock.patch.object(model_management, "should_use_bf16", return_value=True),
        ):
            selected_dtype = model_management.unet_dtype(
                device=torch.device("cpu"),
                model_params=10,
                supported_dtypes=(torch.bfloat16, torch.float32),
                weight_dtype=torch.float8_e4m3fn,
            )
            compute_dtype = model_management.unet_manual_cast(
                selected_dtype,
                torch.device("cpu"),
                supported_dtypes=(torch.bfloat16, torch.float32),
            )

        self.assertEqual(selected_dtype, torch.float8_e4m3fn)
        self.assertEqual(compute_dtype, torch.bfloat16)

        layer = ops.manual_cast.Linear(10, 20, device="cpu", dtype=selected_dtype)
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        fp8_bias = torch.randn(20, dtype=torch.float32).to(torch.float8_e4m3fn)
        layer.load_state_dict({"weight": fp8_weight, "bias": fp8_bias}, strict=False)

        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.bias.dtype, torch.float8_e4m3fn)

        with torch.inference_mode():
            output = layer(torch.randn(5, 10, dtype=compute_dtype))

        self.assertEqual(output.shape, (5, 20))
        self.assertEqual(output.dtype, compute_dtype)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "requires torch fp8 dtype")
    def test_native_fp8_weight_storage_falls_back_when_device_feature_test_fails(self):
        with (
            mock.patch.object(model_management, "args", Configuration(fp8_storage=True)),
            mock.patch.object(model_management, "supports_fp8_storage", return_value=False),
            mock.patch.object(model_management, "should_use_fp16", return_value=False),
            mock.patch.object(model_management, "should_use_bf16", return_value=True),
        ):
            selected_dtype = model_management.unet_dtype(
                device=torch.device("cpu"),
                model_params=10,
                supported_dtypes=(torch.bfloat16, torch.float32),
                weight_dtype=torch.float8_e4m3fn,
            )

        self.assertEqual(selected_dtype, torch.bfloat16)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "requires torch fp8 dtype")
    def test_native_fp8_weight_storage_flag_can_disable_feature_detected_storage(self):
        with (
            mock.patch.object(model_management, "args", Configuration(fp8_storage=False)),
            mock.patch.object(model_management, "supports_fp8_storage", return_value=True),
            mock.patch.object(model_management, "should_use_fp16", return_value=False),
            mock.patch.object(model_management, "should_use_bf16", return_value=True),
        ):
            selected_dtype = model_management.unet_dtype(
                device=torch.device("cpu"),
                model_params=10,
                supported_dtypes=(torch.bfloat16, torch.float32),
                weight_dtype=torch.float8_e4m3fn,
            )

        self.assertEqual(selected_dtype, torch.bfloat16)

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_state_dict_quantized_preserved(self):
        """Test that quantized weights are preserved in state_dict()"""
        # Configure mixed precision
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        # Create and load model
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict1 = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(3.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict1, _ = utils.convert_old_quants(state_dict1, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict1, strict=False)

        # Save state dict
        state_dict2 = model.state_dict()

        # Verify layer1.weight is a QuantizedTensor with scale preserved
        self.assertTrue(torch.equal(state_dict2["layer1.weight"].view(torch.uint8), fp8_weight.view(torch.uint8)))
        self.assertEqual(state_dict2["layer1.weight_scale"].item(), 3.0)
        self.assertEqual(model.layer1.weight._layout_cls, "TensorCoreFP8E4M3Layout")

        # Verify non-quantized layers are standard tensors
        self.assertNotIsInstance(state_dict2["layer2.weight"], QuantizedTensor)
        self.assertNotIsInstance(state_dict2["layer3.weight"], QuantizedTensor)

    @unittest.skipUnless(ops.mixed_precision_quantization_available(), "requires comfy_kitchen-backed quantized tensors")
    def test_weight_function_compatibility(self):
        """Test that weight_function (LoRA) works with quantized layers"""
        # Configure FP8 quantization
        layer_quant_config = {
            "layer1": {
                "format": "float8_e4m3fn",
                "params": {}
            }
        }

        # Create and load model
        fp8_weight = torch.randn(20, 10, dtype=torch.float32).to(torch.float8_e4m3fn)
        state_dict = {
            "layer1.weight": fp8_weight,
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer1.weight_scale": torch.tensor(2.0, dtype=torch.float32),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        model.load_state_dict(state_dict, strict=False)

        # Add a weight function (simulating LoRA)
        # This should trigger dequantization during forward pass
        model.layer1.weight_function = []
        def apply_lora(weight):
            lora_delta = torch.randn_like(weight) * 0.01
            return weight + lora_delta

        model.layer1.weight_function = [apply_lora]

        # Forward pass should work with LoRA (triggers weight_function path)
        input_tensor = torch.randn(5, 10, dtype=torch.bfloat16)
        output = model(input_tensor)

        self.assertEqual(output.shape, (5, 40))

    def test_error_handling_unknown_format(self):
        """Test that unknown formats raise error"""
        # Configure with unknown format
        layer_quant_config = {
            "layer1": {
                "format": "unknown_format_xyz",
                "params": {}
            }
        }

        # Create state dict
        state_dict = {
            "layer1.weight": torch.randn(20, 10, dtype=torch.bfloat16),
            "layer1.bias": torch.randn(20, dtype=torch.bfloat16),
            "layer2.weight": torch.randn(30, 20, dtype=torch.bfloat16),
            "layer2.bias": torch.randn(30, dtype=torch.bfloat16),
            "layer3.weight": torch.randn(40, 30, dtype=torch.bfloat16),
            "layer3.bias": torch.randn(40, dtype=torch.bfloat16),
        }

        state_dict, _ = utils.convert_old_quants(state_dict, metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})})

        # Load should raise KeyError for unknown format in QUANT_FORMAT_MIXINS
        model = SimpleModel(operations=ops.mixed_precision_ops({}))
        with self.assertRaises(KeyError):
            model.load_state_dict(state_dict, strict=False)

    def test_int8_convrot_metadata_loads_into_params(self):
        """ConvRot metadata must reach TensorWiseINT8Layout params."""
        torch.manual_seed(123)
        layer_quant_config = {
            "layer": {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 256,
            }
        }
        weight = torch.randn(16, 256, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.bfloat16)
        q_weight = QuantizedTensor.from_float(
            weight,
            "TensorWiseINT8Layout",
            per_channel=True,
            convrot=True,
            convrot_groupsize=256,
        )
        state_dict = {
            "layer.weight": q_weight._qdata,
            "layer.bias": bias,
            "layer.weight_scale": q_weight._params.scale,
        }

        state_dict, _ = utils.convert_old_quants(
            state_dict,
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        model = torch.nn.Module()
        model.layer = ops.mixed_precision_ops({}).Linear(256, 16, device="cpu", dtype=torch.bfloat16)
        model.load_state_dict(state_dict, strict=False)

        self.assertIsInstance(model.layer.weight, QuantizedTensor)
        self.assertEqual(model.layer.weight._layout_cls, "TensorWiseINT8Layout")
        self.assertTrue(model.layer.weight._params.convrot)
        self.assertEqual(model.layer.weight._params.convrot_groupsize, 256)

        input_tensor = torch.randn(4, 256, dtype=torch.bfloat16)
        loaded_out = model.layer(input_tensor)
        ref_out = torch.nn.functional.linear(input_tensor, q_weight, bias)
        self.assertTrue(torch.equal(loaded_out, ref_out))

        fp16_input = input_tensor.to(torch.float16)
        loaded_fp16_out = model.layer(fp16_input)
        ref_fp16_out = torch.nn.functional.linear(
            fp16_input,
            q_weight.to(dtype=torch.float16),
            bias.to(dtype=torch.float16),
        )
        self.assertTrue(torch.equal(loaded_fp16_out, ref_fp16_out))

        saved = model.state_dict()
        saved_conf = json.loads(saved["layer.comfy_quant"].numpy().tobytes())
        self.assertTrue(saved_conf["convrot"])

    def test_convrot_w4a4_loads_into_params(self):
        """ConvRot W4A4 checkpoints must load as the dedicated kitchen layout."""
        if "convrot_w4a4" not in QUANT_ALGOS:
            self.skipTest("comfy_kitchen does not provide ConvRot W4A4")

        torch.manual_seed(456)
        layer_quant_config = {
            "layer": {
                "format": "convrot_w4a4",
                "convrot_groupsize": 256,
                "linear_dtype": "int8",
            }
        }
        weight = torch.randn(16, 256, dtype=torch.bfloat16)
        bias = torch.randn(16, dtype=torch.bfloat16)
        q_weight = QuantizedTensor.from_float(
            weight,
            "TensorCoreConvRotW4A4Layout",
            convrot_groupsize=256,
            quant_group_size=64,
        )
        state_dict = {
            "layer.weight": q_weight._qdata,
            "layer.bias": bias,
            "layer.weight_scale": q_weight._params.scale,
        }

        state_dict, _ = utils.convert_old_quants(
            state_dict,
            metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
        )
        model = torch.nn.Module()
        model.layer = ops.mixed_precision_ops({}).Linear(256, 16, device="cpu", dtype=torch.bfloat16)
        model.load_state_dict(state_dict, strict=False)

        self.assertIsInstance(model.layer.weight, QuantizedTensor)
        self.assertEqual(model.layer.weight._layout_cls, "TensorCoreConvRotW4A4Layout")
        self.assertEqual(model.layer.weight._params.convrot_groupsize, 256)
        self.assertEqual(model.layer.weight._params.quant_group_size, 64)
        self.assertEqual(model.layer.weight._params.linear_dtype, "int8")

        input_tensor = torch.randn(4, 256, dtype=torch.bfloat16)
        loaded_out = model.layer(input_tensor)
        ref_out = torch.nn.functional.linear(input_tensor, q_weight, bias)
        self.assertTrue(torch.equal(loaded_out, ref_out))

        saved = model.state_dict()
        saved_conf = json.loads(saved["layer.comfy_quant"].numpy().tobytes())
        self.assertEqual(saved_conf["format"], "convrot_w4a4")
        self.assertEqual(saved_conf["convrot_groupsize"], 256)
        self.assertEqual(saved_conf["linear_dtype"], "int8")
        self.assertNotIn("quant_group_size", saved_conf)

    def test_legacy_int8_metadata_normalizes_to_kitchen_tensorwise(self):
        def marker(conf: dict) -> torch.Tensor:
            return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)

        base = {
            "plain.weight": torch.zeros(8, 16, dtype=torch.int8),
            "plain.weight_scale": torch.ones(8, 1),
            "convrot.weight": torch.zeros(8, 256, dtype=torch.int8),
            "convrot.weight_scale": torch.ones(8, 1),
            "bare.weight": torch.zeros(8, 16, dtype=torch.int8),
            "bare.weight_scale": torch.ones(8, 1),
            "plain.comfy_quant": marker({"format": "int8"}),
            "convrot.comfy_quant": marker({"format": "int8_convrot", "convrot_groupsize": 256}),
        }

        out, _ = utils.convert_old_quants(base, metadata={})

        plain_conf = json.loads(out["plain.comfy_quant"].numpy().tobytes())
        convrot_conf = json.loads(out["convrot.comfy_quant"].numpy().tobytes())
        bare_conf = json.loads(out["bare.comfy_quant"].numpy().tobytes())

        self.assertEqual(plain_conf["format"], "int8_tensorwise")
        self.assertEqual(bare_conf["format"], "int8_tensorwise")
        self.assertEqual(convrot_conf["format"], "int8_tensorwise")
        self.assertTrue(convrot_conf["convrot"])
        self.assertEqual(convrot_conf["convrot_groupsize"], 256)

    def test_int8_disabled_on_unsupported_device_falls_back_to_full_precision(self):
        """On a device that can't run comfy_kitchen's fast int8 matmul (e.g. MPS,
        which lacks aten::_int_mm), pick_operations must mark int8 formats as
        disabled so layers dequantize instead of taking the fast quantized path."""
        mm = model_management

        orig_supports_int8 = mm.supports_int8_compute
        mm.supports_int8_compute = lambda device=None: False
        try:
            model_config = SimpleNamespace(quant_config={"layer": {"format": "int8_tensorwise"}})
            operations = ops.pick_operations(torch.bfloat16, torch.bfloat16, model_config=model_config)

            torch.manual_seed(789)
            weight = torch.randn(16, 256, dtype=torch.bfloat16)
            bias = torch.randn(16, dtype=torch.bfloat16)
            q_weight = QuantizedTensor.from_float(weight, "TensorWiseINT8Layout", per_channel=True)
            state_dict = {
                "layer.weight": q_weight._qdata,
                "layer.bias": bias,
                "layer.weight_scale": q_weight._params.scale,
            }
            layer_quant_config = {"layer": {"format": "int8_tensorwise"}}
            state_dict, _ = utils.convert_old_quants(
                state_dict,
                metadata={"_quantization_metadata": json.dumps({"layers": layer_quant_config})},
            )

            model = torch.nn.Module()
            model.layer = operations.Linear(256, 16, device="cpu", dtype=torch.bfloat16)
            model.load_state_dict(state_dict, strict=False)

            self.assertIsInstance(model.layer.weight, QuantizedTensor)
            # The layer must be forced onto the full-precision (dequantized)
            # path since the fast int8 path isn't usable on this device.
            self.assertTrue(model.layer._full_precision_mm)

            # The weight's orig_dtype matches the compute dtype here (both bfloat16),
            # so cast_bias_weight's dtype-change check alone won't dequantize it. Confirm
            # the module still hands a real Tensor (not a QuantizedTensor) to the plain
            # linear() call, since dispatching a QuantizedTensor there would route back
            # into the disabled fast int8 matmul instead of the full-precision fallback.
            seen_weight_types = []
            orig_module_forward = model.layer._forward
            def _capturing_forward(input, weight, bias, _orig=orig_module_forward):
                seen_weight_types.append(type(weight))
                return _orig(input, weight, bias)
            model.layer._forward = _capturing_forward

            input_tensor = torch.randn(4, 256, dtype=torch.bfloat16)
            output = model.layer(input_tensor)
            self.assertEqual(output.shape, (4, 16))
            self.assertEqual(seen_weight_types, [torch.Tensor])
        finally:
            mm.supports_int8_compute = orig_supports_int8

    def test_linear_input_act_respects_full_precision_mm_fallback(self):
        """linear_input_act folds an activation into the INT8 GEMM's input quantizer,
        bypassing Linear.forward entirely. On a device where the fast int8 kernel is
        disabled (e.g. MPS, which lacks aten::_int_mm), it must honor _full_precision_mm
        and dequantize instead, exactly like Linear.forward_comfy_cast_weights does
        (see Comfy-Org/ComfyUI#16284)."""
        operations = ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16)

        torch.manual_seed(456)
        weight = torch.randn(32, 64, dtype=torch.bfloat16)
        bias = torch.randn(32, dtype=torch.bfloat16)

        layer = operations.Linear(64, 32, bias=True, device="cpu", dtype=torch.bfloat16)
        layer.weight = torch.nn.Parameter(
            QuantizedTensor.from_float(weight, "TensorWiseINT8Layout"), requires_grad=False
        )
        layer.bias = torch.nn.Parameter(bias, requires_grad=False)
        layer.quant_format = "int8_tensorwise"
        layer._full_precision_mm = True

        x = torch.randn(4, 128, dtype=torch.bfloat16)

        orig_int8_linear = ops.quant_ops.ck.int8_linear
        ops.quant_ops.ck.int8_linear = unittest.mock.Mock(
            side_effect=NotImplementedError("aten::_int_mm not implemented")
        )
        try:
            output = ops.linear_input_act(layer, x, "swiglu")
        finally:
            ops.quant_ops.ck.int8_linear = orig_int8_linear

        expected = torch.nn.functional.linear(
            ops.INPUT_ACT_EAGER["swiglu"](x), layer.weight.dequantize(), bias
        )
        torch.testing.assert_close(output, expected)

    def test_supports_int8_compute_treats_mps_mode_as_unsupported_when_device_is_none(self):
        """Call sites (like pick_operations' default) may omit load_device. On an
        MPS machine that must still report int8 as unsupported instead of
        silently defaulting to True, matching supports_fp64's handling of the
        same device=None case (see Comfy-Org/ComfyUI#16136)."""
        mm = model_management

        orig_cpu_state = mm.cpu_state
        mm.cpu_state = mm.CPUState.MPS
        try:
            self.assertFalse(mm.supports_int8_compute(None))
        finally:
            mm.cpu_state = orig_cpu_state


if __name__ == "__main__":
    unittest.main()

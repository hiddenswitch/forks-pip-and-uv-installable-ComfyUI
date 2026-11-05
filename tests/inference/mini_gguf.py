"""GGUF checkpoint descriptors and native ggml encoding for miniature inference."""
import ctypes
import ctypes.util
import math
import os

import gguf
import numpy as np
import torch

from comfy.gguf import GGMLLayer, GGMLTensor


def quantize(value, kind):
    """Encode actual floating weights; never substitute the runtime dequantizer."""
    value = value.detach().cpu().float().contiguous().numpy()
    if kind in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
        return torch.from_numpy(gguf.quants.quantize(value, kind))
    path = os.environ.get('COMFYUI_TEST_GGML_LIBRARY') or ctypes.util.find_library('ggml-base')
    if not path:
        raise RuntimeError('Miniature GGUF encoding requires COMFYUI_TEST_GGML_LIBRARY pointing to native libggml-base')
    lib = ctypes.CDLL(path)
    lib.ggml_blck_size.argtypes = [ctypes.c_int]
    lib.ggml_blck_size.restype = ctypes.c_int64
    lib.ggml_type_size.argtypes = [ctypes.c_int]
    lib.ggml_type_size.restype = ctypes.c_size_t
    lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
    lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
    block, size = gguf.GGML_QUANT_SIZES[kind]
    if (lib.ggml_blck_size(kind), lib.ggml_type_size(kind)) != (block, size):
        raise ValueError(f'GGUF/ggml format ABI mismatch: {kind.name}')
    if lib.ggml_quantize_requires_imatrix(kind):
        raise ValueError(f'Miniature GGUF needs an importance matrix for {kind.name}')
    if value.shape[-1] % block:
        raise ValueError(f'Miniature {kind.name} row width must be divisible by {block}: {value.shape}')
    output = np.empty((*value.shape[:-1], value.shape[-1] // block * size), dtype=np.uint8)
    encode = lib.ggml_quantize_chunk
    encode.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    encode.restype = ctypes.c_size_t
    written = encode(kind, value.ctypes.data, output.ctypes.data, 0,
                     math.prod(value.shape[:-1]), value.shape[-1], None)
    if written != output.nbytes:
        raise ValueError(f'Native GGUF encoder wrote {written} bytes, expected {output.nbytes}')
    return torch.from_numpy(output)


class MiniatureGGUF:
    def __init__(self, reader):
        self.reader = reader
        self.formats = {}

    def describe(self, source, *args, **kwargs):
        if not os.fspath(source).lower().endswith('.gguf'):
            raise ValueError(f'Unsupported miniature checkpoint: {source}')
        # The native GGUF reader mmaps payloads; only small buffers are touched.
        result = self.reader(source, *args, **kwargs)
        state = result[0] if kwargs.get('return_metadata', False) else result
        for key, tensor in state.items():
            if tensor.ndim > 1:
                state[key] = tensor.to(device='meta')
        return result

    def shapes(self, module, state, targets):
        self.formats = {key: value.tensor_type for key, value in state.items() if isinstance(value, GGMLTensor)}
        for name, child in module.named_modules():
            if not isinstance(child, GGMLLayer) or not isinstance(child, torch.nn.Linear):
                continue
            prefix = name + '.' if name else ''
            for suffix, shape in (('weight', (child.out_features, child.in_features)), ('bias', (child.out_features,))):
                if prefix + suffix in state:
                    targets[prefix + suffix] = torch.empty(shape, dtype=torch.float32, device='meta')

    def encode(self, weights):
        for key, kind in self.formats.items():
            if key in weights:
                value = weights[key].as_subclass(torch.Tensor)
                weights[key] = GGMLTensor(quantize(value, kind), tensor_type=kind, tensor_shape=value.shape)
        self.formats.clear()

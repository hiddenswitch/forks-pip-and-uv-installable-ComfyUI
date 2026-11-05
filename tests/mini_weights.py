"""Scoped checkpoint loading with real random parameters, independent of the model framework."""
import contextlib
import functools
import json
import math
import os
import struct
from collections import OrderedDict
from unittest.mock import patch

import torch
from safetensors.torch import _TYPES


def random_state(state, targets, *, seed=0, max_bytes=1024**3, device="cpu"):
    """Materialize source keys at receiver shapes, retaining small source buffers."""
    size = sum(t.numel() * t.element_size() for k, t in targets.items() if k in state)
    if size > max_bytes:
        raise ValueError(f'Miniature checkpoint still requires {size / 1024**2:.0f} MiB (limit {max_bytes / 1024**2:.0f})')
    generator = torch.Generator(device=device).manual_seed(seed)
    weights = OrderedDict()
    for name, tensor in state.items():
        if name not in targets:
            # Removed repeated blocks are intentionally absent in the smaller model.
            continue
        target = targets[name]
        if not tensor.is_meta:
            weights[name] = tensor.flatten()[:target.numel()].reshape(target.shape).to(target.dtype)
            continue
        if target.ndim < 2:
            value = torch.ones(target.shape, device=device) if name.endswith(('weight', 'scale')) else torch.zeros(target.shape, device=device)
        else:
            value = torch.randn(target.shape, generator=generator, device=device).div_(math.sqrt(math.prod(target.shape[1:])))
        weights[name] = value.to(target.dtype)
    if hasattr(state, '_metadata'):
        weights._metadata = state._metadata
    return weights


@contextlib.contextmanager
def miniature_checkpoints(reader_owner, reader_name, *, sources=None, seed=0, max_bytes=1024**3, shapes=None, encode=None, describe=None, device="cpu"):
    """Keep original shapes for detection, then initialize the receiving module's shapes.

    Constructors may be configured smaller by the caller. Only checkpoint reads
    and state loading change; every forward and kernel consumes real tensors.
    The per-load budget bounds target parameter bytes, not temporary conversion
    or quantization buffers. Missing keys stay
    missing rather than concealing an incomplete checkpoint. Optional shape and
    encoding callbacks support modules with deferred or quantized parameters.
    A descriptor callback can read additional checkpoint formats without adding
    their framework or storage dependencies here.
    Yields module names that actually received generated weights.
    """
    selected = sources if callable(sources) else None if sources is None else {os.path.realpath(p) for p in sources}
    read = getattr(reader_owner, reader_name)
    load_state = torch.nn.Module.load_state_dict
    loaded = []

    @functools.wraps(read)
    def header_only(source, *args, **kwargs):
        path = os.path.realpath(source)
        if selected is not None and not (selected(os.fspath(source)) if callable(selected) else path in selected):
            return read(source, *args, **kwargs)
        return descriptors(source, *args, **kwargs)

    def descriptors(source, *args, **kwargs):
        if os.fspath(source).lower().endswith(('.pth', '.pt', '.ckpt', '.bin')):
            return read(source, *args, **{**kwargs, 'device': torch.device('meta')})
        if os.fspath(source).lower().endswith('.index.json'):
            with open(source) as f:
                index = json.load(f)
            state = {}
            for shard in sorted(set(index.get('weight_map', index).values())):
                state.update(descriptors(os.path.join(os.path.dirname(source), shard),
                                         **{**kwargs, 'return_metadata': False}))
            return (state, None) if kwargs.get('return_metadata', False) else state
        if not os.fspath(source).lower().endswith(('.safetensors', '.sft')):
            if describe is not None:
                return describe(source, *args, **kwargs)
            raise ValueError(f'Miniature weights require safetensors: {source}')
        with open(source, 'rb') as f:
            data_start = 8 + struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(data_start - 8))
            has_values = os.fstat(f.fileno()).st_size > data_start
            state = {}
            for name, info in header.items():
                if name == '__metadata__' or kwargs.get('include_keys') is not None and name not in kwargs['include_keys']:
                    continue
                dtype = _TYPES[info['dtype']]
                start, end = info['data_offsets']
                small_buffer = len(info['shape']) <= 1 and (not dtype.is_floating_point or has_values and end - start <= 65536)
                if small_buffer:
                    start, end = info['data_offsets']
                    if end - start > (64 * 1024**2 if len(info['shape']) == 1 else 1024**2):
                        raise ValueError(f'Miniature packed tensor is unsupported: {name} in {source}')
                    f.seek(data_start + start)
                    state[name] = (torch.frombuffer(bytearray(f.read(end - start)), dtype=dtype).reshape(info['shape'])
                                   if end > start else torch.empty(info['shape'], dtype=dtype))
                else:
                    state[name] = torch.empty(info['shape'], device='meta', dtype=dtype)
        return (state, header.get('__metadata__', {})) if kwargs.get('return_metadata', False) else state

    @functools.wraps(load_state)
    def initialize(module, state, *args, **kwargs):
        if not any(t.is_meta for t in state.values() if isinstance(t, torch.Tensor)):
            return load_state(module, state, *args, **kwargs)
        targets = module.state_dict() if shapes is None else shapes(module, state)
        weights = random_state(state, targets, seed=seed, max_bytes=max_bytes, device=device)
        if encode is not None:
            encode(module, weights)
        result = load_state(module, weights, *args, **kwargs)
        loaded.append(type(module).__name__)
        return result

    with patch.object(reader_owner, reader_name, header_only), patch.object(torch.nn.Module, 'load_state_dict', initialize):
        yield loaded

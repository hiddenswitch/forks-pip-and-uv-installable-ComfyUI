"""Transport execution-local custom-node caches without retaining worker state."""
from types import MethodType

import torch


class StateCodec:
    def __init__(self, classes):
        self.classes = tuple(classes)

    def encode(self, value):
        if type(value) in self.classes:
            return "object", (self.classes.index(type(value)), self.encode(vars(value)))
        if isinstance(value, dict):
            return "dict", [(self.encode(k), self.encode(v)) for k, v in value.items()]
        if isinstance(value, (tuple, list, set)):
            return type(value).__name__, [self.encode(v) for v in value]
        if value is None or isinstance(value, (bool, int, float, str, torch.Tensor, torch.dtype, torch.device)):
            return "value", value
        raise TypeError(f"Unsupported custom-node cache value: {type(value).__name__}")

    def decode(self, encoded):
        kind, value = encoded
        if kind == "object":
            index, attributes = value
            result = object.__new__(self.classes[index])
            vars(result).update(self.decode(attributes))
            return result
        if kind == "dict":
            return {self.decode(k): self.decode(v) for k, v in value}
        if kind in ("tuple", "list", "set"):
            return {"tuple": tuple, "list": list, "set": set}[kind](self.decode(v) for v in value)
        if kind == "value":
            return value
        raise ValueError(f"Unknown custom-node cache encoding: {kind}")


def _cached_forward(self, encoded, args, kwargs):
    # The root rank owns the durable state. Peers reconstruct only this call's
    # copy, so cancelled jobs cannot strand cache tensors in worker processes.
    state = self.tensor_parallel_state_codec.decode(encoded)
    kwargs["transformer_options"][self.tensor_parallel_state_key] = state
    output = self(*args, **kwargs)
    return output, self.tensor_parallel_state_codec.encode(state)


def install_state_transport(model, key, classes):
    model.tensor_parallel_state_key = key
    model.tensor_parallel_state_codec = StateCodec(classes)
    model.tensor_parallel_cached_forward = MethodType(_cached_forward, model)


class CachedStateExecutor:
    def __init__(self, executor):
        self.executor = executor

    def __getattr__(self, name):
        return getattr(self.executor, name)

    def execute(self, *args, **kwargs):
        return self.execute_method("forward", *args, **kwargs)

    def execute_method(self, method, *args, **kwargs):
        model = self.executor.root_model
        options = kwargs.get("transformer_options", {})
        key = model.tensor_parallel_state_key
        state = options.get(key)
        if method != "forward" or state is None:
            return self.executor.execute_method(method, *args, **kwargs)
        codec = model.tensor_parallel_state_codec
        kwargs = dict(kwargs, transformer_options={k: v for k, v in options.items() if k != key})
        output, updated = self.executor.execute_method(
            "tensor_parallel_cached_forward", codec.encode(state), args, kwargs,
        )
        vars(state).clear()
        vars(state).update(vars(codec.decode(updated)))
        return output

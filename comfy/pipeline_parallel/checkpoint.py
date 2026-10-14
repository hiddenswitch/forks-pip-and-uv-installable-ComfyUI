from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
import struct
from typing import Collection, Mapping

import torch

from .. import utils

from .types import TensorDescriptor


class AbstractBaseCheckpointReader(ABC):
    @property
    @abstractmethod
    def metadata(self) -> Mapping[str, str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def tensors(self) -> Mapping[str, TensorDescriptor]:
        raise NotImplementedError

    @abstractmethod
    def detection_state_dict(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def load_keys(self, keys: Collection[str]) -> dict[str, torch.Tensor]:
        raise NotImplementedError


SHARDED_SAFETENSORS_INDEX_SUFFIX = ".safetensors.index.json"


def is_safetensors_checkpoint(path) -> bool:
    """A single safetensors file, or the index of a sharded safetensors checkpoint."""
    name = os.fspath(path).lower()
    return name.endswith((".safetensors", ".sft", SHARDED_SAFETENSORS_INDEX_SUFFIX))


def _read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as checkpoint:
        header_size = struct.unpack("<Q", checkpoint.read(8))[0]
        return json.loads(checkpoint.read(header_size))


class SafetensorsCheckpointReader(AbstractBaseCheckpointReader):
    """Read selected tensors of a safetensors checkpoint without loading the rest.

    ``path`` is a single ``.safetensors`` file or the ``model.safetensors.index.json`` of a
    sharded checkpoint, whose shards sit beside it. A sharded checkpoint is read shard by
    shard: a request opens only the shards holding the requested keys.
    """

    _VALUE_KEYS = frozenset(("scaled_fp8",))

    def __init__(self, path: str | Path):
        path = Path(path)
        if not is_safetensors_checkpoint(path):
            raise ValueError(
                "Pipeline parallel loading currently requires a safetensors checkpoint "
                "or a sharded safetensors index"
            )
        # absolute but not resolved: a Hugging Face cache symlink points at an extensionless blob,
        # and load_torch_file picks the safetensors reader by the extension; the shards of an index
        # are its siblings in the snapshot directory, not in the blob store
        self.path = path.absolute()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if self.path.name.lower().endswith(SHARDED_SAFETENSORS_INDEX_SUFFIX):
            weight_map = json.loads(self.path.read_text())["weight_map"]
            shards = {name: self.path.parent / name for name in sorted(set(weight_map.values()))}
        else:
            weight_map = None
            shards = {self.path.name: self.path}

        self._metadata = {}
        self._tensors = {}
        self._shards: dict[str, Path] = {}
        for shard_name, shard_path in shards.items():
            header = _read_safetensors_header(shard_path)
            for key, value in header.pop("__metadata__", {}).items():
                if self._metadata.setdefault(key, value) != value:
                    raise ValueError(f"{shard_path.name}: metadata {key!r} differs from another shard's")
            for name, info in header.items():
                if weight_map is not None and weight_map.get(name) != shard_name:
                    raise ValueError(f"{shard_path.name} holds {name}, which its index places elsewhere")
                if name in self._tensors:
                    raise ValueError(f"{name} appears in more than one shard")
                self._tensors[name] = TensorDescriptor(
                    shape=tuple(info["shape"]),
                    dtype=utils._TYPES[info["dtype"]],
                    nbytes=info["data_offsets"][1] - info["data_offsets"][0],
                )
                self._shards[name] = shard_path
        if weight_map is not None:
            missing = sorted(set(weight_map).difference(self._tensors))
            if missing:
                raise ValueError(f"{self.path.name} indexes tensors no shard holds: {missing[:5]}")

    @property
    def metadata(self) -> Mapping[str, str]:
        return self._metadata

    @property
    def tensors(self) -> Mapping[str, TensorDescriptor]:
        return self._tensors

    def detection_state_dict(self) -> dict[str, torch.Tensor]:
        value_keys = {
            key for key in self._tensors
            if key.endswith(".comfy_quant") or key in self._VALUE_KEYS
        }
        values = self.load_keys(value_keys)
        return {
            key: values.get(key, torch.empty(descriptor.shape, dtype=descriptor.dtype, device="meta"))
            for key, descriptor in self._tensors.items()
        }

    def load_keys(self, keys: Collection[str]) -> dict[str, torch.Tensor]:
        unknown = set(keys).difference(self._tensors)
        if unknown:
            raise KeyError(f"Checkpoint does not contain pipeline keys: {sorted(unknown)[:5]}")
        by_shard: dict[Path, set[str]] = {}
        for key in keys:
            by_shard.setdefault(self._shards[key], set()).add(key)
        result = {}
        for shard_path, shard_keys in by_shard.items():
            result.update(utils.load_torch_file(str(shard_path), include_keys=frozenset(shard_keys)))
        return result

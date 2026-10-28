from __future__ import annotations

import torch

from .types import TensorParallelConfig


def tensor_parallel_size(operations) -> int:
    parallel = getattr(operations, "tensor_parallel", None)
    return 1 if parallel is None else parallel.size


def local_size(operations, value: int, name: str) -> int:
    size = tensor_parallel_size(operations)
    if value % size:
        raise ValueError(f"{name} {value} must be divisible by tensor parallel size {size}")
    return value // size


def column_parallel_linear(operations, in_features, out_features, bias=True, *,
                           sections=1, section_sizes=None, device=None, dtype=None):
    if tensor_parallel_size(operations) == 1:
        return operations.Linear(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
    return operations.ColumnParallelLinear(
        in_features,
        out_features,
        bias=bias,
        sections=sections,
        section_sizes=section_sizes,
        device=device,
        dtype=dtype,
    )


def row_parallel_linear(operations, in_features, out_features, bias=False, *,
                        section_sizes=None, device=None, dtype=None):
    if tensor_parallel_size(operations) == 1:
        return operations.Linear(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
    section_options = {} if section_sizes is None else {"section_sizes": section_sizes}
    return operations.RowParallelLinear(
        in_features, out_features, bias=bias, device=device, dtype=dtype, **section_options
    )


def tensor_parallel_operations(base_operations, parallel: TensorParallelConfig):
    """Decorate Comfy operations with Megatron-style tensor-parallel linears."""

    class TensorParallelOperations(base_operations):
        tensor_parallel = parallel

        class ColumnParallelLinear(base_operations.Linear):
            def __init__(self, in_features, out_features, bias=True, *, sections=1,
                         section_sizes=None,
                         device=None, dtype=None):
                if section_sizes is None:
                    if out_features % sections:
                        raise ValueError(
                            f"Column-parallel output {out_features} must divide {sections} sections"
                        )
                    section_sizes = (out_features // sections,) * sections
                else:
                    section_sizes = tuple(section_sizes)
                    if sum(section_sizes) != out_features:
                        raise ValueError(
                            f"Column-parallel sections {section_sizes} do not sum to {out_features}"
                        )
                if any(section % parallel.size for section in section_sizes):
                    raise ValueError(
                        f"Column-parallel sections {section_sizes} must divide across "
                        f"{parallel.size} ranks"
                    )
                super().__init__(
                    in_features,
                    sum(section // parallel.size for section in section_sizes),
                    bias=bias,
                    device=device, dtype=dtype,
                )
                self.tensor_parallel_section_sizes = section_sizes
                self.tensor_parallel_out_features = out_features

        class RowParallelLinear(base_operations.Linear):
            # Fused activation helpers must preserve the collective in forward.
            comfy_force_forward = True

            def __init__(self, in_features, out_features, bias=True, *,
                         section_sizes=None, device=None, dtype=None):
                sections = (in_features,) if section_sizes is None else tuple(section_sizes)
                if sum(sections) != in_features or any(section % parallel.size for section in sections):
                    raise ValueError(
                        f"Row-parallel input {in_features} must divide {parallel.size} ranks"
                    )
                super().__init__(
                    in_features // parallel.size, out_features, bias=bias and parallel.rank == 0,
                    device=device, dtype=dtype,
                )
                self.tensor_parallel_in_features = in_features

            def forward(self, *args, **kwargs):
                return parallel.operations.sum(super().forward(*args, **kwargs))

    TensorParallelOperations.__name__ = f"TensorParallel{base_operations.__name__}"
    return TensorParallelOperations


def full_domain_tensor_parallel_operations(base_operations, parallel: TensorParallelConfig):
    """Preserve the complete input domain for activation-rotating quantization.

    ConvRot rotates across the full feature dimension before quantization. A
    conventional row-parallel linear would rotate each feature shard
    independently, which is not equivalent. Gather the activation, calculate
    an output-row shard, and gather the result instead.
    """
    TensorParallelOperations = tensor_parallel_operations(base_operations, parallel)

    class FullDomainTensorParallelOperations(TensorParallelOperations):
        class RowParallelLinear(base_operations.Linear):
            comfy_force_forward = True

            def __init__(self, in_features, out_features, bias=True, *,
                         section_sizes=None, device=None, dtype=None):
                sections = (in_features,) if section_sizes is None else tuple(section_sizes)
                if sum(sections) != in_features or any(section % parallel.size for section in sections):
                    raise ValueError(f"Input sections {sections} must divide {parallel.size} ranks")
                if out_features % parallel.size:
                    raise ValueError(
                        f"Full-domain row-parallel output {out_features} must divide "
                        f"{parallel.size} ranks"
                    )
                super().__init__(
                    in_features,
                    out_features // parallel.size,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
                self.tensor_parallel_in_features = in_features
                self.tensor_parallel_out_features = out_features
                self.tensor_parallel_input_sections = tuple(section // parallel.size for section in sections)

            def forward(self, input, *args, **kwargs):
                if len(self.tensor_parallel_input_sections) == 1:
                    full_input = parallel.operations.gather(input)
                else:
                    full_input = torch.cat([
                        parallel.operations.gather(section)
                        for section in input.split(self.tensor_parallel_input_sections, dim=-1)
                    ], dim=-1)
                local_output = super().forward(full_input, *args, **kwargs)
                return parallel.operations.gather(local_output)

    FullDomainTensorParallelOperations.__name__ = (
        f"FullDomainTensorParallel{base_operations.__name__}"
    )
    return FullDomainTensorParallelOperations


def gathered_output_operations(base_operations, parallel: TensorParallelConfig):
    """Shard output rows while keeping the full activation quantization domain."""
    class GatheredOutputOperations(base_operations):
        class Linear(base_operations.Linear):
            comfy_force_forward = True

            def __init__(self, in_features, out_features, bias=True, *, device=None, dtype=None):
                if out_features % parallel.size:
                    raise ValueError(f"Output size {out_features} must divide {parallel.size} ranks")
                super().__init__(in_features, out_features // parallel.size, bias=bias, device=device, dtype=dtype)

            def forward(self, *args, **kwargs):
                return parallel.operations.gather(super().forward(*args, **kwargs))

    return GatheredOutputOperations

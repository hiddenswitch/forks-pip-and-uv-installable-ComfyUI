"""Run W4A8 projection and expert sharding through real CUDA kernels and NCCL."""
import json
import importlib
from datetime import timedelta
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from comfy import ops
from comfy.tensor_parallel.operations import gathered_output_operations
from comfy.tensor_parallel.runtime import TorchDistributedTensorParallelOperations
from comfy.tensor_parallel.types import TensorParallelConfig
from comfy.tensor_parallel.distributed import TorchDistributedModelParallelExecutor, _ObjectCoordinator, _run_worker
from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1
from comfy_compatibility.vanilla import prepare_vanilla_environment


def _w4a8_rank(rank, rendezvous):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(31)
        entry = next(ep for ep in distribution("comfyui-hunyuanimage3").entry_points if ep.group == "comfyui.custom_nodes").load()
        root = Path(entry.COMFYUI_VANILLA_NODE_PATH) / "ComfyUI-HunyuanImage3"
        prepare_vanilla_environment()
        assert _vanilla_load_custom_nodes_1(str(root)).NODE_CLASS_MAPPINGS
        sliced = importlib.import_module(root.name + ".hunyuan_image_3.ops").expert_linear_sliced
        normalize_codebooks = importlib.import_module(root.name + ".hunyuan_image_3.loader").shared_codebooks
        device = torch.device("cuda", rank)
        base = ops.mixed_precision_ops({}, torch.bfloat16)
        parallel = TensorParallelConfig(TorchDistributedTensorParallelOperations(rank, 2, device, dist.group.WORLD))
        sharded = gathered_output_operations(base, parallel)
        for bank in (False, True):
            shape = (2, 256, 256) if bank else (256, 256)
            state = {
                "weight": torch.randint(-128, 127, shape, dtype=torch.int8),
                "weight_s_rel": torch.ones((*shape[:-1], 32)).to(torch.float8_e4m3fn),
                "weight_s_channel": torch.full(shape[:-1], 0.01),
                "weight_codebook": torch.linspace(-1, 1, 16).repeat(2, 1) if bank else torch.linspace(-1, 1, 16),
                "comfy_quant": torch.tensor(list(json.dumps({"format": "asym_w4a8_int8", "group_size": 16,
                                                             "convrot_groupsize": 256}).encode()), dtype=torch.uint8),
            }
            reference = base.MoEExperts(2, 512, 256, bias=False) if bank else base.Linear(512, 256, bias=False)
            target = base.MoEExperts(1, 512, 256, bias=False) if bank else sharded.Linear(512, 256, bias=False)
            reference_state = dict(state)
            normalize_codebooks(reference_state)
            reference.load_state_dict(reference_state)
            width = 1 if bank else 128
            owned = {k: v.narrow(0, rank * width, width).clone() if k in ("weight", "weight_s_rel", "weight_s_channel") or (bank and k == "weight_codebook")
                     else v.clone() for k, v in state.items()}
            normalize_codebooks(owned)
            target.load_state_dict(owned)
            for tokens in (1, 5):
                value = torch.randn(tokens, 512).to(device=device, dtype=torch.bfloat16)
                expected = sum(reference.expert_linear(value, i) for i in range(2)) if bank else reference(value)
                actual = parallel.operations.sum(target.expert_linear(value, 0)) if bank else target(value)
                assert expected.shape == (tokens, 256), (bank, tokens, expected.shape, reference.weight.shape)
                assert actual.shape == expected.shape, (bank, tokens, actual.shape, expected.shape)
                torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
                if bank:
                    torch.testing.assert_close(sliced(target, value, 0), sliced(reference, value, rank), rtol=0.01, atol=0.002)
            if bank:
                # Weight patches take the dense fallback, whose flattened bank
                # must still be indexed by expert rather than by output row.
                reference.weight_function = target.weight_function = [lambda weight: weight * 1.1]
                torch.testing.assert_close(target.expert_linear(value, 0), reference.expert_linear(value, rank), rtol=0.01, atol=0.002)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_w4a8_tp2_matches_unsharded_cuda(tmp_path):
    mp.spawn(_w4a8_rank, args=((tmp_path / "nccl-init").as_uri(),), nprocs=2, join=True)


def _failing_rank(rank, rendezvous):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=20))
    group = dist.new_group(backend="nccl", device_id=device, timeout=timedelta(seconds=20))
    operations = TorchDistributedTensorParallelOperations(rank, 2, device, group, dist.group.WORLD)
    coordinator = _ObjectCoordinator()

    class Model:
        def forward(self):
            if rank:
                raise RuntimeError("synthetic peer allocation failure")
            return operations.sum(torch.ones(1, device=device)).item()

    if rank == 0:
        executor = TorchDistributedModelParallelExecutor(Model(), operations, coordinator, [], [],
                                                         [torch.device("cuda", i) for i in range(2)], [0, 0])
        try:
            started = time.monotonic()
            with pytest.raises(RuntimeError):
                executor.execute()
            assert time.monotonic() - started < 10, "peer errors must not wait for the NCCL timeout"
        finally:
            executor.close()
    else:
        _run_worker(operations, coordinator, SimpleNamespace(model=SimpleNamespace(diffusion_model=Model())), "test")
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_peer_failure_interrupts_nccl_collective(tmp_path):
    processes = mp.spawn(_failing_rank, args=((tmp_path / "nccl-failure").as_uri(),), nprocs=2, join=False)
    deadline = time.monotonic() + 45
    try:
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("A peer failure left the root blocked in NCCL")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)

import pytest
import torch
import torch.distributed as dist

from comfy.xdit.runtime import TorchDistributedUlyssesOperations


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("dim", [-1, -2, 0, 3])
def test_all_gather_accepts_negative_dimensions(tmp_path, dim):
    store = dist.FileStore(str(tmp_path / "rendezvous"), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        operations = TorchDistributedUlyssesOperations(0, 1, torch.device("cpu"), dist.group.WORLD)
        mask = torch.arange(132, dtype=torch.float32).reshape(1, 1, 1, 132)
        torch.testing.assert_close(operations.all_gather(mask, dim), mask)
    finally:
        dist.destroy_process_group()

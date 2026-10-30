import pytest
import torch

from comfy.distributed import topology


def _link(source, destination, peer, gbps):
    return topology.DeviceLink(
        source=torch.device("cuda", source),
        destination=torch.device("cuda", destination),
        peer_access=peer,
        bandwidth_bytes_per_second=gbps * 1e9,
    )


def test_max_tensor_parallel_size_counts_identical_devices():
    names = ["NVIDIA RTX A5000", "NVIDIA RTX A5000", "NVIDIA GeForce RTX 3090"]
    assert topology.max_tensor_parallel_size(names, nccl_available=True, windows=False) == 2
    assert topology.max_tensor_parallel_size(["NVIDIA RTX A5000"] * 4, nccl_available=True, windows=False) == 4
    assert topology.max_tensor_parallel_size(["NVIDIA RTX A5000"] * 3, nccl_available=True, windows=False) == 2
    assert topology.max_tensor_parallel_size(["a", "b"], nccl_available=True, windows=False) == 1


def test_max_tensor_parallel_size_requires_nccl_off_windows():
    names = ["NVIDIA RTX A5000"] * 2
    assert topology.max_tensor_parallel_size(names, nccl_available=False, windows=False) == 1
    assert topology.max_tensor_parallel_size(names, nccl_available=True, windows=True) == 1


def test_max_pipeline_parallel_size_reports_the_transport():
    devices = [torch.device("cuda", 0), torch.device("cuda", 1), torch.device("cuda", 2)]
    peer_links = [_link(0, 1, True, 40), _link(1, 0, True, 40), _link(1, 2, True, 40), _link(2, 1, True, 40)]
    assert topology.max_pipeline_parallel_size(devices, peer_links) == (3, "peer")
    broken = [_link(0, 1, True, 40), _link(1, 2, False, 10)]
    assert topology.max_pipeline_parallel_size(devices, broken) == (3, "mp")
    assert topology.max_pipeline_parallel_size(devices[:1], []) == (1, "none")


def test_link_lookup_is_by_ordered_pair():
    links = [_link(0, 1, True, 40), _link(1, 0, False, 10)]
    table = topology.links_by_pair(links)
    assert table[(torch.device("cuda", 0), torch.device("cuda", 1))].peer_access is True
    assert table[(torch.device("cuda", 1), torch.device("cuda", 0))].peer_access is False


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs two CUDA devices")
@pytest.mark.requires_accelerator
def test_measure_device_links_on_real_devices():
    devices = topology.cuda_devices()[:2]
    links = topology.measure_device_links(devices, size_bytes=64 * 1024 * 1024, iterations=2)
    assert len(links) == 2
    for link in links:
        assert link.bandwidth_bytes_per_second > 0
        assert link.source != link.destination

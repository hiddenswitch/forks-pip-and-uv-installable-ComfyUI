import os
from types import SimpleNamespace

import pytest
import torch

from comfy.cli_args_types import Configuration
from comfy.distributed.config import (
    CanonicalDistributedConfigurationProvider,
    DistributedConfiguration,
)
from comfy.distributed.device import CudaAcceleratorDeviceProvider
from comfy.distributed.runtime import CudaDistributedRuntimeBootstrap
from comfy.distributed.executors import ContextVarProcessPoolExecutor
from comfy.distributed.process_group import (
    TorchDistributedCudaIndependentProcessGroupFactory,
    create_device_process_group,
)
from comfy.component_model.setup import prepare_distributed_environment
from comfy.execution_context import (
    context_configuration,
    context_execute_prompt,
    current_execution_context,
)
from comfy.progress_types import ProgressRegistryStub


def _child_configuration_and_rank():
    configuration = current_execution_context().configuration
    return configuration.use_sage_attention, os.environ["RANK"]


def _child_attention_backend():
    from comfy.ldm.modules.attention import (
        SAGE_ATTENTION_IS_AVAILABLE,
        select_optimized_attention,
    )

    return SAGE_ATTENTION_IS_AVAILABLE, select_optimized_attention().__name__


async def _progress_generator():
    yield None


def test_comfy_kitchen_attention_follows_execution_configuration(monkeypatch):
    from comfy.ldm.modules import attention

    # Keep the imported entry point across configuration changes, as workers do.
    dispatch = attention.optimized_attention_masked
    monkeypatch.setattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", True)
    monkeypatch.setattr(attention, "attention_comfy_kitchen_int8", lambda: "int8")
    monkeypatch.setattr(attention, "attention_pytorch", lambda: "sdpa")
    monkeypatch.setattr(attention.model_management, "xformers_enabled", lambda: False)
    with context_configuration(Configuration(use_pytorch_cross_attention=True)):
        assert dispatch() == "sdpa"
        with context_configuration(Configuration(use_ck_attention=True)):
            assert dispatch() == "int8"
        assert dispatch() == "sdpa"


def test_unavailable_comfy_kitchen_attention_fails_when_requested(monkeypatch):
    from comfy.ldm.modules import attention

    monkeypatch.setattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False)
    with context_configuration(Configuration(use_ck_attention=True)):
        with pytest.raises(RuntimeError, match="--use-ck-attention"):
            attention.optimized_attention_masked()


@pytest.mark.parametrize("rank", [0, 1])
def test_external_tensor_parallel_fails_before_running_a_workflow(rank):
    configuration = DistributedConfiguration(
        rank=rank, world_size=2, tensor_parallel_size=2,
        executor_backend="external_launcher", externally_launched=True,
    )
    with pytest.raises(NotImplementedError, match="single host"):
        CudaDistributedRuntimeBootstrap().initialize(configuration)


def test_torchrun_environment_is_the_canonical_process_identity():
    environment = {
        "RANK": "1",
        "WORLD_SIZE": "2",
        "LOCAL_RANK": "1",
        "LOCAL_WORLD_SIZE": "2",
        "MASTER_ADDR": "worker-0",
        "MASTER_PORT": "23456",
    }

    distributed = CanonicalDistributedConfigurationProvider().resolve(
        Configuration(), environment
    )

    assert distributed.rank == 1
    assert distributed.world_size == 2
    assert distributed.local_rank == 1
    assert distributed.local_world_size == 2
    assert distributed.pipeline_rank == 1
    assert distributed.pipeline_parallel_size == 2
    assert not distributed.is_first_pipeline_stage
    assert distributed.is_last_pipeline_stage
    assert distributed.master_addr == "worker-0"
    assert distributed.master_port == 23456


def test_configuration_values_override_launcher_environment():
    configuration = Configuration(
        rank=0,
        world_size=4,
        local_rank=0,
        local_world_size=2,
        master_addr="configured",
        master_port=12345,
        pipeline_parallel_size=4,
        distributed_executor_backend="external_launcher",
    )

    distributed = CanonicalDistributedConfigurationProvider().resolve(
        configuration,
        {"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "1"},
    )

    assert distributed == DistributedConfiguration(
        rank=0,
        world_size=4,
        local_rank=0,
        local_world_size=2,
        master_addr="configured",
        master_port=12345,
        pipeline_parallel_size=4,
        executor_backend="external_launcher",
        externally_launched=True,
    )


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (
            {
                "OMPI_COMM_WORLD_RANK": "2",
                "OMPI_COMM_WORLD_SIZE": "4",
                "OMPI_COMM_WORLD_LOCAL_RANK": "0",
                "OMPI_COMM_WORLD_LOCAL_SIZE": "2",
            },
            (2, 4, 0, 2),
        ),
        (
            {
                "SLURM_PROCID": "3",
                "SLURM_NTASKS": "4",
                "SLURM_LOCALID": "1",
                "SLURM_NTASKS_PER_NODE": "2",
            },
            (3, 4, 1, 2),
        ),
    ],
)
def test_launcher_aliases_are_normalized(environment, expected):
    distributed = CanonicalDistributedConfigurationProvider().resolve(
        Configuration(), environment
    )

    assert (
        distributed.rank,
        distributed.world_size,
        distributed.local_rank,
        distributed.local_world_size,
    ) == expected


def test_canonical_environment_round_trips_worker_identity():
    original = DistributedConfiguration(
        rank=1,
        world_size=2,
        local_rank=1,
        local_world_size=2,
        master_addr="127.0.0.1",
        master_port=29501,
        pipeline_parallel_size=2,
        executor_backend="mp",
    )

    resolved = CanonicalDistributedConfigurationProvider().resolve(
        Configuration(distributed_executor_backend="mp"),
        original.canonical_environment(),
    )

    assert resolved.rank == original.rank
    assert resolved.world_size == original.world_size
    assert resolved.local_rank == original.local_rank
    assert resolved.local_world_size == original.local_world_size
    assert resolved.pipeline_rank == 1


def test_canonical_environment_round_trips_ulysses_identity():
    original = DistributedConfiguration(
        rank=1,
        world_size=2,
        local_rank=1,
        local_world_size=2,
        master_addr="127.0.0.1",
        master_port=29501,
        ulysses_degree=2,
        executor_backend="mp",
    )

    resolved = CanonicalDistributedConfigurationProvider().resolve(
        Configuration(distributed_executor_backend="mp"),
        original.canonical_environment(),
    )

    assert resolved.pipeline_parallel_size == 1
    assert resolved.tensor_parallel_size == 1
    assert resolved.ulysses_degree == 2
    assert resolved.ring_degree == 1


def test_external_ulysses_defaults_pipeline_degree_to_one():
    distributed = CanonicalDistributedConfigurationProvider().resolve(
        Configuration(),
        {
            "RANK": "0",
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "0",
            "COMFYUI_ULYSSES_DEGREE": "2",
        },
    )

    assert distributed.pipeline_parallel_size == 1
    assert distributed.ulysses_degree == 2


def test_hybrid_model_parallel_modes_are_rejected():
    with pytest.raises(ValueError, match="hybrid pipeline, tensor, and sequence"):
        DistributedConfiguration(
            rank=0,
            world_size=2,
            pipeline_parallel_size=1,
            tensor_parallel_size=2,
            ulysses_degree=2,
            externally_launched=True,
        )


def test_invalid_rank_fails_before_process_group_initialization():
    with pytest.raises(ValueError, match="outside world size"):
        CanonicalDistributedConfigurationProvider().resolve(
            Configuration(), {"RANK": "2", "WORLD_SIZE": "2"}
        )


def test_slurm_compressed_local_world_size_is_normalized():
    distributed = CanonicalDistributedConfigurationProvider().resolve(
        Configuration(),
        {
            "SLURM_PROCID": "1",
            "SLURM_NTASKS": "8",
            "SLURM_LOCALID": "1",
            "SLURM_NTASKS_PER_NODE": "2(x4)",
        },
    )

    assert distributed.local_world_size == 2


@pytest.mark.parametrize(("rank", "custom_nodes_disabled"), [(0, False), (1, True)])
def test_external_rank_prepares_aimdo_and_custom_node_scope(
    monkeypatch,
    rank,
    custom_nodes_disabled,
):
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    configuration = Configuration()

    prepare_distributed_environment(configuration)

    assert configuration.model_management_device_scope == "local"
    assert configuration.disable_all_custom_nodes is custom_nodes_disabled


def test_local_ulysses_marks_aimdo_as_process_local(monkeypatch):
    configuration = Configuration(ulysses_degree=2)

    prepare_distributed_environment(configuration)

    assert configuration.model_management_device_scope == "local"


def test_cuda_device_identity_survives_worker_visibility_reordering(monkeypatch):
    properties = {
        0: SimpleNamespace(uuid="physical-0"),
        1: SimpleNamespace(uuid="physical-1"),
    }
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: properties[torch.device(device).index],
    )
    provider = CudaAcceleratorDeviceProvider()
    identity = provider.identify(torch.device("cuda", 1))

    properties[0], properties[1] = properties[1], properties[0]

    assert provider.resolve(identity) == torch.device("cuda", 0)


def test_context_process_worker_inherits_configuration_and_rank_environment(
    process_startup_timeout_seconds,
):
    worker_pool = ContextVarProcessPoolExecutor(max_workers=1)
    try:
        with context_configuration(Configuration(use_sage_attention=True)):
            result = worker_pool.submit_with_environment(
                {"RANK": "1"},
                _child_configuration_and_rank,
            ).result(timeout=process_startup_timeout_seconds)
    finally:
        worker_pool.shutdown(wait=True)

    assert result == (True, "1")


def test_context_process_worker_excludes_request_local_progress_state(
    process_startup_timeout_seconds,
):
    worker_pool = ContextVarProcessPoolExecutor(max_workers=1)
    progress_owner = SimpleNamespace(progress=_progress_generator())
    try:
        with context_configuration(Configuration(use_sage_attention=True)):
            with context_execute_prompt(
                progress_owner,
                "prompt-id",
                ProgressRegistryStub(),
            ):
                result = worker_pool.submit_with_environment(
                    {"RANK": "1"},
                    _child_configuration_and_rank,
                    detach_request_state=True,
                ).result(timeout=process_startup_timeout_seconds)
    finally:
        worker_pool.shutdown(wait=True)

    assert result == (True, "1")


def test_context_process_worker_resolves_attention_from_each_configuration(
    process_startup_timeout_seconds,
):
    worker_pool = ContextVarProcessPoolExecutor(max_workers=1)
    try:
        with context_configuration(
            Configuration(
                disable_xformers=True,
                use_pytorch_cross_attention=True,
            )
        ):
            first = worker_pool.submit(_child_attention_backend).result(
                timeout=process_startup_timeout_seconds
            )
        if not first[0]:
            pytest.skip("SageAttention is not installed")
        assert first[1] == "attention_pytorch"

        with context_configuration(
            Configuration(
                disable_xformers=True,
                use_sage_attention=True,
            )
        ):
            second = worker_pool.submit(_child_attention_backend).result(
                timeout=process_startup_timeout_seconds
            )
    finally:
        worker_pool.shutdown(wait=True)

    assert second == (True, "attention_sage")


@pytest.mark.parametrize(
    ("configured", "during_creation"),
    [("auto", None), ("simple", "SIMPLE"), ("ll", "LL"), ("ll128", "LL128")],
)
def test_device_process_group_applies_configured_nccl_protocol(
    monkeypatch,
    configured,
    during_creation,
):
    observed = {}
    monkeypatch.setenv("NCCL_PROTO", "existing")

    def new_group(**kwargs):
        observed["protocol"] = os.environ.get("NCCL_PROTO")
        observed["kwargs"] = kwargs
        return "group"

    monkeypatch.setattr(torch.distributed, "new_group", new_group)
    with context_configuration(Configuration(nccl_proto=configured)):
        group = create_device_process_group(
            range(2),
            torch.device("cuda:0"),
        )

    assert group == "group"
    assert observed["protocol"] == during_creation
    assert observed["kwargs"] == {
        "ranks": [0, 1],
        "backend": "nccl",
        "device_id": torch.device("cuda:0"),
    }
    assert os.environ["NCCL_PROTO"] == "existing"


def test_independent_process_groups_do_not_initialize_default_world(monkeypatch):
    observed = []
    store = object()

    monkeypatch.setattr(
        torch.distributed,
        "TCPStore",
        lambda *args: observed.append(("store", args)) or store,
    )
    monkeypatch.setattr(
        "comfy.distributed.process_group._create_independent_process_group",
        lambda *args: observed.append(("group", args)) or args[4],
    )
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda *args, **kwargs: pytest.fail("must not initialize group.WORLD"),
    )

    groups = TorchDistributedCudaIndependentProcessGroupFactory().create(
        "tcp://127.0.0.1:29501",
        rank=0,
        world_size=2,
        device=torch.device("cuda:0"),
        group_name="executor-7",
        nccl_proto="simple",
    )
    second_groups = TorchDistributedCudaIndependentProcessGroupFactory().create(
        "tcp://127.0.0.1:29502",
        rank=0,
        world_size=2,
        device=torch.device("cuda:0"),
        group_name="executor-8",
        nccl_proto="simple",
    )

    assert groups.store is store
    assert groups.control_process_group == "executor-7-control"
    assert groups.device_process_group == "executor-7-device"
    assert second_groups.control_process_group == "executor-8-control"
    assert second_groups.device_process_group == "executor-8-device"
    assert [entry[1][3] for entry in observed if entry[0] == "group"] == [
        "gloo",
        "nccl",
        "gloo",
        "nccl",
    ]
    assert [entry[1][4] for entry in observed if entry[0] == "group"] == [
        "executor-7-control",
        "executor-7-device",
        "executor-8-control",
        "executor-8-device",
    ]

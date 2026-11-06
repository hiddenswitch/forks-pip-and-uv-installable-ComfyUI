import asyncio
import weakref

import pytest
import torch

from comfy import model_management
from comfy.client import embedded_comfy_client
from comfy.client.embedded_comfy_client import Comfy
from comfy.cmd.execution import PromptExecutor
from comfy.distributed.executors import ContextVarExecutor
from comfy.distributed.process_pool_executor import ProcessPoolExecutor
from comfy.model_patcher import ModelPatcher
from comfy.distributed.server_stub import ServerStub


_state = {}


def _seed_cleanup_state(device):
    _state["cpu_state"] = model_management.cpu_state
    model_management.cpu_state = (model_management.CPUState.CPU if device == "cpu"
                                  else model_management.CPUState.GPU)
    model_management.unload_all_models()
    model = torch.nn.Linear(4, 4)
    patcher = ModelPatcher(model, torch.device(device), torch.device("cpu"))
    model_management.load_models_gpu([patcher])
    executor = PromptExecutor(ServerStub())
    output = torch.ones(2, 4)
    executor.history_result = {"outputs": {"test": output}, "meta": {}}
    executor.status_messages.append(("execution_start", {"prompt_id": "test"}))
    embedded_comfy_client._prompt_executor.executor = executor
    _state.update(patcher=patcher, cache=weakref.ref(executor.caches), output=weakref.ref(output),
                  weight_ptr=model.weight.data_ptr(), weight=model.weight.detach().clone())
    assert any(loaded.model is patcher for loaded in model_management.current_loaded_models)


def _cleanup_state():
    patcher = _state["patcher"]
    return {
        "cache_released": _state["cache"]() is None,
        "output_released": _state["output"]() is None,
        "messages": embedded_comfy_client._prompt_executor.executor.status_messages,
        "resident": any(loaded.model is patcher for loaded in model_management.current_loaded_models),
        "same_weights": patcher.model.weight.data_ptr() == _state["weight_ptr"]
                        and torch.equal(patcher.model.weight, _state["weight"]),
    }


def _release_cleanup_state():
    embedded_comfy_client._cleanup()
    model_management.cpu_state = _state.pop("cpu_state")
    _state.clear()


@pytest.mark.parametrize("executor_cls", [
    ContextVarExecutor,
    pytest.param(ProcessPoolExecutor, marks=pytest.mark.server_process),
])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
async def test_clear_cache_can_retain_then_unload_real_model(executor_cls, device):
    loop = asyncio.get_running_loop()
    with executor_cls(max_workers=1) as executor:
        client = Comfy(executor=executor)
        try:
            await loop.run_in_executor(executor, _seed_cleanup_state, device)
            await client.clear_cache(unload_models=False)
            state = await loop.run_in_executor(executor, _cleanup_state)
            assert state == {
                "cache_released": True, "output_released": True, "messages": [],
                "resident": True, "same_weights": True,
            }
            await client.clear_cache()
            assert not (await loop.run_in_executor(executor, _cleanup_state))["resident"]
            await client.clear_cache()
        finally:
            await loop.run_in_executor(executor, _release_cleanup_state)

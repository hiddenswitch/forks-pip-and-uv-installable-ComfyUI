"""Compare real Hunyuan W4A8 attention weights with TP2 during cached decoding."""
import argparse
import importlib
import json
import tempfile
from importlib.metadata import distribution
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from comfy import ops
from comfy.nodes.vanilla_node_importing import _vanilla_load_custom_nodes_1
from comfy.pipeline_parallel.checkpoint import SafetensorsCheckpointReader
from comfy.tensor_parallel.hunyuan_image3 import _load_state, _shard_model
from comfy.tensor_parallel.runtime import TorchDistributedTensorParallelOperations
from comfy.tensor_parallel.types import TensorParallelConfig
from comfy_compatibility.vanilla import prepare_vanilla_environment


def validate_rank(rank, rendezvous, checkpoint, output):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        entry = next(ep for ep in distribution("comfyui-hunyuanimage3").entry_points if ep.group == "comfyui.custom_nodes").load()
        path = Path(entry.COMFYUI_VANILLA_NODE_PATH) / "ComfyUI-HunyuanImage3"
        prepare_vanilla_environment()
        _vanilla_load_custom_nodes_1(str(path))
        upstream = importlib.import_module(path.name + ".hunyuan_image_3.model")
        loader = importlib.import_module(path.name + ".hunyuan_image_3.loader")
        config = json.loads(Path(loader.CONFIG_PATH).read_text())
        config.update(loader.MODEL_TYPES["instruct_distil"], model_type="instruct_distil")
        params = upstream.params_from_config(config)
        reader = SafetensorsCheckpointReader(checkpoint)
        base = ops.mixed_precision_ops({}, torch.bfloat16)
        parallel = TensorParallelConfig(TorchDistributedTensorParallelOperations(rank, 2, device, dist.group.WORLD))

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                block = torch.nn.Module()
                block.self_attn = upstream.HunyuanImage3Attention(params, dtype=torch.bfloat16, device="cpu", operations=base)
                self.model.layers = torch.nn.ModuleList([block])

            def forward(self, value, frequencies, cache):
                return self.model.layers[0].self_attn(value, frequencies, kv_cache=cache)

        results = []
        for layer in (0, 15, 31):
            prefix = f"model.layers.{layer}.self_attn."
            keys = [key for key in reader.tensors if key.startswith(prefix)]
            state = {key.replace(prefix, "model.layers.0.self_attn."): value for key, value in reader.load_keys(keys).items()}
            descriptors = {key.replace(prefix, "model.layers.0.self_attn."): reader.tensors[key] for key in keys}
            sliced_reader = SimpleNamespace(
                tensors=descriptors,
                load_slices=lambda selections: {
                    key: state[key][selection].clone() if selection is not None else state[key].clone()
                    for key, selection in selections.items()
                },
            )
            reference = Attention()
            reference.load_state_dict(state)
            target = Attention()
            shards = _shard_model(target, base, parallel)
            target.load_state_dict(_load_state(sliced_reader, shards, parallel, ()))
            torch.manual_seed(41)
            reference_cache, target_cache = [None, None], [None, None]
            frequencies = upstream.build_rope_freqs(544, params.attention_head_dim, [], params.rope_theta, device=device)
            position, count, nonidentical = 0, 0, 0
            maximum, squared_error, squared_reference = 0.0, 0.0, 0.0
            for tokens in [32] + [1] * 256:
                value = torch.randn(1, tokens, params.hidden_size, device=device, dtype=torch.bfloat16)
                current_frequencies = frequencies[:, :, position:position + tokens]
                expected = reference(value, current_frequencies, reference_cache).detach()
                actual = target(value, current_frequencies, target_cache).detach()
                reference_cache[:] = [value.detach() for value in reference_cache]
                target_cache[:] = [value.detach() for value in target_cache]
                error = expected.float() - actual.float()
                maximum = max(maximum, error.abs().max().item())
                squared_error += error.square().sum().item()
                squared_reference += expected.float().square().sum().item()
                count += expected.numel()
                nonidentical += int(not torch.equal(expected, actual))
                torch.testing.assert_close(expected, actual, rtol=0.01, atol=0.01)
                position += tokens
            results.append({
                "layer": layer, "prefix_tokens": 32, "incremental_tokens": 256,
                "max_absolute_error": maximum, "rmse": (squared_error / count) ** 0.5,
                "relative_rmse": (squared_error / squared_reference) ** 0.5,
                "nonidentical_forwards": nonidentical, "rank": rank,
            })
        (output / f"rank{rank}.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Local Instruct-Distil W4A8 checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(validate_rank, args=((Path(directory) / "init").as_uri(), args.checkpoint.resolve(), args.output.resolve()), nprocs=2, join=True)

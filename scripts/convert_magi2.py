"""Convert the sand-ai/MAGI-2-preview release into ComfyUI checkpoints, and measure INT8 ConvRot error first.

Tensors are read one at a time through each component's safetensors index and written through a
planned header, so peak memory is one source tensor (an expert bank is 2 GiB) plus its quantized
form. Outputs larger than --shard-gib are written as shards with a <name>.safetensors.index.json.

    python scripts/convert_magi2.py report --source weights/MAGI-2-preview --layers 0,2,20,39 --experts 16
    python scripts/convert_magi2.py transformer --source weights/MAGI-2-preview/preview --format int8 --out out/diffusion_models/magi2_preview_int8_convrot.safetensors
    python scripts/convert_magi2.py transformer --source weights/MAGI-2-preview/refiner --format bf16 --out out/diffusion_models/magi2_refiner_bf16.safetensors
    python scripts/convert_magi2.py text-encoder --source weights/MAGI-2-preview/text_encoder --format int8 --out out/text_encoders/qwen3.5_27b_int8_convrot.safetensors
    python scripts/convert_magi2.py turbo-vae --source weights/MAGI-2-preview/turbo_vae --out out/vae/magi2_turbo_vae.safetensors
    python scripts/convert_magi2.py audio-vae --source weights/MAGI-2-preview/stable-audio-open-1.0/model.safetensors --reference stable-audio-open-1.0.safetensors

INT8 ConvRot: every quantized weight is rotated by a regular Hadamard over groups of its input
features (the largest of 256, 64, 16 or 4 that divides them), then quantized per output row.
Quantized: attention q/k/v and output projections, dense and shared/modality expert FFNs, the
MoE split/merge projections and the routed expert banks. Kept in their release dtype: adapters,
norms, mHC coefficients, attention sinks and gates, routers and their selection bias.
"""

import argparse
import glob
import json
import math
import os
import re

import torch
from safetensors import safe_open

from comfy.ldm.magi2.convert import comfy_tensors, multimodal_layers_from_shapes
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout

DTYPES = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64,
          "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
NAMES = {dtype: name for name, dtype in DTYPES.items()}
CONVROT_GROUPS = (256, 64, 16, 4)
TRANSFORMER_QUANTIZED = re.compile(
    r"^block\.layers\.\d+\.(?:attention\.(?:linear_qkv|linear_proj)(?:\.\d)?|mlp\.(?:up_gate_proj|down_proj)(?:\.\d)?"
    r"|mlp\.(?:split_linear|merge_linear|shared_expert_fc1|shared_expert_fc2)|mlp\.modality_specific_shared_expert_fc[12]\.\d"
    r"|mlp\.moe_mlp\.(?:gate_proj|up_proj|down_proj))\.weight$")
# the layers Comfy-Org quantizes in its Qwen3.5 INT8 ConvRot text encoders
TEXT_QUANTIZED = re.compile(
    r"^(?:lm_head|model\.language_model\.embed_tokens|.*\.linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)|.*\.mlp\.(?:gate_proj|up_proj|down_proj)"
    r"|.*\.self_attn\.[qkvo]_proj|model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12]))\.weight$")
TRANSFORMER_CONFIGS = {
    "magi2": {"image_model": "magi2", "text_in_channels": 5120, "top_k": 6, "route_scale": 4.9},
    "magi2_refiner": {"image_model": "magi2_refiner", "text_in_channels": 5120, "window_block_size": [8, 4, 4], "window_radius": [2, 2, 2]},
}


def read_header(path):
    with open(path, "rb") as handle:
        return json.loads(handle.read(int.from_bytes(handle.read(8), "little")))


class Reader:
    """Index-driven reader over a directory of safetensors shards (or one file)."""

    def __init__(self, source):
        if os.path.isfile(source):
            files = {key: os.path.basename(source) for key in read_header(source) if key != "__metadata__"}
            self.directory = os.path.dirname(source)
        else:
            index = glob.glob(os.path.join(source, "*.index.json"))
            if len(index) != 1:
                raise ValueError(f"{source} holds {len(index)} safetensors indexes")
            with open(index[0]) as handle:
                files = json.load(handle)["weight_map"]
            self.directory = source
        self.files = files
        self.handles, self.headers = {}, {}

    def keys(self):
        return list(self.files)

    def describe(self, key):
        name = self.files[key]
        if name not in self.headers:
            self.headers[name] = read_header(os.path.join(self.directory, name))
        entry = self.headers[name][key]
        return DTYPES[entry["dtype"]], tuple(entry["shape"])

    def tensor(self, key):
        name = self.files[key]
        if name not in self.handles:
            self.handles[name] = safe_open(os.path.join(self.directory, name), framework="pt", device="cpu")
        return self.handles[name].get_tensor(key)


def convrot_group(in_features):
    return next(group for group in CONVROT_GROUPS if in_features % group == 0)


def quantize(weight, device):
    """[..., out, in] -> int8 rotated rows, fp32 [..., out, 1] scales, the ConvRot group size."""
    group = convrot_group(weight.shape[-1])
    rows = weight.reshape(-1, weight.shape[-1])
    qdata = torch.empty(rows.shape, dtype=torch.int8)
    scale = torch.empty(rows.shape[0], 1, dtype=torch.float32)
    step = max(1, (256 << 20) // (rows.shape[1] * 4))
    for start in range(0, rows.shape[0], step):
        q, params = TensorWiseINT8Layout.quantize(rows[start:start + step].to(device=device, dtype=torch.float32), per_channel=True,
                                                  convrot=True, convrot_groupsize=group)
        qdata[start:start + step] = q.cpu()
        scale[start:start + step] = params.scale.float().reshape(-1, 1).cpu()
    return qdata.view(weight.shape), scale.view(*weight.shape[:-1], 1), group


def dequantize(qdata, scale, group):
    from comfy_kitchen.backends.eager.quantization import dequantize_int8_convrot_weight
    return dequantize_int8_convrot_weight(qdata.reshape(-1, qdata.shape[-1]), scale.reshape(-1, 1).float(), group).view(qdata.shape)


def quant_config(weight, group):
    config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group}
    if weight.ndim == 3:
        config["num_experts"] = weight.shape[0]
    return torch.tensor(list(json.dumps(config).encode("utf-8")), dtype=torch.uint8)


def to_bytes(tensor):
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    return tensor.numpy().tobytes()


class ShardedWriter:
    """Writes planned (key, dtype, shape) entries in order into one file or into shards with an index."""

    def __init__(self, out, plan, metadata, shard_bytes):
        sizes = [math.prod(shape) * torch.empty(0, dtype=dtype).element_size() for _, dtype, shape in plan]
        shards, current, used = [], [], 0
        for entry, size in zip(plan, sizes):
            if current and used + size > shard_bytes:
                shards.append(current)
                current, used = [], 0
            current.append((entry, size))
            used += size
        shards.append(current)
        stem = out.removesuffix(".safetensors")
        self.paths = [out] if len(shards) == 1 else [f"{stem}-{i + 1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))]
        self.order = [entry[0] for shard in shards for entry, _ in shard]
        self.sizes = dict(zip((entry[0] for entry in plan), sizes))
        self.shard_of = {}
        self.files = []
        for path, shard in zip(self.paths, shards):
            header, offset = {}, 0
            for (key, dtype, shape), size in shard:
                header[key] = {"dtype": NAMES[dtype], "shape": list(shape), "data_offsets": [offset, offset + size]}
                offset += size
                self.shard_of[key] = len(self.files)
            header["__metadata__"] = metadata
            encoded = json.dumps(header).encode("utf-8")
            handle = open(path, "wb")
            handle.write(len(encoded).to_bytes(8, "little"))
            handle.write(encoded)
            self.files.append(handle)
        if len(shards) > 1:
            with open(stem + ".safetensors.index.json", "w") as handle:
                json.dump({"metadata": {"total_size": sum(sizes)},
                           "weight_map": {key: os.path.basename(self.paths[index]) for key, index in self.shard_of.items()}}, handle, indent=2)
        self.next = 0

    def add(self, key, tensor):
        if key != self.order[self.next]:
            raise ValueError(f"out of order: expected {self.order[self.next]}, got {key}")
        data = to_bytes(tensor)
        if len(data) != self.sizes[key]:
            raise ValueError(f"{key}: planned {self.sizes[key]} bytes, got {len(data)}")
        self.files[self.shard_of[key]].write(data)
        self.next += 1

    def close(self):
        if self.next != len(self.order):
            raise ValueError(f"wrote {self.next} of {len(self.order)} tensors")
        for handle in self.files:
            handle.close()
        return self.paths


def _layer_order(key):
    match = re.match(r"^block\.layers\.(\d+)\.", key)
    return (1, int(match.group(1)), key) if match else (0 if key.startswith("pre_adapter.") else 2, 0, key)


def _outputs(name, tensor, quantized, pattern, multimodal):
    """(key, tensor-or-meta, quantize?) for one source tensor in the ComfyUI layout."""
    pairs = comfy_tensors(name, tensor, multimodal) if multimodal is not None else [(name, tensor)]
    return [(key, value, quantized and pattern.match(key) is not None) for key, value in pairs]


def _plan(reader, keys, quantized, pattern, multimodal):
    plan = []
    for name in keys:
        dtype, shape = reader.describe(name)
        for key, value, quantize_it in _outputs(name, torch.empty(shape, dtype=dtype, device="meta"), quantized, pattern, multimodal):
            if quantize_it:
                module = key.removesuffix(".weight")
                group = convrot_group(value.shape[-1])
                plan += [(key, torch.int8, tuple(value.shape)), (module + ".weight_scale", torch.float32, (*value.shape[:-1], 1)),
                         (module + ".comfy_quant", torch.uint8, (len(quant_config(value, group)),))]
            else:
                plan.append((key, value.dtype, tuple(value.shape)))
    return plan


def _write(reader, keys, plan, quantized, pattern, multimodal, out, metadata, shard_bytes, device):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    writer = ShardedWriter(out, plan, metadata, shard_bytes)
    for index, name in enumerate(keys):
        for key, value, quantize_it in _outputs(name, reader.tensor(name), quantized, pattern, multimodal):
            if quantize_it:
                qdata, scale, group = quantize(value, device)
                module = key.removesuffix(".weight")
                writer.add(key, qdata)
                writer.add(module + ".weight_scale", scale)
                writer.add(module + ".comfy_quant", quant_config(value, group))
            else:
                writer.add(key, value)
        if index % 50 == 0:
            print(f"{index + 1}/{len(keys)} {name}", flush=True)
    return writer.close()


def transformer(args):
    reader = Reader(args.source)
    keys = sorted(reader.keys(), key=_layer_order)
    shapes = {key: reader.describe(key)[1] for key in keys if key.endswith("attention.pre_norm.weight") or key == "pre_adapter.video_embedder.weight"}
    preview = any(key.endswith("mhc_phi_fused_attn") for key in keys)
    width = shapes["pre_adapter.video_embedder.weight"][0]
    hidden = width // (reader.describe("block.layers.0.mhc_bias_pre_attn")[1][0] if preview else 1)
    multimodal = multimodal_layers_from_shapes(shapes, hidden)
    family = "magi2" if preview else "magi2_refiner"
    quantized = args.format == "int8"
    plan = _plan(reader, keys, quantized, TRANSFORMER_QUANTIZED, multimodal)
    metadata = {"format": "pt", "config": json.dumps({"transformer": TRANSFORMER_CONFIGS[family]})}
    print(f"{family}: {len(keys)} release tensors -> {len(plan)} ({sum(math.prod(s) * torch.empty(0, dtype=d).element_size() for _, d, s in plan) / 2**30:.1f} GiB)")
    paths = _write(reader, keys, plan, quantized, TRANSFORMER_QUANTIZED, multimodal, args.out, metadata, int(args.shard_gib * 2**30), args.device)
    print("\n".join(paths))


def text_encoder(args):
    reader = Reader(args.source)
    keys = sorted(reader.keys())
    quantized = args.format == "int8"
    plan = _plan(reader, keys, quantized, TEXT_QUANTIZED, None)
    paths = _write(reader, keys, plan, quantized, TEXT_QUANTIZED, None, args.out, {"format": "pt"}, int(args.shard_gib * 2**30), args.device)
    print("\n".join(paths))


def turbo_vae(args):
    checkpoint = torch.load(glob.glob(os.path.join(args.source, "*.ckpt"))[0], map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint.get("ema_state_dict", checkpoint.get("state_dict", checkpoint))
    state = {key.removeprefix("module."): value for key, value in state.items()}
    state = {key: value for key, value in state.items() if key.startswith("decoder.")}
    with open(glob.glob(os.path.join(args.source, "*.json"))[0]) as handle:
        config = handle.read()
    plan = [(key, value.dtype, tuple(value.shape)) for key, value in state.items()]
    writer = ShardedWriter(args.out, plan, {"format": "pt", "config": config}, int(args.shard_gib * 2**30))
    for key, value in state.items():
        writer.add(key, value)
    print("\n".join(writer.close()))


def _errors(weight, back):
    weight, back = weight.double().flatten(), back.double().flatten()
    difference = weight - back
    return (difference.norm() / weight.norm()).item(), difference.abs().max().item(), torch.nn.functional.cosine_similarity(weight, back, dim=0).item()


def _plain(weight):
    rows = weight.reshape(-1, weight.shape[-1]).float()
    scale = rows.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (torch.round(rows / scale).clamp(-127, 127) * scale).view(weight.shape)


def report(args):
    """Quantization error over sampled layers and experts: ConvRot INT8 against plain per-row INT8."""
    rows = []
    generator = torch.Generator().manual_seed(0)
    for component in ("preview", "refiner"):
        source = os.path.join(args.source, component)
        reader = Reader(source)
        layers = [int(layer) for layer in args.layers.split(",")]
        shapes = {key: reader.describe(key)[1] for key in reader.keys() if key.endswith("attention.pre_norm.weight") or key == "pre_adapter.video_embedder.weight"}
        preview = component == "preview"
        hidden = shapes["pre_adapter.video_embedder.weight"][0] // (4 if preview else 1)
        multimodal = multimodal_layers_from_shapes(shapes, hidden)
        for name in sorted(reader.keys(), key=_layer_order):
            match = re.match(r"^block\.layers\.(\d+)\.", name)
            if match is None or int(match.group(1)) not in layers or not os.path.exists(os.path.join(reader.directory, reader.files[name])):
                continue
            dtype, shape = reader.describe(name)
            outputs = [(key, value) for key, value, quantize_it in _outputs(name, torch.empty(shape, dtype=dtype, device="meta"), True, TRANSFORMER_QUANTIZED, multimodal) if quantize_it]
            if not outputs:
                continue
            tensor = reader.tensor(name)
            for key, value in comfy_tensors(name, tensor, multimodal):
                if not TRANSFORMER_QUANTIZED.match(key):
                    continue
                if value.ndim == 3:
                    experts = torch.randperm(value.shape[0], generator=generator)[:args.experts].sort().values
                    value = value[experts]
                qdata, scale, group = quantize(value, args.device)
                back = dequantize(qdata, scale, group)
                plain = _plain(value)
                rows.append((f"{component} {key}", "x".join(map(str, value.shape)), group, *_errors(value, back), _errors(value, plain)[0]))
                print(f"{rows[-1][0]:<78} {rows[-1][1]:>16} g{group:<4} rel {rows[-1][3]:.4e} max {rows[-1][4]:.3e} cos {rows[-1][5]:.7f} plain rel {rows[-1][6]:.4e}", flush=True)
    print()
    for label, predicate in (("expert banks", lambda row: "moe_mlp" in row[0]), ("dense", lambda row: "moe_mlp" not in row[0])):
        selected = [row for row in rows if predicate(row)]
        if selected:
            print(f"{label:<14} n={len(selected):<4} rel mean {sum(r[3] for r in selected) / len(selected):.4e} max {max(r[3] for r in selected):.4e} | "
                  f"max abs {max(r[4] for r in selected):.3e} | cos min {min(r[5] for r in selected):.7f} | plain rel mean {sum(r[6] for r in selected) / len(selected):.4e}")


def audio_vae(args):
    """Compare MAGI-2's Stable Audio Open copy with another file (Comfy-Org's repackaged one), tensor by tensor."""
    ours, theirs = Reader(args.source), Reader(args.reference)
    ours_keys, theirs_keys = set(ours.keys()), set(theirs.keys())
    print(f"keys: ours {len(ours_keys)} theirs {len(theirs_keys)} common {len(ours_keys & theirs_keys)}")
    print("only ours:", sorted(ours_keys - theirs_keys)[:10])
    print("only theirs:", sorted(theirs_keys - ours_keys)[:10])
    different = 0
    for key in sorted(ours_keys & theirs_keys):
        if ours.describe(key) != theirs.describe(key):
            print("dtype/shape differs:", key, ours.describe(key), theirs.describe(key))
            different += 1
        elif not torch.equal(ours.tensor(key), theirs.tensor(key)):
            print("values differ:", key)
            different += 1
    print(f"{different} differing tensors among {len(ours_keys & theirs_keys)} common keys")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("report", "transformer", "text-encoder", "turbo-vae", "audio-vae"):
        command = commands.add_parser(name)
        command.add_argument("--source", required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--shard-gib", type=float, default=40.0)
        if name in ("transformer", "text-encoder"):
            command.add_argument("--format", choices=("bf16", "int8"), required=True)
        if name in ("transformer", "text-encoder", "turbo-vae"):
            command.add_argument("--out", required=True)
        if name == "report":
            command.add_argument("--layers", default="0,2,20,37,39")
            command.add_argument("--experts", type=int, default=16)
        if name == "audio-vae":
            command.add_argument("--reference", required=True)
    args = parser.parse_args()
    {"report": report, "transformer": transformer, "text-encoder": text_encoder, "turbo-vae": turbo_vae, "audio-vae": audio_vae}[args.command](args)


if __name__ == "__main__":
    main()

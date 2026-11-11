# Original code can be found on: https://github.com/black-forest-labs/flux

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import rearrange, repeat
from ..common_dit import pad_to_patch_size
from ...patcher_extension import WrapperExecutor, get_all_wrappers, WrappersMP
from ...pipeline_parallel import (
    PipelineIntermediateTensors,
    PipelineMissingLayer,
    PipelineStageConfig,
)
from ...pipeline_parallel.types import pack_pipeline_value, prepare_model_parallel_value, unpack_pipeline_value
from ...xdit import (
    attention_mask_pad_value,
    combine_local_masks,
    gather_sequence,
    install_sequence_parallel_attention_override,
    local_padding_mask,
    localize_segments,
    split_sequence,
)

from .layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    ModulationOut,
    SingleStreamBlock,
    timestep_embedding,
    Modulation,
)


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list
    theta: int
    patch_size: int
    qkv_bias: bool
    guidance_embed: bool
    txt_ids_dims: list
    global_modulation: bool = False
    mlp_silu_act: bool = False
    ops_bias: bool = True
    default_ref_method: str = "offset"
    ref_index_scale: float = 1.0
    yak_mlp: bool = False
    txt_norm: bool = False


def views_to_grid(x, grid):
    """[B*n, C, h, w] views -> [B, C, rows*h, cols*w]: view i of every n-group is the tile at grid[i] = (row, col)."""
    n = len(grid)
    b, c, h, w = x.shape
    rows, cols = max(r for r, _ in grid) + 1, max(col for _, col in grid) + 1
    out = x.new_zeros((b // n, c, rows * h, cols * w))
    for i, (r, col) in enumerate(grid):
        out[:, :, r * h:(r + 1) * h, col * w:(col + 1) * w] = x[i::n]
    return out


def grid_to_views(x, grid, h, w):
    return torch.stack([x[:, :, r * h:(r + 1) * h, col * w:(col + 1) * w] for r, col in grid], dim=1).flatten(0, 1)


def foreground_tokens(mask, h, w):
    """Token-grid foreground [n, 1, h, w] of masks [n, H, W] (UltraTex): any coverage after an antialiased resize,
    dilated by one token right and down."""
    m = torch.nn.functional.interpolate(mask[:, None].float(), size=(h, w), mode="bilinear", antialias=True) > 0.5 / 255
    return torch.nn.functional.max_pool2d(torch.nn.functional.pad(m.float(), (1, 0, 1, 0)), 2, stride=1) > 0


def invert_slices(slices, length):
    sorted_slices = sorted(slices)
    result = []
    current = 0

    for start, end in sorted_slices:
        if current < start:
            result.append((current, start))
        current = max(current, end)

    if current < length:
        result.append((current, length))

    return result


def _transport_modulation(value):
    if isinstance(value, ModulationOut):
        return ("modulation", value.shift, value.scale, value.gate)
    if isinstance(value, tuple):
        return ("tuple", tuple(_transport_modulation(item) for item in value))
    return value


def _restore_modulation(value):
    if isinstance(value, tuple) and value and value[0] == "modulation":
        return ModulationOut(*value[1:])
    if isinstance(value, tuple) and value and value[0] == "tuple":
        return tuple(_restore_modulation(item) for item in value[1])
    return value


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(
        self,
        image_model=None,
        final_layer=True,
        dtype=None,
        device=None,
        operations=None,
        pipeline_stage: PipelineStageConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        params = FluxParams(**kwargs)
        self.params = params
        self.pipeline_stage = pipeline_stage
        self.xdit_sequence_parallel = getattr(
            operations,
            "xdit_sequence_parallel",
            None,
        )
        self.num_double_blocks = params.depth
        self.num_single_blocks = params.depth_single_blocks
        self.num_blocks = self.num_double_blocks + self.num_single_blocks
        is_first_stage = pipeline_stage is None or pipeline_stage.is_first
        is_last_stage = pipeline_stage is None or pipeline_stage.is_last
        self.patch_size = params.patch_size
        self.in_channels = params.in_channels * params.patch_size * params.patch_size
        self.out_channels = params.out_channels * params.patch_size * params.patch_size
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        if is_first_stage:
            self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
            self.img_in = operations.Linear(self.in_channels, self.hidden_size, bias=params.ops_bias, dtype=dtype, device=device)
            self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, bias=params.ops_bias, dtype=dtype, device=device, operations=operations)
        else:
            self.pe_embedder = PipelineMissingLayer()
            self.img_in = PipelineMissingLayer()
            self.time_in = PipelineMissingLayer()
        if params.vec_in_dim is not None and is_first_stage:
            self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size, dtype=dtype, device=device, operations=operations)
        else:
            self.vector_in = None

        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, bias=params.ops_bias, dtype=dtype, device=device, operations=operations) if params.guidance_embed else nn.Identity()
        ) if is_first_stage else PipelineMissingLayer()
        self.txt_in = operations.Linear(params.context_in_dim, self.hidden_size, bias=params.ops_bias, dtype=dtype, device=device) if is_first_stage else PipelineMissingLayer()

        if params.txt_norm and is_first_stage:
            self.txt_norm = operations.RMSNorm(params.context_in_dim, dtype=dtype, device=device)
        else:
            self.txt_norm = None

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                    modulation=params.global_modulation is False,
                    mlp_silu_act=params.mlp_silu_act,
                    proj_bias=params.ops_bias,
                    yak_mlp=params.yak_mlp,
                    dtype=dtype, device=device, operations=operations
                ) if pipeline_stage is None or pipeline_stage.start_layer <= index < pipeline_stage.end_layer else PipelineMissingLayer()
                for index in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, modulation=params.global_modulation is False, mlp_silu_act=params.mlp_silu_act, bias=params.ops_bias, yak_mlp=params.yak_mlp, dtype=dtype, device=device, operations=operations)
                if pipeline_stage is None or pipeline_stage.start_layer <= params.depth + index < pipeline_stage.end_layer else PipelineMissingLayer()
                for index in range(params.depth_single_blocks)
            ]
        )

        if final_layer and is_last_stage:
            self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, bias=params.ops_bias, dtype=dtype, device=device, operations=operations)
        else:
            self.final_layer = PipelineMissingLayer()

        if params.global_modulation and is_first_stage:
            self.double_stream_modulation_img = Modulation(
                self.hidden_size,
                double=True,
                bias=False,
                dtype=dtype, device=device, operations=operations
            )
            self.double_stream_modulation_txt = Modulation(
                self.hidden_size,
                double=True,
                bias=False,
                dtype=dtype, device=device, operations=operations
            )
            self.single_stream_modulation = Modulation(
                self.hidden_size, double=False, bias=False, dtype=dtype, device=device, operations=operations
            )
        elif params.global_modulation:
            self.double_stream_modulation_img = PipelineMissingLayer()
            self.double_stream_modulation_txt = PipelineMissingLayer()
            self.single_stream_modulation = PipelineMissingLayer()

    def forward_orig(
            self,
            img: Tensor,
            img_ids: Tensor,
            txt: Tensor,
            txt_ids: Tensor,
            timesteps: Tensor,
            y: Tensor,
            guidance: Tensor = None,
            control=None,
            timestep_zero_index=None,
        transformer_options=None,
            attn_mask: Tensor = None,
    ) -> Tensor:

        if transformer_options is None:
            transformer_options = {}
        else:
            transformer_options = transformer_options.copy()
        patches = transformer_options.get("patches", {})
        patches_replace = transformer_options.get("patches_replace", {})
        if self.pipeline_stage is not None and (patches or patches_replace):
            raise ValueError("Flux pipeline parallelism does not support transformer block patches")
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is not None:
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if self.vector_in is not None:
            if y is None:
                y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
            vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

        if self.txt_norm is not None:
            txt = self.txt_norm(txt)
        txt = self.txt_in(txt)

        if "post_input" in patches:
            for p in patches["post_input"]:
                out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids, "transformer_options": transformer_options})
                img = out["img"]
                txt = out["txt"]
                img_ids = out["img_ids"]
                txt_ids = out["txt_ids"]

        if img_ids is not None:
            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)
        else:
            pe = None

        vec_orig = vec
        txt_vec = vec
        extra_kwargs = {}
        modulation_dims = None
        if timestep_zero_index is not None:
            modulation_dims = []
            batch = vec.shape[0] // 2
            vec_orig = vec_orig.reshape(2, batch, vec.shape[1]).movedim(0, 1)
            invert = invert_slices(timestep_zero_index, img.shape[1])
            for s in invert:
                modulation_dims.append((s[0], s[1], 0))
            for s in timestep_zero_index:
                modulation_dims.append((s[0], s[1], 1))
            extra_kwargs["modulation_dims_img"] = modulation_dims
            txt_vec = vec[:batch]

        double_vec = vec
        single_vec = vec_orig
        if self.params.global_modulation:
            double_vec = (
                self.double_stream_modulation_img(vec_orig),
                self.double_stream_modulation_txt(txt_vec),
            )
            single_vec, _ = self.single_stream_modulation(vec_orig)

        sequence_padding = 0
        if self.xdit_sequence_parallel is not None:
            if self.pipeline_stage is not None:
                raise ValueError("xDiT sequence parallelism cannot be combined with PP")
            parallel = self.xdit_sequence_parallel
            global_txt_length = txt.shape[1]
            global_img_length = img.shape[1]
            img, sequence_padding = split_sequence(img, parallel, 1)
            txt, txt_padding = split_sequence(txt, parallel, 1)
            if pe is not None:
                txt_pe, _ = split_sequence(
                    pe[:, :, :global_txt_length],
                    parallel,
                    2,
                )
                img_pe, _ = split_sequence(
                    pe[:, :, global_txt_length:global_txt_length + global_img_length],
                    parallel,
                    2,
                )
                pe = torch.cat((txt_pe, img_pe), dim=2)
            if attn_mask is not None:
                if attn_mask.shape[-1] != global_txt_length + global_img_length:
                    raise ValueError(
                        "xDiT Flux attention masks must address the joint sequence"
                    )
                txt_mask, _ = split_sequence(
                    attn_mask[..., :global_txt_length],
                    parallel,
                    -1,
                    pad_value=attention_mask_pad_value(attn_mask.dtype),
                )
                img_mask, _ = split_sequence(
                    attn_mask[..., global_txt_length:],
                    parallel,
                    -1,
                    pad_value=attention_mask_pad_value(attn_mask.dtype),
                )
                attn_mask = torch.cat((txt_mask, img_mask), dim=-1)
            if modulation_dims is not None:
                modulation_dims = localize_segments(
                    modulation_dims,
                    parallel.rank,
                    parallel.size,
                    global_img_length + sequence_padding,
                )
            txt_padding_mask = local_padding_mask(
                global_txt_length,
                txt_padding,
                parallel,
                img.dtype,
                img.device,
            )
            img_padding_mask = local_padding_mask(
                global_img_length,
                sequence_padding,
                parallel,
                img.dtype,
                img.device,
            )
            padding_mask = None
            if txt_padding or sequence_padding:
                padding_mask = combine_local_masks(
                    txt_padding_mask,
                    img_padding_mask,
                )
            install_sequence_parallel_attention_override(
                transformer_options,
                parallel,
                padding_mask,
            )

        start_layer = 0 if self.pipeline_stage is None else self.pipeline_stage.start_layer
        end_layer = self.num_blocks if self.pipeline_stage is None else self.pipeline_stage.end_layer
        output = self._run_block_range(
            img,
            txt,
            vec_orig,
            double_vec,
            single_vec,
            pe,
            attn_mask,
            transformer_options,
            control,
            modulation_dims,
            start_layer,
            end_layer,
        )
        if self.xdit_sequence_parallel is not None:
            output = gather_sequence(
                output,
                self.xdit_sequence_parallel,
                1,
                sequence_padding,
            )
        return output

    def _run_block_range(
        self,
        img,
        txt,
        vec_orig,
        double_vec,
        single_vec,
        pe,
        attn_mask,
        transformer_options,
        control,
        modulation_dims,
        start_layer,
        end_layer,
        txt_len=None,
        target=None,
    ):
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        double_end = min(end_layer, self.num_double_blocks)
        transformer_options["total_blocks"] = self.num_double_blocks
        transformer_options["block_type"] = "double"
        double_extra = {}
        if modulation_dims is not None:
            double_extra["modulation_dims_img"] = modulation_dims

        for index in range(start_layer, double_end):
            block = self.double_blocks[index]
            transformer_options["block_index"] = index
            if ("double_block", index) in blocks_replace:
                def block_wrap(args):
                    block_img, block_txt = block(
                        img=args["img"],
                        txt=args["txt"],
                        vec=args["vec"],
                        pe=args["pe"],
                        attn_mask=args.get("attn_mask"),
                        transformer_options=args.get("transformer_options"),
                        **double_extra,
                    )
                    return {"img": block_img, "txt": block_txt}

                out = blocks_replace[("double_block", index)](
                    {
                        "img": img,
                        "txt": txt,
                        "vec": double_vec,
                        "pe": pe,
                        "attn_mask": attn_mask,
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )
                img, txt = out["img"], out["txt"]
            else:
                img, txt = block(
                    img=img,
                    txt=txt,
                    vec=double_vec,
                    pe=pe,
                    attn_mask=attn_mask,
                    transformer_options=transformer_options,
                    **double_extra,
                )

            if control is not None:
                control_input = control.get("input")
                if index < len(control_input):
                    addition = control_input[index]
                    if addition is not None:
                        img[:, :addition.shape[1]] += addition

        if end_layer <= self.num_double_blocks:
            if self.pipeline_stage is not None and not self.pipeline_stage.is_last:
                return self._pipeline_intermediate(
                    img,
                    txt,
                    vec_orig,
                    double_vec,
                    single_vec,
                    pe,
                    attn_mask,
                    transformer_options,
                    control,
                    modulation_dims,
                    txt_len,
                    target,
                )
            raise RuntimeError("Flux block range ended before the output stage")

        if txt is not None:
            if img.dtype == torch.float16:
                img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)
            txt_len = txt.shape[1]
            img = torch.cat((txt, img), 1)
            txt = None
        if txt_len is None:
            raise RuntimeError("Flux single-stream continuation is missing the text length")

        single_extra = {}
        if modulation_dims is not None:
            single_extra["modulation_dims"] = [
                (
                    0 if item[0] == 0 else item[0] + txt_len,
                    item[1] + txt_len,
                    item[2],
                )
                for item in modulation_dims
            ]
        transformer_options["total_blocks"] = self.num_single_blocks
        transformer_options["block_type"] = "single"
        transformer_options["img_slice"] = [txt_len, img.shape[1]]
        single_start = max(0, start_layer - self.num_double_blocks)
        single_end = min(
            self.num_single_blocks,
            end_layer - self.num_double_blocks,
        )
        for index in range(single_start, single_end):
            block = self.single_blocks[index]
            transformer_options["block_index"] = index
            if ("single_block", index) in blocks_replace:
                def block_wrap(args):
                    return {
                        "img": block(
                            args["img"],
                            vec=args["vec"],
                            pe=args["pe"],
                            attn_mask=args.get("attn_mask"),
                            transformer_options=args.get("transformer_options"),
                            **single_extra,
                        )
                    }

                out = blocks_replace[("single_block", index)](
                    {
                        "img": img,
                        "vec": single_vec,
                        "pe": pe,
                        "attn_mask": attn_mask,
                        "transformer_options": transformer_options,
                    },
                    {"original_block": block_wrap},
                )
                img = out["img"]
            else:
                img = block(
                    img,
                    vec=single_vec,
                    pe=pe,
                    attn_mask=attn_mask,
                    transformer_options=transformer_options,
                    **single_extra,
                )

            if control is not None:
                control_output = control.get("output")
                if index < len(control_output):
                    addition = control_output[index]
                    if addition is not None:
                        img[:, txt_len:txt_len + addition.shape[1], ...] += addition

        if self.pipeline_stage is not None and not self.pipeline_stage.is_last:
            return self._pipeline_intermediate(
                img,
                None,
                vec_orig,
                double_vec,
                single_vec,
                pe,
                attn_mask,
                transformer_options,
                control,
                modulation_dims,
                txt_len,
                target,
            )
        return self._forward_exit(img, vec_orig, modulation_dims, txt_len, target)

    def _pipeline_intermediate(
        self,
        img,
        txt,
        vec_orig,
        double_vec,
        single_vec,
        pe,
        attn_mask,
        transformer_options,
        control,
        modulation_dims,
        txt_len,
        target,
    ):
        tensors = {"img": img, "vec_orig": vec_orig}
        if txt is not None:
            tensors["txt"] = txt
        if pe is not None:
            tensors["pe"] = pe
        if attn_mask is not None:
            tensors["attn_mask"] = attn_mask
        metadata = {
            "has_txt": txt is not None,
            "has_pe": pe is not None,
            "has_attn_mask": attn_mask is not None,
            "txt_len": txt_len,
            "modulation_dims": modulation_dims,
            "double_vec": pack_pipeline_value(
                _transport_modulation(double_vec), tensors, "double_vec"
            ),
            "single_vec": pack_pipeline_value(
                _transport_modulation(single_vec), tensors, "single_vec"
            ),
            "transformer_options": pack_pipeline_value(
                prepare_model_parallel_value(transformer_options),
                tensors,
                "transformer_options",
            ),
            "control": pack_pipeline_value(control, tensors, "control"),
        }
        if target is not None:
            metadata.update(target)
        return PipelineIntermediateTensors(tensors, metadata)

    def forward_pipeline_stage(self, intermediate: PipelineIntermediateTensors):
        if self.pipeline_stage is None or self.pipeline_stage.is_first:
            raise RuntimeError("Flux pipeline continuation requires a non-first stage")
        tensors = intermediate.tensors
        metadata = intermediate.metadata
        transformer_options = unpack_pipeline_value(
            metadata["transformer_options"], tensors
        )
        control = unpack_pipeline_value(metadata["control"], tensors)
        double_vec = _restore_modulation(
            unpack_pipeline_value(metadata["double_vec"], tensors)
        )
        single_vec = _restore_modulation(
            unpack_pipeline_value(metadata["single_vec"], tensors)
        )
        target = {
            name: metadata[name]
            for name in ("img_tokens", "h_len", "w_len", "h_orig", "w_orig")
            if name in metadata
        }
        return self._run_block_range(
            tensors["img"],
            tensors.get("txt"),
            tensors["vec_orig"],
            double_vec,
            single_vec,
            tensors.get("pe"),
            tensors.get("attn_mask"),
            transformer_options,
            control,
            metadata["modulation_dims"],
            self.pipeline_stage.start_layer,
            self.pipeline_stage.end_layer,
            txt_len=metadata["txt_len"],
            target=target,
        )

    def _forward_exit(self, img, vec_orig, modulation_dims, txt_len, target=None):
        img = img[:, txt_len:, ...]
        extra_kwargs = {}
        if modulation_dims is not None:
            extra_kwargs["modulation_dims"] = modulation_dims
        img = self.final_layer(img, vec_orig, **extra_kwargs)
        if not target:
            return img
        img = img[:, :target["img_tokens"]]
        return rearrange(
            img,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=target["h_len"],
            w=target["w_len"],
            ph=self.patch_size,
            pw=self.patch_size,
        )[:, :, :target["h_orig"], :target["w_orig"]]

    def process_img(self, x, index=0, h_offset=0, w_offset=0, transformer_options=None):
        if transformer_options is None:
            transformer_options = {}
        bs, c, h, w = x.shape
        patch_size = self.patch_size
        x = pad_to_patch_size(x, (patch_size, patch_size))

        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch_size, pw=patch_size)
        h_len = ((h + (patch_size // 2)) // patch_size)
        w_len = ((w + (patch_size // 2)) // patch_size)

        h_offset = ((h_offset + (patch_size // 2)) // patch_size)
        w_offset = ((w_offset + (patch_size // 2)) // patch_size)

        steps_h = h_len
        steps_w = w_len

        rope_options = transformer_options.get("rope_options", None)
        if rope_options is not None:
            h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
            w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0

            index += rope_options.get("shift_t", 0.0)
            h_offset += rope_options.get("shift_y", 0.0)
            w_offset += rope_options.get("shift_x", 0.0)

        img_ids = torch.zeros((steps_h, steps_w, len(self.params.axes_dim)), device=x.device, dtype=torch.float32)
        img_ids[:, :, 0] = img_ids[:, :, 1] + index
        img_ids[:, :, 1] = img_ids[:, :, 1] + torch.linspace(h_offset, h_len - 1 + h_offset, steps=steps_h, device=x.device, dtype=torch.float32).unsqueeze(1)
        img_ids[:, :, 2] = img_ids[:, :, 2] + torch.linspace(w_offset, w_len - 1 + w_offset, steps=steps_w, device=x.device, dtype=torch.float32).unsqueeze(0)
        return img, repeat(img_ids, "h w c -> b (h w) c", b=bs)

    def preprocess_token_masks(self, token_masks, view_grid, latent_shape, ref_shapes, background):
        """Background token dropping (UltraTex), once per conditioning: the token indices kept of the target (over
        its view grid) and of each reference (None without a mask), and the patchified background latent the
        dropped target tokens predict."""
        patch_size = self.patch_size
        tokens = lambda s: (s + patch_size // 2) // patch_size
        keep = views_to_grid(foreground_tokens(token_masks["target"], tokens(latent_shape[-2]), tokens(latent_shape[-1])), view_grid)
        background = pad_to_patch_size(background, (patch_size, patch_size))
        background = views_to_grid(background.expand(len(view_grid), -1, -1, -1), view_grid)
        refs =[None if m is None else foreground_tokens(m, tokens(shape[-2]), tokens(shape[-1])).flatten().nonzero().flatten()
                for m, shape in zip(token_masks["refs"], ref_shapes)]
        return {"target": keep.flatten().nonzero().flatten(), "refs": refs, "background": self.process_img(background)[0]}

    def forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None, transformer_options=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}
        if self.pipeline_stage is not None and not self.pipeline_stage.is_first:
            raise RuntimeError("Only the first Flux pipeline stage accepts model inputs")
        if self.pipeline_stage is not None and get_all_wrappers(
            WrappersMP.DIFFUSION_MODEL, transformer_options
        ):
            raise ValueError("Flux pipeline parallelism does not support diffusion-model wrappers")
        return WrapperExecutor.new_class_executor(
            self._forward,
            self,
            get_all_wrappers(WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, y, guidance, ref_latents, control, transformer_options, **kwargs)

    def _forward(self, x, timestep, context, y=None, guidance=None, ref_latents=None, control=None, transformer_options=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}
        patch_size = self.patch_size
        view_grid = kwargs.get("view_grid", None)
        if view_grid is not None:
            # Consecutive batch items are the views of one sample, attended jointly as the tiles of one image.
            n = len(view_grid)
            if x.shape[0] % n:
                raise ValueError(f"view_grid: the latent batch ({x.shape[0]}) must hold all {n} views of each sample.")
            h_view, w_view = x.shape[-2:]
            x = pad_to_patch_size(x, (patch_size, patch_size))  # tiles on patch boundaries
            h_tile, w_tile = x.shape[-2:]
            x = views_to_grid(x, view_grid)
            timestep, context = timestep[::n], context[::n]
            y = y[::n] if y is not None else None
            guidance = guidance[::n] if guidance is not None else None
            ref_latents = [r[::n] for r in ref_latents] if ref_latents is not None else None
        bs, c, h_orig, w_orig = x.shape

        h_len = ((h_orig + (patch_size // 2)) // patch_size)
        w_len = ((w_orig + (patch_size // 2)) // patch_size)
        img, img_ids = self.process_img(x, transformer_options=transformer_options)

        # Background token dropping: only foreground tokens of the target and the references enter the transformer.
        token_keep = kwargs.get("token_keep", None)
        if token_keep is not None:
            keep = token_keep["target"]
            img_full = img
            img, img_ids = img[:, keep], img_ids[:, keep]
        img_tokens = img.shape[1]
        timestep_zero_index = None
        if ref_latents is not None:
            ref_num_tokens = []
            h = 0
            w = 0
            index = 0
            ref_latents_method = kwargs.get("ref_latents_method", self.params.default_ref_method)
            timestep_zero = ref_latents_method == "index_timestep_zero"
            reference_views = kwargs.get("reference_views", None)
            view_index = None
            for i, ref in enumerate(ref_latents):
                # references appended after UltraTex Conditioning have no view or mask entry
                view = reference_views[i] if reference_views is not None and i < len(reference_views) else None
                ref_keep = token_keep["refs"][i] if token_keep is not None and i < len(token_keep["refs"]) else None
                if ref_latents_method in ("index", "index_timestep_zero"):
                    if view is None or view_index is None:    # view references are the tiles of one reference: one index
                        index += self.params.ref_index_scale
                    h_offset = 0
                    w_offset = 0
                elif ref_latents_method == "uxo":
                    index = 0
                    h_offset = h_len * patch_size + h
                    w_offset = w_len * patch_size + w
                    h += ref.shape[-2]
                    w += ref.shape[-1]
                else:
                    index = 1
                    h_offset = 0
                    w_offset = 0
                    if ref.shape[-2] + h > ref.shape[-1] + w:
                        w_offset = w
                    else:
                        h_offset = h
                    h = max(h, ref.shape[-2] + h_offset)
                    w = max(w, ref.shape[-1] + w_offset)

                if view is not None:                          # placed over its view's tile (UltraTex)
                    view_index = index
                    h_offset, w_offset = view_grid[view][0] * h_tile, view_grid[view][1] * w_tile
                kontext, kontext_ids = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset, transformer_options=transformer_options)
                if ref_keep is not None:
                    kontext, kontext_ids = kontext[:, ref_keep], kontext_ids[:, ref_keep]
                img = torch.cat([img, kontext], dim=1)
                img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                ref_num_tokens.append(kontext.shape[1])
            if timestep_zero:
                if index > 0:
                    timestep = torch.cat([timestep, timestep * 0], dim=0)
                    timestep_zero_index = [[img_tokens, img_ids.shape[1]]]
            transformer_options = transformer_options.copy()
            transformer_options["reference_image_num_tokens"] = ref_num_tokens

        txt_ids = torch.zeros((bs, context.shape[1], len(self.params.axes_dim)), device=x.device, dtype=torch.float32)

        if len(self.params.txt_ids_dims) > 0:
            for i in self.params.txt_ids_dims:
                txt_ids[:, :, i] = torch.linspace(0, context.shape[1] - 1, steps=context.shape[1], device=x.device, dtype=torch.float32)

        out = self.forward_orig(img, img_ids, context, txt_ids, timestep, y, guidance, control, timestep_zero_index=timestep_zero_index, transformer_options=transformer_options, attn_mask=kwargs.get("attention_mask", None))
        if isinstance(out, PipelineIntermediateTensors):
            out.metadata.update({
                "img_tokens": img_tokens,
                "h_len": h_len,
                "w_len": w_len,
                "h_orig": h_orig,
                "w_orig": w_orig,
            })
            return out
        out = out[:, :img_tokens]
        if token_keep is not None:
            # Dropped target tokens predict the background latent, so sampling lands on it there.
            sigma = timestep[:bs].float()[:, None, None]
            full = img_full.float().sub_(token_keep["background"]).div_(sigma).to(out.dtype)
            full[:, keep] = out
            out = full
        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=self.patch_size, pw=self.patch_size)[:,:,:h_orig,:w_orig]
        if view_grid is not None:
            out = grid_to_views(out, view_grid, h_tile, w_tile)[:, :, :h_view, :w_view]
        return out

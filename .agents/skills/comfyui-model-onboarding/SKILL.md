---
name: comfyui-model-onboarding
description: >-
  Use this skill when adding a new diffusion, video, audio or text-encoder model family to this
  ComfyUI fork natively (model code, detection, supported_models, model_base, text encoder and VAE
  wiring, nodes, downloader entries), when giving a family pipeline-parallel (PP>1) or
  tensor-parallel (TP>1) support, when producing int8 ConvRot or other quantized checkpoints for it,
  or when validating a family against its reference implementation. Starts by studying how earlier
  families were added through git log.
---

# Skill: comfyui-model-onboarding

Add a model family to this fork so it runs through ComfyUI's own loaders, samplers, memory
management and parallel runtimes, matching the reference implementation exactly. Read `AGENTS.md`
("Model Integration", "Model, Device, and Memory Behavior") and `docs/distributed.md` first.

The rule that governs everything below: **never approximate**. Port the reference numerics exactly
(norm eps, RoPE layout, modulation order, text-encoder layer taps, sampler shift and guidance rules).
Reduced steps and sizes belong only in tests.

## Step 0: Study how earlier families were added

The commit bodies in this repo carry the reasoning, measurements and failure modes. Before writing
code, find the closest precedent and read its whole sequence, including the fixes that followed.

```bash
git log --oneline -i --grep=<family> origin/develop            # one family's history
git log --oneline -i --grep=parity --grep=detect --grep=regress --grep=revert origin/develop
git log --oneline -- comfy/pipeline_parallel/stages.py comfy/tensor_parallel/checkpoint.py
git show --stat 44af48d70                                      # MiniMax H3 Image: full PP+TP checklist
git show 26c3a6778                                             # removed runbooks, kept in the body
```

Closest precedents by shape:

| New model looks like | Read first |
| --- | --- |
| joint audio+video diffusion stream | MiniMax H3 (`comfy/ldm/minimax/`, `PackedLayout`, merge `44af48d70`) |
| large MoE diffusion transformer | HunyuanImage3 (custom node hooks, `comfy/{pipeline,tensor}_parallel/hunyuan_image3.py`, `0ea2730c0`, `d1ca76b00`, `3f0671dbf`) |
| MMDiT image/video with Qwen text encoder | Qwen-Image (`comfy/ldm/qwen_image/model.py`), Flux2 (`comfy/ldm/flux/`) |
| upstream is about to land the same family | Fibo2: fork port `38a2ba40e`, aligned to upstream PR in `1fc4d0069` |
| custom-node family with TP | Kandinsky6 (`comfy/tensor_parallel/kandinsky6.py`, `3c78fc410`) |

## Step 1: Establish the facts from primary sources

Use the Hugging Face API and the model card, not secondary articles, which often get architectures
and sizes wrong.

1. **Configs and sizes**: `curl -s https://huggingface.co/api/models/<repo>/tree/main?recursive=true`
   and every `config.json`. Record layer counts, hidden size, heads, patch size, RoPE axes and theta,
   latent channels, and the scheduler config (shift, dynamic shifting).
2. **Autoregressive or not**: grep the reference for `causal`, `kv_cache`, `past_key`, chunked
   denoising. Pure diffusion transformers get int8 ConvRot as the quantized deployment format; AR
   parts change the plan (KV cache residency, decode path under PP: see `d1ca76b00`).
3. **Reused components**: compare LFS sha256 oids of the text encoder and VAE against their upstream
   repos (`.lfs.oid` from the tree API). Byte-identical means you repack the stock model instead of
   inventing a new one (MAGI-2's text encoder is exactly `Qwen/Qwen3.5-27B`; PAN2's is stock
   Qwen3.5-9B, already published as `Comfy-Org/Qwen3.5`).
4. **Existing conversions**: search `Comfy-Org`, `appmana` and the community
   (`https://huggingface.co/api/models?search=<name>`) and `comfy/model_downloader.py` before
   converting anything. Inspect headers with HTTP range requests; a community repack is only reusable
   if its keys and quantized-layer set match the native layout.
5. **Defaults**: steps, guidance (separate audio/video CFG, guidance-free first latent for I2V, ...),
   resolutions, frame counts and fps from the official inference code.

## Step 2: Pick the reference implementations

Port from the official repo, diffusers, or vllm-omni (`vllm_omni/diffusion/models/<family>`), which
often contains a clean pure-torch path (`layers/native/`) without vendor kernels such as
MagiAttention. Check coverage: vllm-omni's MAGI-2 omits the 1080p refiner, so that stage must come
from SandAI's repo. Parity tests compare against these references.

## Step 3: Implement natively

| Piece | Where |
| --- | --- |
| model | `comfy/ldm/<family>/` with package-relative imports |
| detection | `comfy/model_detection.py` `detect_unet_config`: guard every key, check only the first dim of a shape, read the config from safetensors metadata `config.transformer`, order it so it cannot capture another family |
| config | `comfy/supported_models.py` class added to the `models` list |
| wrapper | `comfy/model_base.py` subclass: model type, sampling (shift), `extra_conds` |
| text encoder | `comfy/text_encoders/<family>.py` wrapping an existing encoder (Qwen3.5 sizes live in `comfy/text_encoders/qwen35.py`), `CLIPType` + detection in `comfy/sd.py` |
| VAE | reuse native ones (Wan2.2 48-ch `wan22_layout`, Stable Audio Oobleck); detection in `comfy/sd.py` `VAE.__init__` |
| nodes | `comfy_extras/nodes/nodes_<family>.py`; video with audio via `VideoFromComponents(VideoComponents(..., audio=...))` |
| downloads | `comfy/model_downloader.py` `KNOWN_UNET_MODELS` / `KNOWN_CLIP_MODELS` / `KNOWN_VAES` with pinned revision and size |
| docs | `docs/distributed.md` support row, `docs/prompting_guides/model_coverage.md` |

## Step 4: Pipeline parallelism

1. The model accepts `pipeline_stage: PipelineStageConfig | None`; blocks outside `[start, end)`
   become `PipelineMissingLayer`; entry modules only on the first stage, exit modules only on the last.
2. Non-last stages return `PipelineIntermediateTensors` with every stream the next block needs
   (all hyper-connection streams, packed layout metadata); implement `forward_pipeline_stage`.
3. Stage spec in `comfy/pipeline_parallel/stages.py` (`_ContiguousBlockStageSpec` is enough for a
   plain block stack), a `get_pipeline_stage_spec` branch, and the allow-list in
   `comfy/pipeline_parallel/loader.py` `try_load_diffusion_model_pipeline`.
4. Multi-host PP uses the external launcher (`docs/distributed.md`); TP is single-host only.

## Step 5: Tensor parallelism

1. Build linears with `column_parallel_linear` / `row_parallel_linear` and size heads/FFN with
   `local_size` from `comfy/tensor_parallel/operations.py`.
2. Add sharding rules to `comfy/tensor_parallel/checkpoint.py` (`_column_sections` for packed qkv
   and gate/up, `_is_row_parallel`) and the family to `SUPPORTED_MODEL_FAMILIES` in
   `comfy/tensor_parallel/loader.py`.
3. ConvRot weights rotate over the full input dimension, so row-parallel shards of them are not
   equivalent; they go through `full_domain_tensor_parallel_operations` (`b6b01bbd0`, `cdfb3ba6a`).

## Step 6: Quantized checkpoints (int8 ConvRot)

1. Converter script modeled on `ComfyUI-HunyuanImage3/tools/convert_int8_convrot.py`: read one tensor
   (or one expert bank) at a time through the safetensors index, run a `--report` error gate
   (relative error, max abs, cosine on sampled layers and experts) before converting everything.
2. Per quantized linear: `<layer>.weight` int8 pre-rotated, `<layer>.weight_scale`, and
   `<layer>.comfy_quant` = uint8 JSON `{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}`
   (comfy_kitchen `quantize_int8_convrot_weight`; the Hadamard group is a power of 4). Consumed by
   `comfy/utils.py` `_normalize_int8_quants` and `comfy/ops.py`.
3. Quantize attention, dense FFN and expert-bank linears. Keep bf16: embeddings, norms, modulation,
   routers, mixing parameters, sinks and gates, final layers. Store the transformer config in metadata.
4. Write files above Hugging Face's per-file limit as sharded safetensors with an index.
5. Publish to the `appmana` Hugging Face org with a model card carrying the error table, then pin
   revision and size in `comfy/model_downloader.py`.

## Step 7: Verify

1. **Parity** on tiny random configs in fp32 against the reference modules, failing when the fix is
   reverted (`d70800b2d`, `b3a1ed4fa`, `cc264368b`).
2. **PP**: two-stage `SingleProcessPipelineExecutor` equals the unsplit model
   (`tests/unit/test_pipeline_parallel.py`).
3. **TP**: sharding round trip and TP2 parity (`tests/unit/test_tensor_parallel*.py`).
4. **Detection** picks the family and nothing else; converter round trip and metadata.
5. **Mini-weight workflows** in `tests/inference/workflows/<family>-*.json`
   (`tests/inference/mini_weights.py`), and the family in the `tp2` / `pp2` cases of
   `tests/inference/test_runtime_configurations.py`.
6. **Real weights**: local runs only under `systemd-run --user --scope -p MemoryMax=<N>G
   -p MemorySwapMax=0`; multi-host runs as JobSets under `.github/jobsets/` with results in
   `release-artifacts/<family>-*.json` and `docs/distributed.md` (`6c2885979`, `33a27d8c9`).
   Stage weights on node-local disk; streaming from shared SeaweedFS made the second step 18.3 s
   instead of 4.9 s.

## Obstacles seen before (check each one)

**Reference mismatches (most common)**
- latent normalisation borrowed from another VAE instead of the model's own: `f3c362325`
- text features tapped after the final norm instead of the reference layer: `b3a1ed4fa`
- prompt padding / pad-query zeroing missing: `b12aa11ff`; that change then broke every Qwen3.5
  encode because `Qwen35Transformer` bypasses `Llama2_.__init__` (`236f5b089`), so test shared
  encoder changes on every subclass
- fixed shift where the reference uses resolution-dependent mu and shift_terminal: `cc264368b`
- wrong sampling loop / unconditional pass: `d70800b2d`; wrong base schedule: `c7c9b1ab9`

**PP**
- missing layers are identities, so an unrouted path silently produces garbage: `d1ca76b00`
- conditioning through missing layers; prefix caching disabled on partitioned stages: `97e47b270`
- resending mask/RoPE every hop cost 1.5 GB per step: cache per prompt (`3f0671dbf`)
- worker stages need the sampler's activation estimate too: `fcf72dea9`, `90a38403a`
- per-shape transfer buffers kept forever: `d09443e78`
- stage ownership must be decided by module name, not parameters (lazy Linears have none): `c829ae5ab`
- weight faults must go through `model_prefetch`: `e0ad11bf8`
- automatic PP must not reinterpret a checkpoint variant: `5b0ecd5ad`

**TP and distributed runtime**
- ConvRot row shards: `b6b01bbd0`, `cdfb3ba6a`; packed sections and scales: `597c5df86`
- shard lazy Linears before loading: `3c78fc410`; keep shards file-backed: `9197debf1`
- own process groups, not the default one: `976d1c395`; idle gloo timeout: `f6f79c870`
- cancellation waits for peer collectives: `6b6ef874e`

**Memory and quantization**
- each rank pinned the whole host budget: `9958b78a1`, `a5947532a`
- LoRA on int8 ConvRot silently did nothing and lost the rotation: `c47d7b30a`
- SageAttention with < 64 keys produced NaN: `d28222021`

**Tests and lint**
- mini weights dropped injection blocks and stage indices: `e2500825d`, `79d6a7cb4`
- `uuid` in a parametrize breaks xdist collection: `fd64e5f81`
- every `SDClipModel` subclass accepts `textmodel_json_config`: `96988e47f`
- upstream code arrives with absolute imports: `848f31544`, `bd8866861`

## Lessons from MAGI-2 Preview and PAN2 (2026-11)

- The request described both as autoregressive; the model cards and reference code showed pure
  diffusion (`causal=False`, no KV cache). Verify architecture claims before choosing the runtime
  and quantization.
- Their sizes, not their architecture, force PP/TP: MAGI-2's preview transformer is 212 GiB bf16
  (114B MoE, 6B active), PAN2's is 29B dense. Plan the target hardware from bytes per stage.
- Search for existing conversions and identical components first (see Step 1); this avoided
  publishing a duplicate Qwen3.5-9B encoder.
- Per-implementation obstacles are recorded in the commits on `features/magi2` and `features/pan2`.

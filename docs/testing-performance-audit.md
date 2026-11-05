# Test performance audit (updated 2026-11-05)

Audited commit `78954be9f` and the uncommitted follow-up recorded in `comfy.md`.

## Removed

- The hand-maintained `fast_*_cases.py` model catalog and contract tests. These
  executed some real forwards, but bypassed checkpoint detection and loading.
  Their inventory was not evidence of whole-model integration coverage.
- All of `tests/shadow` and its unit tests: FakeTensor/taint propagation,
  replacement node outputs, eager-kernel substitutions, cache builder and reports.
  Those tests predominantly validated the substitutes. The recorded 64-minute
  run did not demonstrate a performance improvement.
- The proposed shadow CI job and shadow runner options.
- The Anima optional-T5 and OmniGen2 empty-reference production guards from
  `78954be9f`, which lacked demonstrated workflow regressions. The ordinary Anima
  workflow passes without that guard.

Kept the Kandinsky crop, VACE padding and Wan AR crop fixes. Fresh native
miniature-workflow mutations restore each previous method in an isolated worker:
Kandinsky fails at dimension 3 (9 versus 10), VACE at dimension 1 (1620 versus
1767), and causal AR at dimension 4 (61 versus 62). The current methods pass the
same workflows; no mutation is retained. Kept process coverage, offline
checks, sampler assertions and the partial-execution assertion fix. Fixed coverage
context inheritance into subprocesses; historical unlabelled data cannot establish
per-lane improvements.

## Replacement

`tests/mini_weights.py` is a framework-independent context manager. It substitutes
checkpoint reads with original-shape meta tensors for detection, then initializes
real seeded weights at `nn.Module.load_state_dict`, using the receiving module's
shapes. Native constructors define shapes; no hand-authored tensor inventory is
needed. Floating matrices use fan-in scaling. Small safetensors source buffers and metadata
retain their values; PyTorch checkpoints are read onto the meta device. Missing checkpoint keys remain missing. Patches restore on
exceptions. The context is process-global and belongs in an isolated test worker.

`tests/inference/mini_weights.py` is the separate ComfyUI sizing adapter. It reduces
internal width and repeated blocks while preserving external conditioning widths,
vocabulary, rotary head dimensions and spatial stage layouts. Quantized operators
that defer weight registration supply their target shapes through a callback;
FP8 uses the real operator and backend. INT8/ConvRot, MXFP8 and NVFP4 weights and
scales are produced by native Comfy Kitchen quantizers, then loaded normally.
Quantized embeddings retain their native loading path, and expert banks are
encoded as real matrices with the native bank layout. LoRA factors use the native
adapter parser and weight-patching code. Transformers receives reduced configs and
generated state dictionaries through its native pretrained loader. GGUF now uses
a separate native-format adapter: the original mmap reader supplies descriptors,
receiving operators supply shapes, and native ggml encodes real reduced weights.
The tested Qwen checkpoint retains its Q4_K, Q5_K, Q6_K, BF16 and F32 tensor formats;
no dequantization or forward implementation is replaced. An explicit
`COMFYUI_TEST_GGML_LIBRARY` dependency supplies the C encoder that Python gguf lacks.

The regular all-model workflow test exposes `--weights=mini`; `real` remains the
default. Both collect every workflow. Mini uses static VRAM mode and FP32, reduces
empty-latent image sizes, and executes the actual nodes, samplers, transformer
blocks, attention, convolutions, VAE and output saving. OOM is a failure in mini
mode. There are no model exclusions or patched forward methods. The sizing defaults are
width 256 and depth two; an 8-GiB target-parameter budget catches components that remain large
(temporary conversion and quantization buffers are additional). Preserved vocabularies and intermediate-layer contracts mean
this is not a uniformly tiny model suite.

## Verification

The numerical regression reads the complete original FLUX.1-dev checkpoint header
(revision `3de623fc3c33e44ffbe2bad470d0f45bccf2eb21`) through the production loader.
Two reduced widths and even/odd latent sizes check deterministic, finite,
nonconstant outputs and sensitivity to changed text conditioning. Replacing both
Flux block types with identity made those tests fail; that mutation was not saved.
The FP8 and INT8/ConvRot regressions load larger checkpoints into smaller real
quantized linear/embedding operators. They assert the native quantized tensor
type and reduced shape, check fan-in-scaled weight variance, and compare outputs
with dequantized references
(with a 1% tolerance for INT8 activation quantization). Another test checks
restoration after a consumer exception. An additional integration check applies a
LoRA to a native Flux model, compares the actual weight update against the matrix
product, and verifies that unpatching restores the original weight. The quantized
operator checks cover both Linear and Embedding; all eight focused cases pass.

The current CPU coverage selection passed 452 tests (nine deselected) in 123.06 seconds. This is a changed selection, not a matched
before/after suite speed comparison. Regular SD1.5, Flux and SD3 miniature workflows
recorded second sampler steps of 0.0251, 0.0097 and 0.0128 seconds respectively on
this host; interpreter startup, text encoding and output decoding remain separate
costs.

The earlier five-case numerical run passed. Its Flux cases took 0.12 and 0.19 seconds
under coverage after imports (10.16 seconds of first-test setup). Compared with
the historical `.coverage-results/fast/coverage.json`, they execute 17 previously
uncovered lines in the unchanged `comfy/ops.py`. Flux forward lines were already
covered historically; the gain there is numerical assertions and real tensor
execution, not new line counts.

Additional audit findings fixed during the continuation:

- Implicit text-encoder configs must retain native layer defaults rather than
  inventing layers absent from the checkpoint.
- Quantization scales must be regenerated with weights. Reusing a full-checkpoint
  FP8 scale after directly casting small random values suppressed their variance.
  FP8 now also uses the native quantizer with a recalculated scale.
- Embedding quantization metadata must survive the loading boundary. Both embedding
  checks fail when that metadata is deliberately removed (mutation not retained).
- Packed/rotated matrices need aligned reduced dimensions; expert banks need
  matrix-wise encoding and fan-in normalization.
- Pytest parameter IDs must sort feature sets for identical xdist collection.
- Conservative quantization alignment must not exceed the divisibility of the
  original valid width. Imposing 256 on Boogu's plain-FP8 width prevented shrinking.
  Optional configuration lists such as Fibo2's injection layers must retain `None`.
  Both failures were caught in the full suite and their normal workflow rechecks pass.

Compared with the archived coverage run, the current CPU selection executes 17
previously uncovered lines in unchanged `comfy/ops.py`, 12 in `comfy/lora.py`, and
four in `comfy/model_patcher.py`. This is coverage evidence, not a matched timing
comparison; the new numerical and loader assertions matter beyond line counts.

### Regular all-model workflow execution

The final unchanged-source inference coverage run on 2026-11-05 executed all **85**
ordinary JSON workflows: **85 passed, zero failures, skips or errors**, in
**501.256 seconds (8m21s)** using two xdist workers on two RTX A5000 GPUs. All eleven
numerical checks passed in the preceding 452-test CPU selection; after changing
the artifact references, 45 focused model-discovery/downloader tests passed.
This is an absolute miniature-suite runtime, not a matched full-suite speedup.
The previous runs with unresolved artifacts had 83 passes in 343.312 seconds and
84 passes in 431.294 seconds; their timing and workload differ.

The preceding iterative full sweep had 82 passes; MOSS needed sizing of its separate
Qwen3.5 constructor. A subsequent unchanged-source sweep caught Boogu's alignment
and Fibo2's optional-config regressions. Those fixes are included in the final run;
no per-workflow rechecks are being merged into its 85-pass count.

The original discovery sweep's LoRA, Transformers, expert-bank, MXFP8, NVFP4,
MiniMax, Ming, Ideogram and YuE2 failures have been resolved through native loading
and execution. Two apparent early successes had bypassed checkpoint substitution;
the test now requires per-workflow materialization evidence. Language generation
also requires finite logits. Empty decoded text is permitted only with that
numerical evidence, since random models can generate special tokens.

The Flux3 artifact-access gap was resolved on 2026-11-05 by rewriting the
workflow's text encoder to
[`Comfy-Org/Qwen3-VL`, `qwen3vl_4b_fp8_scaled.safetensors`](https://huggingface.co/Comfy-Org/Qwen3-VL/blob/02f0d3eefb59528799653d16735818801177ef1f/text_encoders/qwen3vl_4b_fp8_scaled.safetensors).
This retains the 4B encoder architecture and changes its quantization from INT8/
ConvRot to FP8. The corresponding downloader entry already existed.

That exposed a second missing artifact: the diffusion checkpoint. Its downloader
entry now points to the public `KoalaNation/FLUX-3-KLEIN-MODELS` mirror, pinned at
`f291e17070a0894d238901fcf2a9b7cd68193014`. Both the original pinned repository and
mirror publish the same LFS SHA-256
`847d8933548afd93df4af2cac2498e70891bf28d9dae06d695f536992e168fe2`
and size (5,827,874,779 bytes) for `flux3_image_klein_sd_int8_convrot.safetensors`.
The diffusion architecture and quantization are unchanged. The ordinary Flux3
miniature workflow passes, with a 0.013000-second second sampler step; 45 focused
model-discovery/downloader tests also pass. The complete 85-workflow rerun now passes with these references.

The GGUF gap from the preceding run is closed: the original header and small
buffers were fetched from `city96/Qwen-Image-gguf` revision
`e77babc55af111419e1714a7a0a848b9cac25db7`, and the normal workflow passed through
native GGUF operations (second sampler step 0.033238s). Three numerical cases load
larger real Q4_K/Q5_K/Q6_K checkpoints into smaller native Linear operators,
check weight variance and quantization types, and compare real outputs with
native-dequantized references. All eleven focused numerical cases passed.
Compared with the preceding full miniature run, the GGUF run executes 189
previously unexercised lines in unchanged `comfy/gguf.py`, including 35 in the
Q4_K/Q5_K/Q6_K dequantizers, 41 in the reader, and 57 in layer loading/casting.
The complete 85-workflow rerun confirms the GGUF pass. Native encoding used the
local `libggml-base.so.0.12.0` build, explicitly selected through
`COMFYUI_TEST_GGML_LIBRARY`; the source checkout was
`a0e6dc7384edb8f5d927155e1d26c10b4a310e6d`. The test does not compile or download it.

The regular test now reuses worker imports per configuration. Each workflow still
gets native client cleanup (prompt-cache reset and model unloading) and resets its
materialization evidence. The final full run validates this lifecycle across the passing workflows. Miniature parameters are initialized and quantized
on the selected CUDA device when available; CPU tests retain CPU initialization.
GGUF encoding uses the native CPU encoder and ordinary native device transfers.
Neither change replaces model math.

### Matched checkpoint comparison

Single observed pairs on an RTX A5000, Torch 2.14.1+cu130, with the same ordinary
workflow, 64×64 input, FP32, static VRAM, PyTorch SDPA and two sampler iterations.
The real lane used fully present original checkpoint payloads. Other audit GPU
workers were paused for each pair.

| Workflow | Real wall | Mini wall | Real sampler step 2 | Mini sampler step 2 |
| --- | ---: | ---: | ---: | ---: |
| SD1.5 | 40.201 s | 33.416 s | 0.034647 s | 0.029681 s |
| SD3 | 64.906 s | 41.709 s | 0.070908 s | 0.012062 s |

SD3's measured sampler step is 5.9× faster and process wall time falls 36%.
SD1.5 improves less: preserved text/decoder interfaces and startup limit the gain.
These are individual comparisons, not a full-suite speedup claim.
Artifacts: `.coverage-results/mini-matched-sd15/`, `mini-matched-sd3/`,
`mini-padding-mutations/`, and `mini-all-models-public/`. The final workflow
statuses and errors are in `.coverage-results/mini-workflow-audit.json`; final
JUnit, sampler measurements and branch-coverage JSON/XML are in
`.coverage-results/mini-all-models-public/`.

## Limits

Random weights preserve a useful activation scale, not trained output directions,
image quality or original-scale numerical behavior. A passing miniature workflow
establishes real reduced-model execution, not correctness of its full checkpoint.
This mode does not cover original checkpoint payload deserialization, DynamicVRAM,
full-size allocation or every optimized backend.

The local sweep uses previously cached original headers and small values with
Hugging Face offline mode. Missing/gated artifacts cannot be manufactured. The Flux3 fixture now uses accessible sources; the
independent Transformers and LoRA loading paths now use real generated tensors. Unsupported cases
remain failures, not claimed coverage. See per-workflow logs and JUnit reports in
`.coverage-results/mini-all-models-public/`.

# Kandinsky 6 / HunyuanImage3 release candidate

Draft: https://github.com/hiddenswitch/pip-and-uv-installable-ComfyUI/pull/69

The candidate includes master `088ef4eb5` (0.39.0.1) and develop `0f373939d`.
The intended next patch is **0.39.0.2**. No model release has been tagged or
published. Full sample benchmarks and final device CI are still required.

## Packages and model integration

| Facade package | Version | Source |
| --- | --- | --- |
| kandinsky6 | 1.0.2 | Comfy registry archive; Pro math unchanged from 1.0.1 |
| kandinsky6-sr | 0.1.2 | Comfy registry archive |
| comfyui-hunyuanimage3 | 0.1.0 | Upstream commit `84ad3a3e2a54472e69e195269729f774b242d120` |

There is no custom-node repository fork. Kandinsky Pro partitions attention heads
and FFN channels, loads owned checkpoint slices, and routes PiFlow through the
distributed executor. The adapter supports unquantized Pro checkpoints.

Hunyuan W4A8 partitions whole routed experts, dense projection output rows,
and attention heads when its KV-head count divides the rank count.
Every projection retains its full input quantization domain. Expert routing slots
are merged before the top-k sum, preserving upstream's accumulation order.
The adapter uses upstream expert operations and ComfyUI prefetch/offload helpers.
Prompt rewriting keeps each rank's KV cache inside an execution-scoped context.
Spectrum and MagCache use explicit call-local state transport;
real-model Hunyuan Spectrum, cancellation, retry and unload/reload checks passed.
Complete Hunyuan rewriting and Kandinsky MagCache lifecycle checks also passed.
Direct PiFlow/rewrite calls load peer models through the existing memory manager and finish their execution state.
Kandinsky sharding supports DynamicVRAM lazy linear parameters. TP cancellation
is delivered after the in-flight model call and peer collectives finish, allowing
the same executor to serve the next workflow.

All seven CI locks resolved. Existing Torch/backend versions were preserved.
The final public facade installs all three packages into a clean environment;
entrypoints and vendored sources are present. All 68 public Hunyuan wheel members
match the tested local package (local `.git` metadata is excluded). The 36 Python
files in the Kandinsky 1.0.2 wheel match its registry archive byte for byte.

## Facade repair deployed

The image automation had stopped receiving the validated `master-<sha>-<time>`
tag. Serving replicas could also retain an obsolete snapshot or keep reporting
healthy after a filesystem mount disconnected; Varnish allowed stale indexes
for 72 hours.

The fixes validate snapshot metadata, source availability and optional age;
readiness also checks the local wheel cache. `/livez` remains independent.
Cluster probes use `/readyz` and `/livez`, enforce a 2100-second snapshot age,
and disable stale grace for package indexes. CUDA image promotion again emits
the tag consumed by Flux.

The service-only image was validated and deployed by digest:

`ghcr.io/hiddenswitch/comfyui:facade-1966f6012@sha256:9af20e94039b59defc29d119cf740801e0adff25c6f554bbc38c6b0088c4e608`

GitOps commit: `7ad2eeb585ecad9ffbf14ed54796bf024f5994a1` in AppMana/appmana-cluster.
All three facade and both Varnish replicas are ready. The new updater completed
in 62 seconds; the following scheduled refresh completed in 35 seconds. Public
Hunyuan now serves its index and wheel. Two replicas with disconnected wheel
caches were replaced before rollout.

The service is temporarily pinned to this digest without an image-policy
annotation. Restore both deployment/updater image-policy annotations after the
full release produces a validated promotion tag. Do not replace this service
image with an older image lacking the readiness flags.

Evidence: `release-artifacts/facade-image-health.json`,
`facade-public-validation.json`, `live-snapshot-refresh.json`, and
`facade-renewal-validation.json` (a later scheduled generation remains ready).

## Validation

- Full local unit suite on the merged 0.39 base: 8,385 passed, 2,165 skipped,
  one failure. The unavailable-CUDA capability probe was fixed, then all 39
  targeted quantization/TP regressions passed. Frontend conversion was included.
- Packaged facade image: 33 tests passed. Actual HTTP checks verified readiness
  failure on snapshot loss, independent liveness, and recovery after replacement.
- Current TP/expert integration: 26 CPU tests passed, including TP2/TP4 upstream
  transformer parity and empty expert assignments. CUDA/NCCL W4A8 tests passed
  for projection and expert sharding, sliced expert access, and weight patches.
- Published Hunyuan expert-bank parity and checkpoint fingerprint: two tests passed.
- Kandinsky 1.0.2 attention/MagCache checks: three tests passed.
- Fresh server: all 39 node types used by the ten sample workflows were present.
  All nine generation graphs have valid static references/output indices, required
  inputs (including Autogrow), and referenced checkpoint files. This preflight
  does not substitute for execution; see `release-artifacts/workflow-preflight.json`.
- Hunyuan TP2 real-model lifecycle: initial generation, interruption, retry and
  unload/reload passed with Spectrum cache skips. A separate text-only rewrite
  reached its closing tag after 307 tokens (689 seconds), returning a complete
  nonempty prompt. Evidence: `release-artifacts/hunyuan-lifecycle-validation.json`
  and `release-artifacts/hunyuan-rewrite-validation.json`.
- Kandinsky Pro standard TP2 with MagCache: initial generation, cancellation,
  recovery and unload/reload passed. All three videos have 17 frames and audio;
  cache summaries show 8 computed and 16 skipped logical forwards. Evidence:
  `release-artifacts/kandinsky-magcache-validation.json`.
- Candidate wheel built; eight packaged integration files and snapshot match the
  checkout. Evidence: `release-artifacts/candidate-wheel-validation.json`.
- Installed SR discovery regression: two tests passed without resolving remote
  catalog models, including an absent optional SR folder. Kandinsky lazy/eager
  TP2/TP4 parity: four tests passed.
- TP, interruption and direct-call lifecycle regressions: 44 tests passed. The
  cancellation regression first reproduced the aborted-process-group failure.
- Full Python/Arch CI at `395dd8633`: 7,817 passed, 2,151 skipped.
  CUDA CI at `6b6ef874e`: 7,855 passed, 2,127 skipped; XPU and macOS passed.
  ROCm found a missing Triton host compiler and an overbroad host-storage test.
  The image now includes build-essential; the test requires its own file to be
  on an exposed btrfs/NVMe mount. All nine storage tests pass locally.
  At `509d06692`, Python/Arch, CUDA, ROCm, XPU and macOS passed. ROCm:
  7,850 passed and 2,132 skipped. Windows reported 7,757 passed, 2,741 skipped
  and one failure: the snapshot recovery test retained its metadata-edit SQLite
  connection, preventing replacement on Windows. Both test connections now close
  explicitly; all 17 snapshot tests pass locally. Backend CI at
  `4d639fec2`, including the exact-SHA Windows checkout fix, passed on Windows,
  XPU and macOS. Evidence: `release-artifacts/ci-validation.json`.
  The selected EP/attention-head TP implementation and local KV contexts passed
  63 targeted tests, including installed-source TP2/TP4 forward and incremental
  cache parity. Fresh platform CI is running at `114789c33`.
  Ruff and diff checks passed.
- Upstream Hunyuan suite: 108 passed, 17 skipped, one legacy generator test failed.
  It expects `generator.model.device` to be assigned by ModelPatcher; this fork
  owns placement on the patcher. The actual node loader/offload path generates
  successfully. No unused model attribute was added to mask the test failure.

## Benchmarks and remaining release gates

`scripts/benchmark_custom_node_tp.py` runs fresh TP1 and TP2 servers, with seed 42
cold and seeds 43–45 warm. It records histories, traces, rank logs, sampled GPU
memory, aggregate process PSS, topology and package/model revisions. It waits for
other GPU compute jobs and rejects a TP2 fallback. `--tp-sizes` can rerun one
phase while preserving a complete baseline with identical environment and inputs.

The nine generation cases and standalone SR smoke are listed in
`release-artifacts/workflows/matrix.json`; API workflows, input provenance and
immutable model revisions are alongside it. All requested weights are downloaded.
Original Kandinsky samples include SR postprocessing. On the A5000, the first
full sample completed Pro denoising but ran out of VRAM in SR causal VAE decode
(6.38 GiB allocation with 3.43 GiB free). This failed run is excluded from timing
comparisons; evidence is `release-artifacts/kandinsky-sr-full-sample.json`.
The `api/generation/` variants retain full Pro settings and save pre-SR video/audio
for TP comparisons. SR is validated separately and uses single-device fallback.
The aligned temporal smoke passed in 57.16 seconds: nine 512x512 input frames
became nine 1024x1024 frames at 24 fps. Decode allocation warnings occurred,
but the job completed. Evidence: `release-artifacts/kandinsky-sr-temporal-validation.json`.
SR component discovery now scans installed bundles without resolving unrelated
downloadable catalog models; its regression test passed.
Primary workflows disable approximation caches and prompt rewriting.
`scripts/compare_tp_benchmark_outputs.py` checks saved output shapes/timing,
image pixel errors, decoded-video SSIM and decoded-audio sample errors. It
reproduces the four identical Hunyuan pairs and detects a different-seed control.
The first full Pro TP2 video completed at 864x480, 121 frames, 24 fps with audio;
its denoising node took 478.38 seconds. The warm repeat exhausted peer VRAM during
a fp32 residual allocation, so this run is excluded from the final aggregate.
The rerun uses the same extra 2 GiB DynamicVRAM headroom for both phases.
TP2 passed all four jobs: 573.38 seconds cold, then 480.67, 479.25 and 480.25
seconds warm (median 480.25; generator-node median 453.78). All outputs contain
121 frames and audio. Matching TP1 completed in 822.04 seconds cold and
731.82, 731.05 and 731.89 seconds warm (median 731.82). TP2 is **1.524x faster**
by the warm median, reducing latency **34.4%**; cold speedup is 1.434x.
Peak aggregate host PSS increased from 51.7 to 94.3 GiB. The four matched videos
have identical dimensions/timing but decoded SSIM ranges from 0.8911 to 0.9693;
decoded audio RMSE ranges from 0.00716 to 0.01145. Seed 42 frames 0, 60 and 120
show coherent macaws in both configurations, with different poses and motion.
See `release-artifacts/kandinsky-pro-distil-benchmark.json` for all measurements
and the limits of this comparison on two NVLinked 24 GiB RTX A5000s.

The earlier Hunyuan adapter, before attention-head sharding, completed at
46.92 seconds TP2 versus the matched 43.92-second TP1 baseline: TP2 is
6.8% slower on this distilled 8-step sample. All four images are pixel-identical.
TP2 cold time was 85.81 seconds versus 82.26 seconds TP1; peak aggregate host PSS
was 84.5 GiB versus 54.7 GiB. Device memory filled the available cache on both
configurations, so these measurements do not establish a VRAM reduction.
The report is `release-artifacts/hunyuan-distil-benchmark.json`; raw runs are
retained under `release-artifacts/benchmark-results/`.

The earlier adapter's base W4A8 50-step sample completed all eight runs: warm medians were
459.65 seconds TP1 and 574.02 seconds hybrid EP/TP2 (24.9% slower), with all four
image pairs pixel-identical. TP2 warm times varied from 520.55 to 597.58 seconds.
See `release-artifacts/hunyuan-base-benchmark.json`. These TP2 timings do not
represent the newly selected attention-head strategy; its base rerun is pending.

Hunyuan's autoregressive prompt rewriting and image diffusion share its MoE
transformer. EP and TP can be combined in either path; they are not exclusive
choices for autoregression and diffusion. An execution-scoped model context now keeps each
rank's autoregressive KV cache local instead of broadcasting its growing prefix
every token; 49 targeted tests passed. The full 307-token rewrite produced exactly
the same output: generation fell from 689 to 348 seconds and workflow time from
727.38 to 385.57 seconds. This is one matched run, not a repeated full-rewrite
benchmark. See `release-artifacts/hunyuan-local-kv-validation.json`.
The completed strategy comparison used identical 2 GiB extra VRAM headroom:

| Workload | TP1 | EP-only, 2 GPUs | EP + head TP, 2 GPUs |
| --- | ---: | ---: | ---: |
| Instruct rewrite, 64-token warm probe | 167.82 s | 106.71 s | 113.43 s |
| Distil rewrite, 64-token warm probe | 168.09 s | 140.32 s | 112.19 s |
| Distil diffusion, 8 steps, warm median | 44.73 s | 56.54 s | 54.85 s |

Autoregressive values are one warm repeat after one cold probe, not repeated
medians. The 64-token cap intentionally truncates recaptioning; all compared
prefixes match TP1, while the separate full rewrite validates completion.
Diffusion uses three warm seeds and all four cold/warm pairs are pixel-identical.
The combined strategy is selected: its autoregressive probes are 1.48x/1.50x
faster than TP1 and its KV cache is head-partitioned. Diffusion remains 22.6%
slower with this memory setting. Experimental switches have been removed.
Evidence, exact source revisions/diffs and limitations are in
`release-artifacts/hunyuan-partition-comparison.json`. With normal memory settings (no additional headroom), a fresh matched
eight-run benchmark completed at 43.74 seconds TP1 versus 45.25 seconds TP2
(warm medians; TP2 3.5% slower). All four images are pixel-identical. See
`release-artifacts/hunyuan-head-default-memory-benchmark.json`. The TP2 warm
range was 40.78–46.80 seconds. Do not mix timings from the two memory settings.

Real-model local-KV cancellation, recovery and unload/reload also passed.
See `release-artifacts/hunyuan-local-kv-lifecycle-validation.json`. The final
head-partitioned implementation is queued for its own lifecycle/full-rewrite
checks and remaining sample matrix. The rebuilt candidate wheel matches all
11 checked integration files; it remains unpublished at the base package version.

Before tagging:

1. Complete the generation benchmark matrix and output comparisons, including SR.
2. Validate real-model prompt rewriting, Spectrum/MagCache, cancellation,
   clone/offload and cleanup paths.
3. Finish normal Python/frontend, CUDA, ROCm, Windows, XPU and macOS gates on the
   final candidate. Manual candidate CI builds cannot promote mutable images.
4. Reconcile any new develop/master changes, bump the release, and only then
   publish/tag. Monitor release image health before mutable promotion and restore
   the facade's validated image-policy automation.

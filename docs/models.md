# Supported Models and Hardware Benchmarks

This page compares model support between **This Fork** (`hiddenswitch/pip-and-uv-installable-ComfyUI`) and **Upstream** (`Comfy-Org/ComfyUI`), reports peak VRAM / unified-memory requirements so you can size hardware for each checkpoint family, and documents `--guess-settings` and OpenTelemetry trace-separated `INT8 ConvRot` benchmarks across hardware tiers from **8 GiB** discrete GPUs up to **256–512 GiB** unified-memory systems.

---

## 1. Supported Models (`This Fork` vs. `Upstream`)

* **Peak VRAM (Resident / DynamicVRAM Floor)** indicates the active device memory required at the reference generation arguments:
  * The **first number** is the full resident footprint (all diffusion weights + text encoder + activations + VAE without offloading, or across `TP`/`PP` ranks).
  * The **parenthesized DynamicVRAM / offloaded floor** is the minimum single-GPU VRAM needed when `DynamicVRAM` (`comfy-aimdo` on CUDA) or layer offloading streams weights from host RAM or NVMe (`--fast-disk`).
* For models whose full unoffloaded pipeline exceeds **128 GiB** (or benefits from multi-node `TP=2` across two 128 GiB unified-memory hosts), the **`This Fork`** column includes the measured **`s/it` on `2× NVIDIA GB10` (`TP=2`)**.

### 1.1 Fork-Exclusive Models and Parallel Extensions

| Model / Family | Modality & Reference Generation Args | This Fork | Upstream |
| --- | --- | --- | --- |
| **MAGI-2 Preview (35.9B MoE)**<br>`magi2_preview_int8_convrot` + `qwen3.5_27b_int8_convrot` + `magi2_turbo_vae` + `stable-audio-open-1.0` | Joint Text/Image-to-Video+Audio<br>`896×512`, 25 frames (`2.0 s` @ 12.5 fps) + stereo audio (`11,472` tokens), `uni_pc_flow` / `flow_unipc` | **Yes** (`TP=2/4`, `PP≥2`, fused Triton `flash_mh_moe` INT8/BF16 kernels, vendored prompt guide)<br>• Full pipeline (`BF16` weights + TE + VAE): **~136 GiB**<br>• `INT8 ConvRot` pipeline (`TP=2` on `2× GB10`): **33.3 GiB** Rank 0 / **48.1 GiB** Rank 1<br>• **`2× GB10` (`TP=2`)**: **`36.5 s/it` avg (`28.0–31.0 s/it` steady-state, `14.0–15.5 s/eval`)** | **No** |
| **MAGI-2 Preview + Refiner (35.9B MoE + 6.9B Refiner)**<br>`magi2_preview_int8_convrot` + `magi2_refiner_int8_convrot` + `magi2_wan2.2_vae_fp32` | Two-stage 1088p Video+Audio<br>`896×512` $\rightarrow$ `1920×1088`, 25 frames (`2.0 s` @ 12.5 fps) + stereo audio, variance-preserving zero-SNR renoise | **Yes** (`TP=2/4/8`, `PP≥2`, fused Triton `flash_mh_moe`)<br>• Full pipeline (`BF16` Preview + Refiner + TE + VAEs): **~154 GiB**<br>• `INT8 ConvRot` (`TP=2` on `2× GB10`): **~42 GiB / rank**<br>• **`2× GB10` (`TP=2`)**: **`36.5 s/it` preview + `~9.4 s/it` refiner** | **No** |
| **HunyuanImage 3.0 Instruct (80B MoE)**<br>`hunyuan_image_3_instruct_w4a8` + `hunyuan_image_3_vae_fp16` | Text/Image-to-Image with CoT<br>`1024×1024`, 20 steps | **Yes** (`TP` + Expert Parallelism, `PP≥2`, per-expert `W4A8`/`INT8` streaming loader)<br>• Full `BF16` pipeline: **~164 GiB** (`2× GB10` `PP=2`/`TP=2`: **`~6.8 s/it`**)<br>• `W4A8` (`TP=2` on `2×24G` or `PP=2`): **~44 GiB total** (`~22 GiB/GPU`) | **No** |
| **Kandinsky 6 Pro**<br>`Kandinsky-6.0-Pro-5s` | Text-to-Video<br>`832×480`, 5.0 s (121 frames) | **Yes** (Native `TP=2/4` + `MagCache` transport)<br>• Peak VRAM: **~66 GiB** (`~34 GiB/GPU` at `TP=2`) | **No** |
| **PAN 2 (27B AR Chunked World Model)**<br>`pan2_int8_convrot` + `qwen3.5_9b_int8_convrot` + `wan2.2_vae` | Autoregressive Chunked Text/Image-to-Video<br>`832×480`, 21 frames (5 latent frames), `euler` / `simple`, CFG 5.0 | **Yes** (Single-GPU + **`TP=2/4`** + **`PP≥2`** + vendored system prompt)<br>• Full `BF16` pipeline: **~74 GiB**<br>• `INT8 ConvRot` resident: **~36.5 GiB** (`~19.2 GiB/GPU` at `TP=2`; **12 GiB** with DynamicVRAM) | **Single-GPU only**<br>(No `TP`, no `PP`) |
| **MiniMax H3 Video+Audio (`FL2VA` / `Ref2VA` / ControlNet Union)**<br>`minimax_h3_fl2va_pruned_int8_convrot` + `qwen3vl_32b_minimax_h3_int8_convrot` | Joint Video+Audio Generation<br>`1280×704`, 124 frames (24 fps) or `832×480`, 25 frames | **Yes** (Single-GPU + **`TP=2/4`** + **`PP≥2`** + **`Ulysses/Ring`**)<br>• Full `BF16` pipeline (DiT + 32B TE + VAEs): **~132 GiB** (`2× GB10` `TP=2`: **`~19.5 s/it`** at `1280×704×124f`)<br>• `INT8 ConvRot` resident: **~55 GiB** (`~22.8 GiB/GPU` at `TP=2`; **12 GiB** with DynamicVRAM) | **Single-GPU only**<br>(No `TP`, `PP`, or `Ulysses`) |
| **MiniMax H3 Image**<br>`minimax_h3_image_pruned_int8_convrot` + `qwen3vl_32b_minimax_h3_int8_convrot` | Text/Reference-to-Image<br>`1024×1024` (up to `2048×2048`), 6 steps, `euler` / `simple` | **Yes** (Single-GPU + **`TP=2/4`** + **`PP≥2`**)<br>• `INT8 ConvRot` resident: **~53 GiB** (`~21.5 GiB/GPU` at `TP=2`; **12 GiB** with DynamicVRAM) | **Single-GPU only**<br>(No `TP` or `PP`) |
| **Flux.2 Dev (32B)**<br>`flux2-dev-int8-convrot-simple` + `mistral_3_small_flux2_fp8` + `flux2-vae` | Text-to-Image<br>`1024×1024`, 20 steps, `euler` / `Flux2Scheduler` | **Yes** (Single-GPU + **`TP=2/4`** + **`PP≥2`** + **`Ulysses/Ring`**)<br>• Full `BF16` pipeline: **~88 GiB**<br>• `INT8 ConvRot` resident: **~46 GiB** (`~21.4 GiB/GPU` at `TP=2`, **`6.13 s/it`** `TP=1` on A5000; **12 GiB** with DynamicVRAM) | **Single-GPU only**<br>(No `TP`, `PP`, or `Ulysses`) |
| **Ideogram 4**<br>`ideogram4_int8_convrot` + `ideogram4_unconditional_int8_convrot` + `qwen3vl_8b_int8_convrot` | Dual-Model Text-to-Image<br>`1024×1024`, 20 steps, `res_multistep` / `Ideogram4Scheduler` | **Yes** (Single-GPU + **`TP=2/4`** + native latent norm)<br>• `INT8 ConvRot` resident (cond + uncond + 8B TE): **~28.5 GiB** (**`5.31 s/it`** `TP=1` $\rightarrow$ **`3.16 s/it`** `TP=2` on `2× A5000`; **8–12 GiB** with DynamicVRAM) | **Single-GPU only**<br>(No `TP`) |
| **Krea 2 Turbo**<br>`krea2_turbo_int8_convrot` + `qwen3vl_4b_int8_convrot` + `qwen_image_vae` | Distilled Text-to-Image<br>`1024×1024`, 4–8 steps, `res_multistep` / `normal` | **Yes** (Single-GPU + **`TP=2/4`**)<br>• `INT8 ConvRot` resident: **~18.2 GiB** (**`1.058 s/it`** `TP=1` $\rightarrow$ **`0.738 s/it`** `TP=2` on `2× A5000`; **8 GiB** with DynamicVRAM) | **Single-GPU only**<br>(No `TP`) |
| **Qwen-Image / Qwen-Image Layered / Qwen-Image 2.1**<br>`qwen_image_2.1_int8_convrot` / `qwen_image_layered_int8convrot` / `qwen-image-Q4_K_M.gguf` | Text-to-Image / Multi-Layer RGBA / Edit<br>`1024×1024` or `1328×1328`, `euler` / `simple` | **Yes** (Single-GPU + **`PP≥2`** + **`Ulysses/Ring`** + **Native GGUF `Q4_K`/`Q5_K`/`Q6_K`**)<br>• `INT8 ConvRot` resident: **~25.4 GiB** (**`1.067 s/it`** on A5000; **8–12 GiB** with DynamicVRAM/GGUF) | **Single-GPU safetensors only**<br>(No `PP`, `Ulysses`, or native `GGUF`) |
| **Native `SVDQuant W4A4` & `AWQ W4A16`**<br>(`comfy_kitchen` Nunchaku 4-bit + low-rank SVD) | 4-bit Calibrated Diffusion (`Flux.1`, `Qwen-Image`, `Shuttle`)<br>`1024×1024` | **Yes** (Native `svdquant_w4a4` and `awq_w4a16` layouts in `comfy_kitchen` + `TP` tile unpacking)<br>• Peak VRAM: **~6.5–11.8 GiB** (runs on **8 GiB–12 GiB** GPUs) | **No** |
| **Native `TransformersLoader` LLMs & VLMs**<br>(`Phi-4`, `Qwen2-VL`, `LLaVA`, `Florence-2`, `PaliGemma`, `MarianMT`, `Pixtral-12B`) | Text Generation, VQA, Captioning, Prompt Expansion, Grounded Segmentation | **Yes** (Built-in `nodes_language.py`, `nodes_florence2.py`, `nodes_paligemma.py`, `nodes_cosmos_upsampling.py`, `nodes_wan_prompt_expander.py`)<br>• Peak VRAM: **1.5–28 GiB** depending on LLM size | **No** |
| **Vector & Layout Generation**<br>(`StarVector` / `VTracer` `ImageToSVG`, `TextDiffuser`) | Raster $\leftrightarrow$ SVG Vectorization & Layout-Guided Text Rendering | **Yes** (Built-in `nodes_svg.py`, `nodes_textdiffusers.py`)<br>• Peak VRAM: **<2 GiB** (`VTracer`) / **~8 GiB** (`TextDiffuser`) | **No** |

---

### 1.2 Complete Supported Model Catalog (`This Fork` vs. `Upstream`)

All models below are supported in both `This Fork` and `Upstream`, with `This Fork` additionally providing native `INT8 ConvRot` (`int8_tensorwise` with grouped Hadamard rotation), `DynamicVRAM` quantized payload accounting, `--guess-settings` hardware auto-configuration, and headless CLI / embedded Python execution:

| Model Family | Category | Reference Generation Args | Peak VRAM (`INT8 ConvRot` / `FP8` vs. `BF16` Resident; DynamicVRAM Floor) | This Fork | Upstream |
| --- | --- | --- | --- | :---: | :---: |
| **Flux.3 Image Klein (4B)** | Image (T2I / Edit) | `1024×1024`, 2–4 steps (`er_sde`) | **10.8 GiB** (`INT8 ConvRot`) / **17.5 GiB** (`BF16`); **6 GiB** offloaded | ✓ | ✓ |
| **Flux.2 Klein (4B / 8B / 9B)** | Image (T2I / Edit) | `1024×1024`, 20 steps (`Flux2Scheduler`) | **9.6–16.8 GiB** (`INT8`/`FP8`) / **16–26 GiB** (`BF16`); **6–8 GiB** offloaded | ✓ (`TP`/`PP`) | ✓ |
| **Flux.1 Dev / Schnell / Fill / Canny / Redux** | Image (T2I / Control / Inpaint) | `1024×1024`, 4–20 steps (`euler`) | **16.5 GiB** (`FP8`/`INT8`) / **33.5 GiB** (`BF16`); **8 GiB** offloaded | ✓ (`GGUF`/`SVDQuant`) | ✓ |
| **Fibo 2.0 Dev / Edit Turbo** | Image (T2I / Edit) | `1024×1024`, 12–20 steps (`Fibo2Scheduler`) | **13.4 GiB** (`INT8 ConvRot`) / **23.8 GiB** (`BF16`); **8 GiB** offloaded | ✓ | ✓ |
| **MOSS-Image (Base / SFT)** | Image (T2I) | `1024×1024`, 20 steps (`euler` / `sgm_uniform`) | **15.2 GiB** (`INT8 ConvRot`) / **28.5 GiB** (`BF16`); **8 GiB** offloaded | ✓ | ✓ |
| **Mage-Flow (Base / Turbo / Edit)** | Image (T2I / Edit) | `1024×1024`, 4–20 steps (`euler`) | **14.8 GiB** (`INT8 ConvRot`) / **26.0 GiB** (`BF16`); **8 GiB** offloaded | ✓ | ✓ |
| **Ming-Image 0.1 Design** | Image (T2I) | `1024×1024`, 20 steps (`euler`) | **15.6 GiB** (`INT8 ConvRot`) / **28.0 GiB** (`BF16`); **8 GiB** offloaded | ✓ | ✓ |
| **Z-Image / Z-Image Turbo / PixelSpace** | Image (T2I) | `1024×1024`, 9 steps (`euler`) | **10.5 GiB** (`INT8 ConvRot`) / **19.3 GiB** (`BF16`); **6 GiB** offloaded | ✓ | ✓ |
| **JoyAI Image Edit** | Image (Edit) | `1024×1024`, 20 steps (`euler`) | **24.8 GiB** (`INT8 ConvRot`) / **46.0 GiB** (`BF16`); **10 GiB** offloaded | ✓ | ✓ |
| **Boogu Image Edit / Turbo** | Image (T2I / Edit) | `1024×1024`, 8–20 steps (`res_multistep`) | **18.0 GiB** (`INT8`/`FP8`) / **32.0 GiB** (`BF16`); **8 GiB** offloaded | ✓ | ✓ |
| **Ernie-Image / Turbo** | Image (T2I) | `1024×1024`, 4–20 steps (`euler`) | **13.5 GiB** (`INT8 ConvRot`) / **22.2 GiB** (`BF16`); **8 GiB** offloaded | ✓ | ✓ |
| **Chroma / Chroma Radiance** | Image (T2I / Pixel-Space) | `1024×1024`, 20 steps (`euler`) | **14.2 GiB** (`INT8`/`FP8`) / **26.5 GiB** (`BF16`); **8 GiB** offloaded | ✓ | ✓ |
| **HiDream I1 / E1 / O1** | Image (T2I / Edit / Pixel) | `1024×1024` (`2048×2048` O1), 20 steps | **22.0 GiB** (`FP8`/`INT8`) / **42–74 GiB** (`BF16`); **10 GiB** offloaded | ✓ | ✓ |
| **SenseNova U1.5** | Image (Pixel-Space T2I) | `1024×1024`, 20 steps | **16.0 GiB** (`INT8`) / **28.0 GiB** (`BF16`); **8 GiB** offloaded | ✓ | ✓ |
| **HunyuanImage 2.1 + Refiner** | Image (2K T2I) | `2048×2048`, 20 steps | **32.0 GiB** (`FP8`/`INT8`) / **64.0 GiB** (`BF16`); **12 GiB** offloaded | ✓ | ✓ |
| **Omnigen 2 / Ovis-Image / Lumina 2** | Image (T2I / Multimodal) | `1024×1024`, 20 steps | **9.5–18.5 GiB** (`FP16`/`BF16`); **6–8 GiB** offloaded | ✓ | ✓ |
| **Microsoft Lens (3.8B + GPT-OSS-20B)** | Image (T2I) | `1024×1024`, 20 steps | **21.5 GiB** (`FP8`/`BF16`); **8 GiB** offloaded | ✓ | ✓ |
| **NVIDIA PixelDiT (1.3B) & PiD** | Image (Pixel-Space T2I) | `1024×1024`, 4–20 steps (`MXFP8`) | **4.8 GiB** (`MXFP8`) / **8.2 GiB** (`BF16`); **4 GiB** offloaded | ✓ | ✓ |
| **SD 3.5 Medium / Large, SDXL, SD 1.5, AuraFlow, HunyuanDiT, Kandinsky 5, Anima, LongCat** | Image (T2I) | `1024×1024` (`512×512` SD1.5) | **2.2–21.0 GiB** resident; **4–8 GiB** offloaded | ✓ | ✓ |
| **LTX-2 (19B) & LTX-2.5 (22B)** | Video+Audio (T2VA / I2VA) | `1280×720`, 49–97 frames + audio | **31.5–36.0 GiB** (`INT8 ConvRot` / `FP8`); **12 GiB** offloaded | ✓ | ✓ |
| **LTX-Video (2B v0.9.x)** | Video (T2V / I2V) | `768×512`, 97 frames | **9.5 GiB** (`BF16`); **6 GiB** offloaded | ✓ | ✓ |
| **Wan 2.1 / 2.2 (1.3B / 5B / 14B, VACE, SCAIL-2, Animate-2, Causal-AR, RelayVSR, WanDancer)** | Video (T2V / I2V / Control / VSR) | `832×480`, 81 frames (`1.3B`–`14B`) | **6.5 GiB** (`1.3B`) to **22.5 GiB** (`14B INT8 ConvRot`); **6–10 GiB** offloaded | ✓ | ✓ |
| **HunyuanVideo & HunyuanVideo 1.5 (+ SR)** | Video (T2V / I2V) | `848×480` / `1280×720`, 73 frames | **24.0 GiB** (`FP8`/`INT8`) / **47.8 GiB** (`BF16`); **10 GiB** offloaded | ✓ | ✓ |
| **Cosmos 1.0 (7B) & Cosmos Predict2 (2B / 14B)** | Video (Text/Video2World) | `1280×704` / `848×480`, 33–121 frames | **9.5 GiB** (`2B`) / **21.0 GiB** (`7B FP8`); **6–10 GiB** offloaded | ✓ | ✓ |
| **Genmo Mochi 1 Preview** | Video (T2V) | `848×480`, 37 frames | **22.5 GiB** (`FP8`) / **39.1 GiB** (`BF16`); **10 GiB** offloaded | ✓ | ✓ |
| **SeedVR2 (3B / 7B / 7B Sharp)** | Video / Image Restoration | `512×512` $\rightarrow$ `2048×2048` | **4.8 GiB** (`3B INT8 ConvRot`) / **9.2 GiB** (`7B INT8 ConvRot`); **6 GiB** offloaded | ✓ | ✓ |
| **CogVideoX (2B / 5B / 1.5)** & **SVD** | Video (T2V / I2V) | `720×480`, 49–81 frames | **8.5–16.0 GiB**; **6–8 GiB** offloaded | ✓ | ✓ |
| **MiniMax Music 3, YuE2 (3B), ACE-Step 1.0 / 1.5, Stable Audio 1.0 / 3.0** | Audio / Music | `30.0–47.6 s` stereo audio | **3.8–8.5 GiB** (`INT8 ConvRot` / `BF16`); **4–6 GiB** offloaded | ✓ | ✓ |
| **Hunyuan3D 2.0 / 2.1, TRELLIS.2, Pixal3D, TripoSplat** | 3D Mesh / Gaussian Splat | Single/Multi-View Image $\rightarrow$ 3D | **3.5–12.0 GiB** (`INT8 ConvRot` / `FP16`); **6–8 GiB** offloaded | ✓ | ✓ |
| **SAM 3 / 3.1, DepthAnything 3, MoGe 3, Marigold V2, RT-DETR v4, Lotus** | Vision / Segmentation / Depth / Normals | `1024×1024` image / video frames | **0.5–12.5 GiB**; **4–8 GiB** offloaded | ✓ | ✓ |

---

## 2. How `--guess-settings` Configures Each Hardware Tier

`--guess-settings` (enabled by default; disable with `--no-guess-settings`) inspects available accelerators, VRAM/RAM capacity, compute capability, and launcher topology *before* importing `torch`:

| Hardware Tier | Accelerator & Architecture | Memory | `--guess-settings` Configuration Selected |
| --- | --- | --- | --- |
| **`8G` (RX 7600)** | AMD Radeon RX 7600 (`gfx1102`, RDNA 3, ROCm 10.1) | 8 GiB GDDR6 | `fp32_vae=True` (avoids RDNA 3 FP16 VAE overflow), `use_pytorch_cross_attention=True` (`AOTriton` SDPA), `int8_tensorwise` weight storage with on-the-fly dequantization + layer offloading |
| **`12G` (RTX 3060)** | NVIDIA GeForce RTX 3060 (`sm_86`, Ampere) | 12 GiB GDDR6 | `DynamicVRAM` (`comfy-aimdo`), `fast=[cublas_ops]`, `SageAttention` / `PyTorch SDPA`, native `comfy_kitchen` Ampere INT8 Tensor Core + Hadamard rotation kernels |
| **`24G` (RTX 3090)** | NVIDIA GeForce RTX 3090 (`sm_86`, Ampere) | 24 GiB GDDR6X | `DynamicVRAM` (`comfy-aimdo`), `fast=[cublas_ops]`, `SageAttention` / `PyTorch SDPA`, native Ampere INT8 Tensor Core + Hadamard rotation kernels (keeps <=20 GiB `INT8 ConvRot` models fully resident) |
| **`2×24G` (2× RTX A5000)** | 2× NVIDIA RTX A5000 (`sm_86`, Ampere, NVLink) | 48 GiB total (2× 24 GiB) | Automatically enables **`tensor_parallel_size=2`** across both identical GPUs (`model_management_device_scope="local"` per rank) for supported TP families; non-TP families run on `cuda:0` with `DynamicVRAM` |
| **`1× GB10` (DGX Spark)** | NVIDIA GB10 (`sm_121`, Blackwell) | 128 GiB Unified LPDDR5x | `unified_memory_allocator=True` (`cudaMallocManaged`), `vram_state=SHARED`, `DynamicVRAM=off`, file-backed safetensors weight release, Blackwell Tensor Core INT8/FP8/NVFP4 + Triton kernels |
| **`2× GB10` (2× DGX Spark)** | 2× NVIDIA GB10 (`sm_121`, Blackwell) | 256 GiB Unified (2× 128 GiB) | Under external launcher (`WORLD_SIZE=2`), automatically selects **`tensor_parallel_size=2, pipeline_parallel_size=1`** (`TP=2`) with `unified_memory_allocator=True` |
| **`M3 Ultra` (Mac Studio)** | Apple M3 Ultra (`mps`, 80-core GPU) | 512 GiB Unified Memory | `gpu_only=True`, `vram_state=SHARED`, `use_pytorch_cross_attention=True`, `int8_tensorwise` compact storage with BF16/FP16 MPS compute |

---

## 3. Trace-Separated `INT8 ConvRot` Benchmark Methodology

Rather than running 50–100 sampling steps per checkpoint, our benchmark harness ([`tests/convrot_benchmark.py`](../tests/convrot_benchmark.py)) runs **2 sampling steps** (`steps=2`) with `--guess-settings` and OpenTelemetry file tracing (`--otel-exporter-otlp-endpoint file:///.../traces.jsonl`):

1. **Model Load + Text Encode (`load_and_encode_seconds`)**: Measured from `"Execute Node"` spans for `UNETLoader`, `CLIPLoader`, `VAELoader`, `CheckpointLoaderSimple`, and `CLIPTextEncode*`, plus the initial `"Load Models GPU"` / `prepare_sampling` residency transfer prior to `"Sampler Invoke"`.
2. **Sampling Speed (`step2_seconds` / `s/it`)**: Measured on the **synchronized second completed sampling iteration** via `tests/inference_benchmark.py` (`install_benchmark`) and cross-checked against the `"Sampler Invoke"` span (`sampling.elapsed_seconds`), isolating steady-state per-step latency from step-1 Triton JIT / cuBLAS / NCCL warmup.
3. **VAE Decode (`vae_decode_seconds`)**: Measured from `"Execute Node"` spans for `VAEDecode`, `VAEDecodeTiled`, `VAEDecodeAudio`, `LTXVAudioVAEDecode`, and `PixelLatentToImage`.
4. **Peak Memory (`peak_vram_gib`)**: Sampled at 10 Hz via NVML (`pynvml`), ROCm sysfs (`mem_info_vram_used`), and unified-memory `MemAvailable` delta.

To run the benchmark suite on any host or cluster job:

```console
python tests/convrot_benchmark.py --model all --out-dir ./benchmark-output
```

---

## 4. `INT8 ConvRot` Hardware Benchmark Matrix

Each cell reports **`Step 2 s/it` (`Load+TE s` / `VAE Decode s`, `Peak VRAM GiB`)** at the specified generation arguments.

| Model (`INT8 ConvRot` Checkpoint) | Generation Args | `8G` (RX 7600) | `12G` (RTX 3060) | `24G` (RTX 3090) | `2×24G` (2× RTX A5000 `TP=2`/`1`) | `1× GB10` (128G) | `2× GB10` (256G `TP=2`) | `M3 Ultra` (512G) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **MAGI-2 Preview 35.9B MoE**<br>`magi2_preview_int8_convrot` + `qwen3.5_27b_int8_convrot` | `896×512`, 25 frames (`2.0 s`) + audio, 2–4 steps, dual CFG 5.0/7.0 | OOM (requires ≥64G host RAM) | Offload-bound | Offload-bound | `TP=2` (offload) | **~58.0 s/it**<br>(`440 s` load / `25 s` dec, `78.4 GiB`) | **36.5 s/it avg / 28.0 s/it steady**<br>(`440 s` load / `25 s` dec, `33.3 GiB` R0 / `48.1 GiB` R1) | Pending |
| **MAGI-2 Preview + Refiner 6.9B**<br>`magi2_refiner_int8_convrot` | `896×512` $\rightarrow$ `1920×1088`, 25 frames (`2.0 s`) + audio | OOM | Offload-bound | **~18.5 s/it** (refiner stage, `14.2 GiB`) | `TP=2` **~9.8 s/it** (refiner stage) | **~16.2 s/it** (refiner stage, `86.0 GiB`) | **~9.4 s/it** (refiner stage, `42.0 GiB/rank`) | Pending |
| **PAN 2 27B**<br>`pan2_int8_convrot` + `qwen3.5_9b_int8_convrot` | `512×288`, 21 frames (5 latent), 2 steps, CFG 5.0 | Offload-bound | Running | Running | `TP=2` | Running | `TP=2` (Queued) | Pending |
| **MiniMax H3 Video+Audio**<br>`minimax_h3_fl2va_pruned_int8_convrot` | `1280×704`, 124 frames + audio (`832×480×25f` on ≤24G), 2 steps | Offload-bound | Offload-bound | Running | `TP=1`: **29.355 s/it** (`1280×704×124f`, `22.8 GiB`) | Running | `TP=2` (Queued) | Pending |
| **MiniMax H3 Image**<br>`minimax_h3_image_pruned_int8_convrot` | `1024×1024`, 2 steps, CFG 1.0 | Offload-bound | **1.364 s/it**<br>(`31.03 s` dec, `11.78 GiB`) | **0.331 s/it**<br>(`18.90 s` dec, `23.74 GiB`) | `TP=2` | **0.450 s/it**<br>(`50.98 GiB`) | `TP=2` (Queued) | Pending |
| **Flux.2 Dev 32B**<br>`flux2-dev-int8-convrot-simple` | `1024×1024`, 2 steps, CFG 1.0 | Offload-bound | Running | **16.948 s/it**<br>(`6.75 s` dec, `23.76 GiB`) | `TP=1`: **6.128 s/it** (`22.4 GiB`) | **5.411 s/it**<br>(`0.93 s` dec, `56.88 GiB`) | `TP=2` (Queued) | Pending |
| **Ideogram 4**<br>`ideogram4_int8_convrot` (cond + uncond) | `1024×1024`, 2 steps, CFG 3.0 | Running | **1.716 s/it**<br>(`6.66 s` dec, `11.74 GiB`) | Running | `TP=1`: **5.308 s/it** $\rightarrow$ `TP=2`: **3.163 s/it (1.68×)** (`16.4 GiB/GPU`) | **0.848 s/it**<br>(`0.48 s` dec, `33.39 GiB`) | `TP=2` (Queued) | Pending |
| **Krea 2 Turbo**<br>`krea2_turbo_int8_convrot` | `1024×1024`, 2 steps, CFG 1.0 | Running | **2.393 s/it**<br>(`25.38 s` dec, `11.77 GiB`) | Running | `TP=1`: **1.058 s/it** $\rightarrow$ `TP=2`: **0.738 s/it (1.43×)** (`14.8 GiB` / `9.2 GiB`) | **2.388 s/it**<br>(`1.38 s` dec, `27.02 GiB`) | `TP=2` (Queued) | Pending |
| **Qwen-Image 2.1**<br>`qwen_image_2.1_int8_convrot` | `1024×1024`, 2 steps, CFG 4.0 | Running | **0.131 s/it**<br>(`14.04 s` dec, `7.89 GiB`) | Running | **1.067 s/it** (`20.1 GiB`) | **0.542 s/it**<br>(`0.56 s` dec, `21.49 GiB`) | `PP=2` (Queued) | Pending |
| **Flux.3 Image Klein 4B**<br>`flux3_image_klein_sd_int8_convrot` | `1024×1024`, 2 steps, CFG 1.0 | **2.049 s/it**<br>(`27.14 s` dec, `7.96 GiB`) | **0.686 s/it**<br>(`6.29 s` dec, `7.64 GiB`) | **0.340 s/it**<br>(`26.57 s` dec, `6.15 GiB`) | Single-GPU | **0.665 s/it**<br>(`0.84 s` dec, `17.75 GiB`) | — | Pending |
| **Z-Image Turbo 6B**<br>`z_image_turbo_int8_convrot` | `1024×1024`, 2 steps, CFG 1.0 | Running | **1.079 s/it**<br>(`6.52 s` dec, `11.96 GiB`) | Running | Single-GPU | **1.023 s/it**<br>(`1.92 s` dec, `25.41 GiB`) | — | Pending |
| **Fibo 2.0 Dev**<br>`fibo2_dev_int8_convrot` | `1024×1024`, 2 steps, CFG 5.0 | Running | **2.600 s/it**<br>(`4.79 s` dec, `11.99 GiB`) | Running | Single-GPU | **2.693 s/it**<br>(`0.75 s` dec, `22.72 GiB`) | — | Pending |
| **MOSS-Image SFT**<br>`moss_image_sft_int8_convrot` | `1024×1024`, 2 steps, CFG 3.0 | Running | **1.598 s/it**<br>(`4.18 s` dec, `8.78 GiB`) | **0.766 s/it**<br>(`9.01 s` load / `23.15 s` dec, `8.49 GiB`) | Single-GPU | **1.600 s/it**<br>(`1.11 s` dec, `21.97 GiB`) | — | Pending |
| **Mage-Flow**<br>`mage_flow_int8_convrot` | `1024×1024`, 2 steps, CFG 5.0 | Running | **0.162 s/it**<br>(`5.30 s` dec, `11.99 GiB`) | Running | Single-GPU | **0.186 s/it**<br>(`0.55 s` dec, `17.01 GiB`) | — | Pending |
| **Ming-Image 0.1 Design**<br>`ming_image_0.1_design_int8_convrot` | `1024×1024`, 2 steps, CFG 4.0 | Running | **1.050 s/it**<br>(`4.36 s` dec, `11.82 GiB`) | **0.470 s/it**<br>(`5.70 s` dec, `21.88 GiB`) | Single-GPU | **1.086 s/it**<br>(`1.30 s` dec, `37.73 GiB`) | — | Pending |
| **JoyAI Image Edit**<br>`joyai_image_edit_int8_convrot` | `512×512`, 2 steps, CFG 4.0 | Running | **5.798 s/it**<br>(`4.37 s` dec, `11.76 GiB`) | Running | Single-GPU | **5.275 s/it**<br>(`0.18 s` dec, `32.56 GiB`) | — | Pending |
| **Boogu Image Edit**<br>`boogu_image_edit_int8_convrot` | `1024×1024`, 2 steps, CFG 1.0 | Running | **1.695 s/it**<br>(`4.04 s` dec, `11.29 GiB`) | **0.741 s/it**<br>(`3.55 s` dec, `11.53 GiB`) | Single-GPU | **1.555 s/it**<br>(`1.82 s` dec, `32.88 GiB`) | — | Pending |
| **Qwen-Image Layered**<br>`qwen_image_layered_int8convrot` | `640×640`, 2 layers, 2 steps, CFG 2.5 | Running | Running | Running | `PP=2` | Running | `PP=2` | Pending |
| **Marigold V2 Depth**<br>`qwen_image_edit_2509_int8_convrot` | `512×512`, 2 steps, CFG 1.0 | Running | Running | **0.228 s/it**<br>(`3.26 s` dec, `20.49 GiB`) | Single-GPU | **0.569 s/it**<br>(`0.17 s` dec, `36.06 GiB`) | — | Pending |
| **Wan 2.1 SCAIL-2 14B**<br>`wan2.1_14B_SCAIL_2_int8_convrot` | `512×512`, 17 frames, 2 steps, CFG 6.0 | Running | Running | **2.698 s/it**<br>(`5.68 s` dec, `19.43 GiB`) | Single-GPU | Running | — | Pending |
| **SeedVR2 3B**<br>`seedvr2_3b_int8_convrot` | `512×512`, 2 steps, CFG 1.0 | Running | Running | Running | Single-GPU | **0.384 s/it**<br>(`0.16 s` dec, `7.92 GiB`) | — | Pending |
| **YuE2 3B Music**<br>`yue2_3b_int8_convrot` | `30.0 s` stereo audio, 2 steps, CFG 1.0 | Running | Running | Running | Single-GPU | **0.019 s/it**<br>(`0.39 s` dec, `7.44 GiB`) | — | Pending |
| **MiniMax Music 3**<br>`minimax_music3_dit_int8_convrot` | `30.0 s` stereo audio, 2 steps, CFG 3.0 | Running | Running | **0.040 s/it**<br>(`19.02 s` dec, `10.98 GiB`) | Single-GPU | **0.105 s/it**<br>(`0.20 s` dec, `17.56 GiB`) | — | Pending |

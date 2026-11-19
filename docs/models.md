# Supported Models and Hardware Benchmarks

Model support, memory estimates, and measured INT8 ConvRot results.

---

## Supported models

Memory figures in the support tables are approximate. See the benchmark results below for measured peaks at specific settings.

### Fork additions

| Model | This Fork | Upstream |
| --- | --- | --- |
| MAGI-2 Preview | TP/PP, ~136 GiB BF16. 2× GB10 TP=2 ran at 36.5 s/step on average (896×512, 25 frames, 4 steps). | — |
| MAGI-2 Preview + Refiner | TP/PP, ~154 GiB BF16 or ~42 GiB per rank with INT8 TP=2. | — |
| HunyuanImage 3.0 Instruct | TP/PP, ~164 GiB BF16 or ~22 GiB per GPU with W4A8 TP=2. | — |
| Kandinsky 6 Pro | TP, ~66 GiB total. | — |
| PAN 2 | TP/PP, ~37 GiB INT8 resident or ~12 GiB offloaded. | Single GPU |
| MiniMax H3 Video+Audio | TP/PP, ~132 GiB BF16 or ~55 GiB INT8 resident. | Single GPU |
| MiniMax H3 Image | TP/PP, ~53 GiB INT8 resident or ~12 GiB offloaded. | Single GPU |
| Flux.2 Dev 32B | TP/PP, ~46 GiB INT8 resident or ~12 GiB offloaded. | Single GPU |
| Ideogram 4 | TP, ~29 GiB INT8 resident. | Single GPU |
| Krea 2 Turbo | TP, ~18 GiB INT8 resident. | Single GPU |
| Qwen-Image family | PP, native GGUF, ~25 GiB INT8 resident. | Single GPU |
| SVDQuant W4A4 and AWQ W4A16 | Native loading, ~7–12 GiB for the listed image models. | — |
| TransformersLoader | Built-in text and vision models, ~2–28 GiB depending on model. | — |
| Vector and layout generation | StarVector, VTracer, TextDiffuser. Under 2 GiB for VTracer, ~8 GiB for TextDiffuser. | — |

### Shared model catalog

These model families are available in both versions. Memory estimates vary with checkpoint and settings.

| Model | Type | Example settings | Memory |
| --- | --- | --- | --- |
| **Flux.3 Image Klein (4B)** | Image (T2I / Edit) | `1024×1024`, 2–4 steps (`er_sde`) | **10.8 GiB** (`INT8 ConvRot`) / **17.5 GiB** (`BF16`); **6 GiB** offloaded |
| **Flux.2 Klein (4B / 8B / 9B)** | Image (T2I / Edit) | `1024×1024`, 20 steps (`Flux2Scheduler`) | **9.6–16.8 GiB** (`INT8`/`FP8`) / **16–26 GiB** (`BF16`); **6–8 GiB** offloaded |
| **Flux.1 Dev / Schnell / Fill / Canny / Redux** | Image (T2I / Control / Inpaint) | `1024×1024`, 4–20 steps (`euler`) | **16.5 GiB** (`FP8`/`INT8`) / **33.5 GiB** (`BF16`); **8 GiB** offloaded |
| **Fibo 2.0 Dev / Edit Turbo** | Image (T2I / Edit) | `1024×1024`, 12–20 steps (`Fibo2Scheduler`) | **13.4 GiB** (`INT8 ConvRot`) / **23.8 GiB** (`BF16`); **8 GiB** offloaded |
| **MOSS-Image (Base / SFT)** | Image (T2I) | `1024×1024`, 20 steps (`euler` / `sgm_uniform`) | **15.2 GiB** (`INT8 ConvRot`) / **28.5 GiB** (`BF16`); **8 GiB** offloaded |
| **Mage-Flow (Base / Turbo / Edit)** | Image (T2I / Edit) | `1024×1024`, 4–20 steps (`euler`) | **14.8 GiB** (`INT8 ConvRot`) / **26.0 GiB** (`BF16`); **8 GiB** offloaded |
| **Ming-Image 0.1 Design** | Image (T2I) | `1024×1024`, 20 steps (`euler`) | **15.6 GiB** (`INT8 ConvRot`) / **28.0 GiB** (`BF16`); **8 GiB** offloaded |
| **Z-Image / Z-Image Turbo / PixelSpace** | Image (T2I) | `1024×1024`, 9 steps (`euler`) | **10.5 GiB** (`INT8 ConvRot`) / **19.3 GiB** (`BF16`); **6 GiB** offloaded |
| **JoyAI Image Edit** | Image (Edit) | `1024×1024`, 20 steps (`euler`) | **24.8 GiB** (`INT8 ConvRot`) / **46.0 GiB** (`BF16`); **10 GiB** offloaded |
| **Boogu Image Edit / Turbo** | Image (T2I / Edit) | `1024×1024`, 8–20 steps (`res_multistep`) | **18.0 GiB** (`INT8`/`FP8`) / **32.0 GiB** (`BF16`); **8 GiB** offloaded |
| **Ernie-Image / Turbo** | Image (T2I) | `1024×1024`, 4–20 steps (`euler`) | **13.5 GiB** (`INT8 ConvRot`) / **22.2 GiB** (`BF16`); **8 GiB** offloaded |
| **Chroma / Chroma Radiance** | Image (T2I / Pixel-Space) | `1024×1024`, 20 steps (`euler`) | **14.2 GiB** (`INT8`/`FP8`) / **26.5 GiB** (`BF16`); **8 GiB** offloaded |
| **HiDream I1 / E1 / O1** | Image (T2I / Edit / Pixel) | `1024×1024` (`2048×2048` O1), 20 steps | **22.0 GiB** (`FP8`/`INT8`) / **42–74 GiB** (`BF16`); **10 GiB** offloaded |
| **SenseNova U1.5** | Image (Pixel-Space T2I) | `1024×1024`, 20 steps | **16.0 GiB** (`INT8`) / **28.0 GiB** (`BF16`); **8 GiB** offloaded |
| **HunyuanImage 2.1 + Refiner** | Image (2K T2I) | `2048×2048`, 20 steps | **32.0 GiB** (`FP8`/`INT8`) / **64.0 GiB** (`BF16`); **12 GiB** offloaded |
| **Omnigen 2 / Ovis-Image / Lumina 2** | Image (T2I / Multimodal) | `1024×1024`, 20 steps | **9.5–18.5 GiB** (`FP16`/`BF16`); **6–8 GiB** offloaded |
| **Microsoft Lens (3.8B + GPT-OSS-20B)** | Image (T2I) | `1024×1024`, 20 steps | **21.5 GiB** (`FP8`/`BF16`); **8 GiB** offloaded |
| **NVIDIA PixelDiT (1.3B) & PiD** | Image (Pixel-Space T2I) | `1024×1024`, 4–20 steps (`MXFP8`) | **4.8 GiB** (`MXFP8`) / **8.2 GiB** (`BF16`); **4 GiB** offloaded |
| **SD 3.5 Medium / Large, SDXL, SD 1.5, AuraFlow, HunyuanDiT, Kandinsky 5, Anima, LongCat** | Image (T2I) | `1024×1024` (`512×512` SD1.5) | **2.2–21.0 GiB** resident; **4–8 GiB** offloaded |
| **LTX-2 (19B) & LTX-2.5 (22B)** | Video+Audio (T2VA / I2VA) | `1280×720`, 49–97 frames + audio | **31.5–36.0 GiB** (`INT8 ConvRot` / `FP8`); **12 GiB** offloaded |
| **LTX-Video (2B v0.9.x)** | Video (T2V / I2V) | `768×512`, 97 frames | **9.5 GiB** (`BF16`); **6 GiB** offloaded |
| **Wan 2.1 / 2.2 (1.3B / 5B / 14B, VACE, SCAIL-2, Animate-2, Causal-AR, RelayVSR, WanDancer)** | Video (T2V / I2V / Control / VSR) | `832×480`, 81 frames (`1.3B`–`14B`) | **6.5 GiB** (`1.3B`) to **22.5 GiB** (`14B INT8 ConvRot`); **6–10 GiB** offloaded |
| **HunyuanVideo & HunyuanVideo 1.5 (+ SR)** | Video (T2V / I2V) | `848×480` / `1280×720`, 73 frames | **24.0 GiB** (`FP8`/`INT8`) / **47.8 GiB** (`BF16`); **10 GiB** offloaded |
| **Cosmos 1.0 (7B) & Cosmos Predict2 (2B / 14B)** | Video (Text/Video2World) | `1280×704` / `848×480`, 33–121 frames | **9.5 GiB** (`2B`) / **21.0 GiB** (`7B FP8`); **6–10 GiB** offloaded |
| **Genmo Mochi 1 Preview** | Video (T2V) | `848×480`, 37 frames | **22.5 GiB** (`FP8`) / **39.1 GiB** (`BF16`); **10 GiB** offloaded |
| **SeedVR2 (3B / 7B / 7B Sharp)** | Video / Image Restoration | `512×512` $\rightarrow$ `2048×2048` | **4.8 GiB** (`3B INT8 ConvRot`) / **9.2 GiB** (`7B INT8 ConvRot`); **6 GiB** offloaded |
| **CogVideoX (2B / 5B / 1.5)** & **SVD** | Video (T2V / I2V) | `720×480`, 49–81 frames | **8.5–16.0 GiB**; **6–8 GiB** offloaded |
| **MiniMax Music 3, YuE2 (3B), ACE-Step 1.0 / 1.5, Stable Audio 1.0 / 3.0** | Audio / Music | `30.0–47.6 s` stereo audio | **3.8–8.5 GiB** (`INT8 ConvRot` / `BF16`); **4–6 GiB** offloaded |
| **Hunyuan3D 2.0 / 2.1, TRELLIS.2, Pixal3D, TripoSplat** | 3D Mesh / Gaussian Splat | Single/Multi-View Image $\rightarrow$ 3D | **3.5–12.0 GiB** (`INT8 ConvRot` / `FP16`); **6–8 GiB** offloaded |
| **SAM 3 / 3.1, DepthAnything 3, MoGe 3, Marigold V2, RT-DETR v4, Lotus** | Vision / Segmentation / Depth / Normals | `1024×1024` image / video frames | **0.5–12.5 GiB**; **4–8 GiB** offloaded |
---

## INT8 ConvRot benchmarks

Each stage runs two steps. Cells show **seconds for step 2 · peak GiB**. GB10 memory is estimated from the drop in available system memory on each GPU host. For 2× GB10, the two numbers follow rank order. Runs use tensor parallelism unless marked PP. A dash means no verified result. Runs that crashed during cleanup are excluded.

### Images

| Model | Settings | RX 7600 | RTX 3060 | RTX 3090 | 1× GB10 | 2× GB10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Boogu Image Edit | 1024×1024, CFG 1, res_multistep / normal | — | 1.695 · 11.3 | 0.741 · 11.5 | 1.555 · 32.9 | — |
| Fibo 2.0 Dev | 1024×1024, CFG 5, Euler / Fibo2 | — | 2.600 · 12.0 | 1.350 · 22.3 | — | — |
| Flux.2 Dev 32B | 1024×1024, CFG 1, Euler / Flux2 | — | 6.568 · 11.8 | 16.948 · 23.8 | 5.411 · 56.9 | 3.711 · 44.1 / 23.6 |
| Flux.3 Klein 4B | 1024×1024, CFG 1, er_sde / simple | — | 0.686 · 7.6 | 0.340 · 6.2 | 0.665 · 17.8 | — |
| Ideogram 4 | 1024×1024, CFG 3, res_multistep / Ideogram4 | — | 4.291 · 10.7 | 1.838 · 11.0 | — | 2.920 · 29.0 / 16.4 |
| JoyAI Image Edit | 512×512, CFG 4, Euler / simple | — | 5.798 · 11.8 | 2.215 · 17.0 | 5.275 · 32.6 | — |
| Krea 2 Turbo | 1024×1024, CFG 1, res_multistep / normal | — | 2.393 · 11.8 | 0.943 · 14.1 | 2.388 · 27.0 | 1.865 · 24.0 / 14.4 |
| Mage-Flow | 1024×1024, CFG 5, Euler / simple | 1.837 · 7.0 | 0.720 · 11.7 | 0.266 · 16.1 | — | — |
| Marigold V2 Depth | 512×512, Euler, manual sigmas | — | 1.446 · 11.8 | 0.228 · 20.5 | 0.569 · 36.1 | — |
| Ming-Image 0.1 Design | 1024×1024, CFG 4, Euler / simple | — | 1.050 · 11.8 | 0.470 · 21.9 | 1.086 · 37.7 | — |
| MiniMax H3 Image | 1024×1024, CFG 1, Euler / simple | — | 1.364 · 11.8 | 0.331 · 23.7 | — | — |
| MOSS-Image SFT | 1024×1024, CFG 3, Euler / sgm_uniform | — | 1.598 · 8.8 | 0.766 · 8.5 | 1.600 · 22.0 | — |
| Qwen-Image 2.1 | 1024×1024, CFG 4, Euler / simple | — | 2.250 · 7.9 | — | 2.160 · 26.5 | — |
| Qwen-Image Layered | 640×640, 2 layers, CFG 2.5, Euler / simple | — | — | — | 5.127 · 36.3 | 5.331 · 29.3 / 12.8 (PP) |
| SeedVR2 3B | 512×512, CFG 1, Euler / simple | 0.333 · 5.7 | 0.167 · 4.0 | 0.122 · 4.5 | 0.384 · 7.9 | — |
| Z-Image Turbo 6B | 1024×1024, CFG 1, Euler / simple | — | 1.079 · 12.0 | 0.475 · 21.9 | 1.023 · 25.4 | — |

### Video and audio

| Model | Settings | RX 7600 | RTX 3060 | RTX 3090 | 1× GB10 | 2× GB10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PAN 2 | 1248×704, 241 frames at 24 fps, CFG 5, Euler / simple | — | — | — | — | 72.070 · 95.5 / 22.2 |
| YuE2 3B | 30 s audio, CFG 1, Euler / simple | 0.171 · 4.9 | 0.064 · 2.8 | — | — | — |
| MiniMax H3 Video+Audio | 1280×704, 124 frames at 24 fps, CFG 1, res_multistep / simple | — | — | — | — | — |
| MAGI-2 Preview | 896×512, 10 s target at 12.5 fps, video CFG 5, audio CFG 7, UniPC Flow | — | — | — | — | — |
| MAGI-2 Preview + Refiner | Same preview, then 1920×1088 at 25 fps, refiner CFG 2, UniPC Flow | — | — | — | — | — |
| Wan 2.1 SCAIL-2 | 832×480, 81 frames at 8 fps, CFG 6, Euler / normal | — | — | — | — | — |

<details>
<summary>Earlier runs at smaller settings</summary>

| Model | Settings | RX 7600 | RTX 3060 | RTX 3090 | 1× GB10 | 2× GB10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ideogram 4 | 384×688, CFG 3, res_multistep / Ideogram4 | — | 1.716 · 11.7 | 0.880 · 10.5 | 0.848 · 33.4 | 1.079 · 28.8 / 15.5 |
| Mage-Flow | 512×512, CFG 5, Euler / simple | — | 0.162 · 12.0 | 0.095 · 16.1 | 0.186 · 17.0 | — |
| Qwen-Image 2.1 | 256×256, CFG 4, Euler / simple | — | 0.131 · 7.9 | 0.119 · 8.1 | 0.542 · 21.5 | — |
| MiniMax H3 Video+Audio | 608×352, 5 frames at 24 fps, res_multistep / simple | — | — | 0.168 · 23.7 | — | — |
| PAN 2 | 512×288, 21 frames at 24 fps, CFG 5, Euler / simple | — | 3.773 · 11.8 | 0.840 · 23.8 | 1.588 · 44.1 | 2.015 · 33.4 / 15.6 |
| Wan 2.1 SCAIL-2 | 512×512, 17 frames at 8 fps, CFG 6, Euler / normal | — | 6.043 · 11.8 | 2.698 · 19.4 | 5.704 · 31.9 | — |
| MiniMax Music 3 | 0.07 s audio, res_multistep / simple | — | 0.055 · 10.7 | 0.040 · 11.0 | 0.105 · 17.6 | — |
| YuE2 3B | 1 s audio, CFG 1, Euler / simple | 0.100 · 2.4 | 0.043 · 2.7 | 0.033 · 3.9 | 0.019 · 7.4 | — |

</details>

The benchmark saves the prepared workflow and separate timings for load and encode, sampling, and VAE decode.

```console
python tests/convrot_benchmark.py --model all --out-dir ./benchmark-output
```

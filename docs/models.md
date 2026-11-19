# Supported models

<a id="int8-convrot-benchmarks"></a>

Memory figures below are estimates. For inference throughput and total run time, see the [INT8 ConvRot benchmarks](benchmarks.md).

## Fork additions

| Model | This Fork | Upstream |
| --- | --- | --- |
| MAGI-2 Preview | TP/PP, 114B MoE, 107.6 GiB of INT8 transformer weights. See the [125-frame results](benchmarks.md). | — |
| MAGI-2 Preview + Refiner | TP/PP, 107.6 GiB of preview weights plus 6.3 GiB of refiner weights (INT8). | — |
| HunyuanImage 3.0 Instruct | TP/PP, ~164 GiB BF16 or ~22 GiB per GPU with W4A8 TP=2. | — |
| Kandinsky 6 Pro | TP, ~66 GiB total. | — |
| PAN 2 | TP/PP, ~37 GiB INT8 resident or ~12 GiB offloaded. | Single GPU |
| MiniMax H3 Video+Audio | TP/PP, ~132 GiB BF16 or ~55 GiB INT8 resident. | Single GPU |
| MiniMax H3 Image | TP/PP, ~53 GiB INT8 resident or ~12 GiB offloaded. | Single GPU |
| Flux.2 Dev 32B | TP/PP, ~46 GiB INT8 resident or ~12 GiB offloaded. | Single GPU |
| Ideogram 4 | TP, ~29 GiB INT8 resident. | Single GPU |
| Krea 2 Turbo | TP, ~18 GiB INT8 resident. | Single GPU |
| Qwen-Image and Qwen-Image Layered | PP, native GGUF, ~25 GiB INT8 resident. | Single GPU |
| SVDQuant W4A4 and AWQ W4A16 | Native loading, ~7–12 GiB for the listed image models. | — |
| TransformersLoader | Built-in text and vision models, ~2–28 GiB depending on model. | — |
| Vector and layout generation | StarVector, VTracer, TextDiffuser. Under 2 GiB for VTracer, ~8 GiB for TextDiffuser. | — |

## Shared model catalog

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

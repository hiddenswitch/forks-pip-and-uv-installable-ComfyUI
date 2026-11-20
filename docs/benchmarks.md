# INT8 ConvRot benchmarks

## Iterations per second

Throughput uses the second step. Sampling time includes both steps per stage.

| Model | RX 7600 | RTX 3060 | RTX 3090 | 1x GB10 | 2x GB10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PAN 2<br><sub>1248x704, 241 frames at 24 fps</sub> | — | 0.0083 | 0.0181 | 0.0089 | 0.0139 |
| MAGI-2 Preview<br><sub>896x512, 125 frames at 12.5 fps, 44.1 kHz stereo</sub> | — | 0.0016 | 0.0041 | 0.0019 | 0.0012 |
| MAGI-2 Preview + Refiner<br><sub>896x512, 125 frames at 12.5 fps → 1920x1088, 249 frames at 25 fps, 44.1 kHz stereo</sub> | — | 0.0008 | 0.0011 | 0.0015 | 0.0004 |
| MiniMax H3 Image | 0.0881 | 0.7332 | 3.0191 | 2.1201 | 1.2980 |
| Fibo 2.0 Dev | — | 0.3848 | 0.7410 | 0.3754 | — |
| MOSS-Image SFT | 0.2535 | 0.6259 | 1.3060 | 0.6251 | — |
| Flux.3 Klein 4B | 0.6079 | 1.4568 | 2.9381 | 1.5043 | — |
| Ming-Image 0.1 Design<br><sub>RX 7600 uses CPU text encoding and decode</sub> | 0.3811 | 0.9527 | 2.1273 | 0.9208 | — |
| Qwen-Image 2.1 | — | 0.4444 | 1.0137 | 0.4592 | — |
| YuE2 3B | 5.8440 | 15.5329 | 21.1258 | 22.7245 | — |
| Marigold V2 Depth | 0.0363 | 0.3119 | 1.4433 | 0.6347 | — |
| MiniMax Music 3 | 0.8107 | 4.0794 | 2.8680 | 2.6691 | — |
| MiniMax H3 Video+Audio<br><sub>1280x704, 124 frames at 24 fps, 32 kHz stereo</sub> | — | 0.0154 | 0.0468 | 0.0167 | 0.0306 |
| Mage-Flow | 0.5444 | 1.3895 | 3.7536 | 1.6922 | — |
| JoyAI Image Edit | — | 0.1077 | 0.3015 | 0.1084 | — |
| Boogu Image Edit | — | 0.5908 | 1.3499 | 0.6431 | — |
| Krea 2 Turbo | — | 0.4178 | 1.0603 | 0.4188 | 0.5363 |
| Wan 2.1 SCAIL-2<br><sub>832x480, 81 frames at 8 fps</sub> | — | 0.0129 | 0.0371 | 0.0185 | — |
| Ideogram 4 | — | 0.2330 | 0.5441 | 0.2941 | 0.3425 |
| Qwen-Image Layered | — | 0.1225 | 0.4352 | 0.1951 | 0.1876 |
| Z-Image Turbo 6B | 0.3800 | 0.9267 | 2.1045 | 0.9776 | — |
| Flux.2 Dev 32B | 0.0224 | 0.1481 | 0.3068 | 0.1848 | 0.2695 |
| SeedVR2 3B | 0.2105 | 0.4911 | 0.9772 | 0.3916 | — |

## Timing breakdown

Loading includes prompt encoding. A dash indicates unavailable timing, including loading runs with downloads.

### Model loading (s)

| Model | RX 7600 | RTX 3060 | RTX 3090 | 1x GB10 | 2x GB10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PAN 2<br><sub>1248x704, 241 frames at 24 fps</sub> | — | 9.17 | 34.98 | 66.85 | 56.59 |
| MAGI-2 Preview<br><sub>896x512, 125 frames at 12.5 fps, 44.1 kHz stereo</sub> | — | 43.91 | 44.43 | 186.87 | 445.38 |
| MAGI-2 Preview + Refiner<br><sub>896x512, 125 frames at 12.5 fps → 1920x1088, 249 frames at 25 fps, 44.1 kHz stereo</sub> | — | 252.94 | 235.92 | 184.60 | 442.83 |
| MiniMax H3 Image | 47.73 | 758.37 | 840.30 | 78.19 | 84.41 |
| Fibo 2.0 Dev | — | 218.55 | 286.11 | 24.36 | — |
| MOSS-Image SFT | 26.14 | 238.74 | 11.81 | — | — |
| Flux.3 Klein 4B | 361.48 | 482.49 | 824.92 | — | — |
| Ming-Image 0.1 Design<br><sub>RX 7600 uses CPU text encoding and decode</sub> | 208.39 | 453.61 | 1044.58 | — | — |
| Qwen-Image 2.1 | — | 186.04 | 192.84 | 30.58 | — |
| YuE2 3B | 477.13 | 94.55 | 64.07 | 25.10 | — |
| Marigold V2 Depth | 367.82 | 16.27 | 17.95 | 44.79 | — |
| MiniMax Music 3 | 260.95 | 274.41 | 243.71 | 70.27 | — |
| MiniMax H3 Video+Audio<br><sub>1280x704, 124 frames at 24 fps, 32 kHz stereo</sub> | — | 4.70 | 3.66 | 100.93 | 66.07 |
| Mage-Flow | 1043.57 | 203.18 | 859.37 | 23.18 | — |
| JoyAI Image Edit | — | 11.99 | 8.41 | 44.73 | — |
| Boogu Image Edit | — | 6.30 | 259.56 | — | — |
| Krea 2 Turbo | — | 136.33 | 187.47 | — | 30.71 |
| Wan 2.1 SCAIL-2<br><sub>832x480, 81 frames at 8 fps</sub> | — | 188.65 | 142.19 | — | — |
| Ideogram 4 | — | 194.16 | 157.75 | 43.41 | 36.76 |
| Qwen-Image Layered | — | 302.60 | 365.78 | — | 49.56 |
| Z-Image Turbo 6B | 28.58 | 242.90 | 1013.30 | — | — |
| Flux.2 Dev 32B | 39.28 | 13.77 | 10.95 | — | — |
| SeedVR2 3B | 150.17 | 9.89 | 28.25 | 9.14 | — |

### Sampling (s)

| Model | RX 7600 | RTX 3060 | RTX 3090 | 1x GB10 | 2x GB10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PAN 2<br><sub>1248x704, 241 frames at 24 fps</sub> | — | 250.80 | 703.92 | 240.09 | 148.28 |
| MAGI-2 Preview<br><sub>896x512, 125 frames at 12.5 fps, 44.1 kHz stereo</sub> | — | 1278.23 | 484.26 | 1197.66 | 1670.86 |
| MAGI-2 Preview + Refiner<br><sub>896x512, 125 frames at 12.5 fps → 1920x1088, 249 frames at 25 fps, 44.1 kHz stereo</sub> | — | 4797.66 | 3604.49 | 2875.35 | 10144.17 |
| MiniMax H3 Image | 27.42 | 496.42 | 385.27 | 1.66 | 2.38 |
| Fibo 2.0 Dev | — | 238.02 | 187.79 | 7.07 | — |
| MOSS-Image SFT | 8.85 | 124.38 | 570.61 | 4.32 | — |
| Flux.3 Klein 4B | 4.20 | 122.14 | 702.21 | 2.92 | — |
| Ming-Image 0.1 Design<br><sub>RX 7600 uses CPU text encoding and decode</sub> | 6.22 | 144.34 | 224.19 | 3.50 | — |
| Qwen-Image 2.1 | — | 189.40 | 147.43 | 5.80 | — |
| YuE2 3B | 1.04 | 40.73 | 35.80 | 0.42 | — |
| Marigold V2 Depth | 52.48 | 11.94 | 4.65 | 4.11 | — |
| MiniMax Music 3 | 3.14 | 68.80 | 55.51 | 1.28 | — |
| MiniMax H3 Video+Audio<br><sub>1280x704, 124 frames at 24 fps, 32 kHz stereo</sub> | — | 139.74 | 45.77 | 115.65 | 69.65 |
| Mage-Flow | 4.10 | 104.99 | 296.72 | 2.32 | — |
| JoyAI Image Edit | — | 23.79 | 9.63 | 19.35 | — |
| Boogu Image Edit | — | 6.66 | 183.35 | 4.55 | — |
| Krea 2 Turbo | — | 384.26 | 350.98 | 5.51 | 8.12 |
| Wan 2.1 SCAIL-2<br><sub>832x480, 81 frames at 8 fps</sub> | — | 578.88 | 419.92 | 112.97 | — |
| Ideogram 4 | — | 464.82 | 420.52 | 8.31 | 10.33 |
| Qwen-Image Layered | — | 935.93 | 556.27 | 11.42 | 12.56 |
| Z-Image Turbo 6B | 6.58 | 126.08 | 446.54 | 3.24 | — |
| Flux.2 Dev 32B | 89.26 | 24.23 | 16.18 | 12.04 | 9.82 |
| SeedVR2 3B | 11.44 | 82.07 | 285.24 | 7.49 | — |

### Decode (s)

| Model | RX 7600 | RTX 3060 | RTX 3090 | 1x GB10 | 2x GB10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PAN 2<br><sub>1248x704, 241 frames at 24 fps</sub> | — | 208.04 | 90.99 | 150.05 | 171.05 |
| MAGI-2 Preview<br><sub>896x512, 125 frames at 12.5 fps, 44.1 kHz stereo</sub> | — | 22.47 | 7.25 | 16.25 | 68.05 |
| MAGI-2 Preview + Refiner<br><sub>896x512, 125 frames at 12.5 fps → 1920x1088, 249 frames at 25 fps, 44.1 kHz stereo</sub> | — | 288.93 | 105.44 | 128.08 | 422.19 |
| MiniMax H3 Image | 3.58 | 28.51 | 16.11 | 0.74 | 1.15 |
| Fibo 2.0 Dev | — | 9.31 | 0.68 | 0.68 | — |
| MOSS-Image SFT | 30.65 | 3.44 | 20.34 | 1.02 | — |
| Flux.3 Klein 4B | 19.81 | 5.09 | 23.64 | 0.75 | — |
| Ming-Image 0.1 Design<br><sub>RX 7600 uses CPU text encoding and decode</sub> | 14.83 | 3.48 | 4.31 | 1.23 | — |
| Qwen-Image 2.1 | — | 12.20 | 10.12 | 2.13 | — |
| YuE2 3B | 1.39 | 0.56 | 0.42 | 1.16 | — |
| Marigold V2 Depth | 0.00 | 0.69 | 0.22 | 0.79 | — |
| MiniMax Music 3 | 4.68 | 1.08 | 0.60 | 3.96 | — |
| MiniMax H3 Video+Audio<br><sub>1280x704, 124 frames at 24 fps, 32 kHz stereo</sub> | — | 96.27 | 29.74 | 40.23 | 41.22 |
| Mage-Flow | 0.50 | 3.87 | 3.95 | 0.52 | — |
| JoyAI Image Edit | — | 0.46 | 0.21 | 1.34 | — |
| Boogu Image Edit | — | 1.15 | 2.34 | 1.74 | — |
| Krea 2 Turbo | — | 12.35 | 3.55 | 1.30 | 1.25 |
| Wan 2.1 SCAIL-2<br><sub>832x480, 81 frames at 8 fps</sub> | — | 45.39 | 18.36 | 27.30 | — |
| Ideogram 4 | — | 11.23 | 4.87 | 0.98 | 0.77 |
| Qwen-Image Layered | — | 5.02 | 1.02 | 1.21 | 1.17 |
| Z-Image Turbo 6B | 31.47 | 4.90 | 7.75 | 1.83 | — |
| Flux.2 Dev 32B | 20.75 | 1.26 | 0.82 | 0.83 | 0.73 |
| SeedVR2 3B | 435.18 | 15.28 | 31.71 | 2.47 | — |

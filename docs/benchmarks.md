# INT8 ConvRot benchmarks

## Iterations per second

Two steps per stage. Throughput is measured on the second step.

| Model | RX 7600 | RTX 3060 | RTX 3090 | 1x GB10 | 2x GB10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boogu Image Edit | — | 0.59 | 1.35 | 0.643 | — |
| Fibo 2.0 Dev | — | 0.385 | 0.741 | 0.375 | — |
| Flux.2 Dev 32B | 0.0224 | 0.152 | 0.059 | 0.185 | 0.269 |
| Flux.3 Klein 4B | 0.608 | 1.46 | 2.94 | 1.5 | — |
| Ideogram 4 | — | 0.233 | 0.544 | 0.294 | 0.343 |
| JoyAI Image Edit | — | 0.108 | 0.302 | 0.108 | — |
| Krea 2 Turbo | — | 0.418 | 1.06 | 0.419 | 0.536 |
| Mage-Flow | 0.544 | 1.39 | 3.75 | 1.69 | — |
| Marigold V2 Depth | 0.0363 | 0.312 | 1.44 | 0.635 | — |
| Ming-Image 0.1 Design | — | 0.953 | 2.13 | 0.921 | — |
| MiniMax H3 Image | 0.0881 | 0.733 | 3.02 | 2.12 | 1.3 |
| MOSS-Image SFT | 0.254 | 0.626 | 1.31 | 0.625 | — |
| Qwen-Image 2.1 | — | 0.444 | 1.01 | 0.459 | — |
| Qwen-Image Layered | — | 0.123 | 0.435 | 0.195 | 0.188 |
| SeedVR2 3B | 0.21 | 0.491 | 0.977 | 0.392 | — |
| Z-Image Turbo 6B | 0.349 | 0.927 | 2.1 | 0.978 | — |
| PAN 2<br><sub>1248x704, 241 frames at 24 fps</sub> | — | 0.0083 | 0.0181 | 0.00894 | 0.0139 |
| MiniMax Music 3 | 0.811 | 4.08 | 2.87 | 2.67 | — |
| YuE2 3B | 5.84 | 15.5 | 21.1 | 22.7 | — |
| MiniMax H3 Video+Audio<br><sub>1280x704, 124 frames at 24 fps, 32 kHz stereo</sub> | — | 0.0154 | 0.0468 | 0.0167 | 0.0306 |
| MAGI-2 Preview<br><sub>896x512, 125 frames at 12.5 fps, 44.1 kHz stereo</sub> | — | 0.00156 | 0.00415 | 0.00188 | 0.00121 |
| MAGI-2 Preview + Refiner<br><sub>896x512, 125 frames at 12.5 fps → 1920x1088, 249 frames at 25 fps, 44.1 kHz stereo</sub> | — | 0.000841 | 0.00112 | 0.00147 | 0.000396 |
| Wan 2.1 SCAIL-2<br><sub>832x480, 81 frames at 8 fps</sub> | — | 0.0129 | 0.0371 | 0.0185 | — |

## Total seconds

Elapsed time from process start to exit, including loading and decoding.

| Model | RX 7600 | RTX 3060 | RTX 3090 | 1x GB10 | 2x GB10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boogu Image Edit | — | 946.0 | 505.9 | 368.0 | — |
| Fibo 2.0 Dev | — | 686.0 | 843.5 | 51.4 | — |
| Flux.2 Dev 32B | 194.5 | 1371.2 | 1826.6 | 1060.0 | 542.6 |
| Flux.3 Klein 4B | 481.7 | 754.5 | 1710.2 | 166.1 | — |
| Ideogram 4 | — | 722.7 | 633.5 | 71.3 | 62.6 |
| JoyAI Image Edit | — | 93.2 | 91.8 | 84.4 | — |
| Krea 2 Turbo | — | 630.4 | 1038.2 | 273.6 | 69.9 |
| Mage-Flow | 1102.8 | 378.2 | 1210.5 | 42.4 | — |
| Marigold V2 Depth | 499.9 | 100.5 | 109.9 | 70.8 | — |
| Ming-Image 0.1 Design | — | 682.5 | 1377.2 | 445.3 | — |
| MiniMax H3 Image | 138.9 | 1444.0 | 1300.9 | 101.7 | 113.8 |
| MOSS-Image SFT | 138.2 | 755.6 | 672.8 | 632.5 | — |
| Qwen-Image 2.1 | — | 441.8 | 398.0 | 60.4 | — |
| Qwen-Image Layered | — | 1401.0 | 980.9 | 346.5 | 77.1 |
| SeedVR2 3B | 644.4 | 170.7 | 433.6 | 39.6 | — |
| Z-Image Turbo 6B | 165.6 | 474.8 | 1484.2 | 173.6 | — |
| PAN 2<br><sub>1248x704, 241 frames at 24 fps</sub> | — | 543.9 | 920.0 | 488.7 | 405.6 |
| MiniMax Music 3 | 317.4 | 396.3 | 346.7 | 93.1 | — |
| YuE2 3B | 531.4 | 182.7 | 151.8 | 45.7 | — |
| MiniMax H3 Video+Audio<br><sub>1280x704, 124 frames at 24 fps, 32 kHz stereo</sub> | — | 418.3 | 174.3 | 291.3 | 210.1 |
| MAGI-2 Preview<br><sub>896x512, 125 frames at 12.5 fps, 44.1 kHz stereo</sub> | — | 1410.8 | 619.5 | 1424.6 | 2216.3 |
| MAGI-2 Preview + Refiner<br><sub>896x512, 125 frames at 12.5 fps → 1920x1088, 249 frames at 25 fps, 44.1 kHz stereo</sub> | — | 5427.9 | 4008.3 | 3220.5 | 11046.9 |
| Wan 2.1 SCAIL-2<br><sub>832x480, 81 frames at 8 fps</sub> | — | 870.4 | 625.6 | 439.4 | — |

# Media and SVG workflows

## High-bit-depth images and HDR

Use **Save Image (Advanced)** to select the output format and precision:

- PNG supports 8-bit and 16-bit output.
- EXR supports 16-bit and 32-bit floating-point output, with explicit input color-space selection.
- AVIF provides high-bit-depth and HDR output options.

Match the input color-space setting to the actual image data. Selecting an HDR
container does not by itself restore highlights missing from an SDR image.
Use the image color-space and SDR-to-HDR nodes when a workflow needs conversion.
Video nodes also expose codec, bit-depth, and color-space choices. Available
encoders depend on the installed media libraries.

These capabilities build on the media support shared with upstream ComfyUI.

## Raster and SVG conversion

Use `ImageToSVG` to vectorize an image and `SVGToImage` to rasterize SVG. Save
the vector result or continue processing its raster representation.

![SVG conversion example](assets/svg_01.png)

Try the [SVG conversion workflow](../tests/inference/workflows/svg-0.json).

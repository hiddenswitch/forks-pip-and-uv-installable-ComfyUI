"""MAGI-2's distilled Wan 2.2 decoder (TurboVAED, decoder only).

Decodes Wan 2.2 latents (MAGI-2's normalized latents are mapped to this basis by the VAE, see
magi2.latents) in overlapping temporal chunks: a first chunk of
first_chunk_size latent frames plus one frame of right context, then step_size frames with one
frame of context on each side; each context frame's temporal_compression output frames are cut.
"""

import torch
import torch.nn.functional as F
from torch import nn

from ... import ops

ops = ops.disable_weight_init


def rms_channels(x, eps):
    """RMS over the channel axis in fp32, returned in the input dtype."""
    return (x * torch.rsqrt(x.float().pow(2).mean(1, keepdim=True) + eps)).to(x.dtype)


class TimePadConv3d(nn.Module):
    """3D convolution padded in space with zeros and in time by repeating the edge frames."""

    def __init__(self, in_channels, out_channels, kernel_size=3, dtype=None, device=None, operations=ops):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.conv = operations.Conv3d(in_channels, out_channels, self.kernel_size, padding=(0, self.kernel_size[1] // 2, self.kernel_size[2] // 2),
                                      dtype=dtype, device=device)

    def forward(self, x):
        pad = (self.kernel_size[0] - 1) // 2
        if pad:
            x = torch.cat((x[:, :, :1].repeat(1, 1, pad, 1, 1), x, x[:, :, -1:].repeat(1, 1, pad, 1, 1)), dim=2)
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, after_upsampler=False, dtype=None, device=None, operations=ops):
        super().__init__()
        self.after_upsampler = after_upsampler
        self.conv1 = TimePadConv3d(in_channels, out_channels, dtype=dtype, device=device, operations=operations)
        self.conv2 = TimePadConv3d(out_channels, out_channels, dtype=dtype, device=device, operations=operations)
        self.conv_shortcut = TimePadConv3d(in_channels, out_channels, 1, dtype=dtype, device=device, operations=operations) if in_channels != out_channels else None

    def forward(self, x):
        h = rms_channels(x, 1e-8)
        h = F.relu(h) if self.after_upsampler else F.silu(h)
        h = self.conv2(F.silu(rms_channels(self.conv1(h), 1e-8)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(rms_channels(x, 1e-6))
        return h + x


class Resample(nn.Module):
    """Wan-style 2x spatial upsample, optionally doubling time with a learned frame split first."""

    def __init__(self, channels, temporal, dtype=None, device=None, operations=ops):
        super().__init__()
        self.resample = nn.Sequential(nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                                      operations.Conv2d(channels, channels, 3, padding=1, dtype=dtype, device=device))
        self.time_conv = TimePadConv3d(channels, channels * 2, (3, 1, 1), dtype=dtype, device=device, operations=operations) if temporal else None

    def forward(self, x, first_chunk):
        batch, channels, frames = x.shape[:3]
        if self.time_conv is not None:
            x = self.time_conv(x).unflatten(1, (2, channels)).permute(0, 2, 3, 1, 4, 5).flatten(2, 3)
            if first_chunk:
                x = x[:, :, 1:]
        frames = x.shape[2]
        x = x.transpose(1, 2).flatten(0, 1)
        x = self.resample[1](self.resample[0](x.float()).to(x.dtype))
        return x.unflatten(0, (batch, frames)).transpose(1, 2)


class MidBlock(nn.Module):
    def __init__(self, channels, layers, dtype=None, device=None, operations=ops):
        super().__init__()
        self.resnets = nn.ModuleList(ResnetBlock(channels, channels, dtype=dtype, device=device, operations=operations) for _ in range(layers))

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layers, upsample, spatial_only, dtype=None, device=None, operations=ops):
        super().__init__()
        self.conv_in = ResnetBlock(in_channels, out_channels, dtype=dtype, device=device, operations=operations) if in_channels != out_channels else None
        self.upsamplers = nn.ModuleList([Resample(out_channels, not spatial_only, dtype=dtype, device=device, operations=operations)]) if upsample else None
        self.resnets = nn.ModuleList(ResnetBlock(out_channels, out_channels, after_upsampler=upsample, dtype=dtype, device=device, operations=operations)
                                     for _ in range(layers))

    def forward(self, x, first_chunk):
        if self.conv_in is not None:
            x = self.conv_in(x)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, first_chunk)
        for resnet in self.resnets:
            x = resnet(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_channels, out_channels, block_out_channels, layers_per_block, upsample, spatial_only, patch_size,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.patch_size = patch_size
        blocks = tuple(reversed(block_out_channels))
        layers = tuple(reversed(layers_per_block))
        self.conv_in = TimePadConv3d(latent_channels, blocks[0], dtype=dtype, device=device, operations=operations)
        self.mid_block = MidBlock(blocks[0], layers[0], dtype=dtype, device=device, operations=operations)
        channels = blocks[0]
        self.up_blocks = nn.ModuleList()
        for index, (output, up, spatial) in enumerate(zip(blocks, reversed(upsample), reversed(spatial_only))):
            self.up_blocks.append(UpBlock(channels, output, layers[index + 1], up, spatial, dtype=dtype, device=device, operations=operations))
            channels = output
        self.conv_out = TimePadConv3d(channels, out_channels * patch_size * patch_size, dtype=dtype, device=device, operations=operations)

    def forward(self, x, first_chunk):
        x = self.mid_block(self.conv_in(x))
        for block in self.up_blocks:
            x = block(x, first_chunk)
        x = self.conv_out(F.silu(rms_channels(x, 1e-8)))
        batch, channels, frames, height, width = x.shape
        p = self.patch_size
        x = x.view(batch, channels // (p * p), p, p, frames, height, width).permute(0, 1, 4, 5, 3, 6, 2)
        return x.reshape(batch, channels // (p * p), frames, height * p, width * p)


class TurboVAEDecoder(nn.Module):
    def __init__(self, latent_channels=48, out_channels=3, block_out_channels=(64, 128, 256, 512), layers_per_block=(2, 2, 2, 3, 3),
                 upsample=(False, True, True, True), spatial_only=(False, True, False, False), patch_size=2, first_chunk_size=7, step_size=7,
                 temporal_compression=4, dtype=None, device=None, operations=ops):
        super().__init__()
        self.first_chunk_size = first_chunk_size
        self.step_size = step_size
        self.temporal_compression = temporal_compression
        self.decoder = Decoder(latent_channels, out_channels, block_out_channels, layers_per_block, upsample, spatial_only, patch_size,
                               dtype=dtype, device=device, operations=operations)

    def decode(self, z):
        first, step, overlap = self.first_chunk_size, self.step_size, self.temporal_compression
        frames = z.shape[2]
        padding = first - frames if frames < first else -(frames - first) % step
        if padding:
            z = torch.cat((z, z[:, :, -1:].repeat(1, 1, padding, 1, 1)), dim=2)
            frames += padding
        if frames == first:
            out = self.decoder(z, True)
        else:
            chunks = [self.decoder(z[:, :, :first + 1], True)[:, :, :-overlap]]
            for start in range(first, frames, step):
                last = start + step == frames
                chunk = self.decoder(z[:, :, start - 1:start + step + (0 if last else 1)], False)
                chunks.append(chunk[:, :, overlap:] if last else chunk[:, :, overlap:-overlap])
            out = torch.cat(chunks, dim=2)
        if padding:
            out = out[:, :, :-padding * self.temporal_compression]
        return out

    def encode(self, x):
        raise RuntimeError("The MAGI-2 turbo VAE has only a decoder; encode with the Wan 2.2 VAE")


def decoder_config(sd):
    """TurboVAEDecoder arguments read from a decoder state dict (decoder.* keys)."""
    count = lambda prefix: len({key[len(prefix):].split(".")[0] for key in sd if key.startswith(prefix)})
    blocks = count("decoder.up_blocks.")
    up = [f"decoder.up_blocks.{i}.upsamplers.0.resample.1.weight" in sd for i in range(blocks)]
    return {
        "latent_channels": sd["decoder.conv_in.conv.weight"].shape[1],
        "out_channels": sd["decoder.conv_out.conv.weight"].shape[0] // 4,
        "block_out_channels": tuple(reversed([sd[f"decoder.up_blocks.{i}.resnets.0.conv1.conv.weight"].shape[0] for i in range(blocks)])),
        "layers_per_block": tuple(reversed([count("decoder.mid_block.resnets.")] + [count(f"decoder.up_blocks.{i}.resnets.") for i in range(blocks)])),
        "upsample": tuple(reversed(up)),
        "spatial_only": tuple(reversed([flag and f"decoder.up_blocks.{i}.upsamplers.0.time_conv.conv.weight" not in sd for i, flag in enumerate(up)])),
    }

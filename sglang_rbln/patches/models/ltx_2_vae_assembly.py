# SPDX-License-Identifier: Apache-2.0
"""Blend and join the LTX-2 VAE's decoded tiles where they were decoded.

Upstream's ``tiled_decode`` decodes every spatial tile, then blends each into its
upper and left neighbours and concatenates them; ``_temporal_tiled_decode`` does
the same across temporal chunks. With resident regions the tiles are rbln
tensors, and each eager blend, slice and cat on them took the tile to the host
first: ~53 ms for each of the 90 tiles of a yak request, then the blend on the
CPU. Here the decode loops stay upstream's, and the joining half of each runs as
a module compiled for the VAE's NPUs. Tiles stay resident; each finished stretch
of video comes down once, into one buffer.
"""

from __future__ import annotations

from typing import Any

import torch
from diffusers.models.autoencoders.vae import DecoderOutput
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

from sglang_rbln.compilation import placement
from sglang_rbln.compilation.backend import static_compile_kwargs

VAE = "sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae.AutoencoderKLLTX2Video"
TILED_DECODE = f"{VAE}.tiled_decode"
TEMPORAL_TILED_DECODE = f"{VAE}._temporal_tiled_decode"
ASSEMBLIES = "_rbln_tile_assemblies"


def blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
    """Upstream's blend_v/h/t without writing into ``b``:
    ``b[y] = a[y - extent] * (1 - y / extent) + b[y] * (y / extent)``.

    It rounds at upstream's steps: each product to the tensor dtype, then their
    sum. Both weights are the float32 values upstream's Python floats take in the
    multiply.
    """
    extent = min(a.shape[dim], b.shape[dim], extent)
    if extent <= 0:
        return b
    shape = [1] * b.dim()
    shape[dim] = extent
    ramp = [y / extent for y in range(extent)]
    keep = (
        torch.tensor([1 - w for w in ramp], dtype=torch.float32)
        .view(shape)
        .to(b.device)
    )
    take = torch.tensor(ramp, dtype=torch.float32).view(shape).to(b.device)
    head = b.narrow(dim, 0, extent)
    tail = a.narrow(dim, a.shape[dim] - extent, extent)
    blended = (tail * keep).to(b.dtype) + (head * take).to(b.dtype)
    return torch.cat([blended, b.narrow(dim, extent, b.shape[dim] - extent)], dim)


class SpatialAssembly(nn.Module):
    """The second half of upstream's ``tiled_decode``, over a grid of decoded tiles."""

    def __init__(
        self, vae, grid: tuple[int, int], sample_size: tuple[int, int]
    ) -> None:
        super().__init__()
        self.grid = grid
        self.sample_size = sample_size
        self.blend = (
            vae.tile_sample_min_height - vae.tile_sample_stride_height,
            vae.tile_sample_min_width - vae.tile_sample_stride_width,
        )
        self.stride = (vae.tile_sample_stride_height, vae.tile_sample_stride_width)

    def forward(self, *tiles: torch.Tensor) -> torch.Tensor:
        count, cols = self.grid
        rows = [list(tiles[i * cols : (i + 1) * cols]) for i in range(count)]
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = blend(rows[i - 1][j], tile, self.blend[0], 3)
                if j > 0:
                    tile = blend(row[j - 1], tile, self.blend[1], 4)
                row[j] = tile
                result_row.append(tile[:, :, :, : self.stride[0], : self.stride[1]])
            result_rows.append(torch.cat(result_row, dim=4))
        height, width = self.sample_size
        return torch.cat(result_rows, dim=3)[:, :, :, :height, :width]


class TemporalStep(nn.Module):
    """One step of the second half of upstream's ``_temporal_tiled_decode``: blend a
    decoded chunk into the previous one and cut the frames it contributes."""

    def __init__(self, vae) -> None:
        super().__init__()
        self.blend = vae.tile_sample_min_num_frames - vae.tile_sample_stride_num_frames
        self.stride = vae.tile_sample_stride_num_frames

    def forward(self, previous: torch.Tensor, decoded: torch.Tensor):
        tile = blend(previous, decoded[:, :, :-1], self.blend, 2)
        return tile[:, :, : self.stride], tile


def _resident(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type == "rbln"


def _assembly(vae, key: tuple, build, *tensors: torch.Tensor) -> nn.Module:
    """``build()``, compiled for the decoder's NPUs when the tensors live there."""
    if not all(map(_resident, tensors)):
        return build()
    assemblies: dict[tuple, nn.Module] = vae.__dict__.setdefault(ASSEMBLIES, {})
    if key not in assemblies:
        module = build()
        if devices := placement.devices_of(vae.decoder):
            placement.place(module, devices)
        module.compile(**static_compile_kwargs(module))
        assemblies[key] = module
    return assemblies[key]


def tiled_decode(self, z, temb, causal=None, return_dict=True):
    """Upstream's decode loop, then its joining half where the tiles are."""
    _, _, _, height, width = z.shape
    ratio = self.spatial_compression_ratio
    min_height = self.tile_sample_min_height // ratio
    min_width = self.tile_sample_min_width // ratio
    stride_height = self.tile_sample_stride_height // ratio
    stride_width = self.tile_sample_stride_width // ratio

    rows = []
    for i in range(0, height, stride_height):
        row = []
        for j in range(0, width, stride_width):
            row.append(
                self.decoder(
                    z[:, :, :, i : i + min_height, j : j + min_width],
                    temb,
                    causal=causal,
                )
            )
        rows.append(row)

    tiles = [tile for row in rows for tile in row]
    grid = (len(rows), len(rows[0]))
    sample_size = (height * ratio, width * ratio)
    key = ("spatial", grid, sample_size, *(tuple(tile.shape) for tile in tiles))
    assembly = _assembly(
        self, key, lambda: SpatialAssembly(self, grid, sample_size), *tiles
    )
    dec = assembly(*tiles)
    return DecoderOutput(sample=dec) if return_dict else (dec,)


def temporal_tiled_decode(self, z, temb, causal=None, return_dict=True):
    """Upstream's temporal loop, joining each chunk into the previous one as it comes
    and writing the finished frames into one host buffer."""
    batch_size, num_channels, num_frames, height, width = z.shape
    num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1
    min_height = self.tile_sample_min_height // self.spatial_compression_ratio
    min_width = self.tile_sample_min_width // self.spatial_compression_ratio
    min_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
    stride_frames = (
        self.tile_sample_stride_num_frames // self.temporal_compression_ratio
    )

    output = None
    written = 0
    previous = None
    for i in range(0, num_frames, stride_frames):
        tile = z[:, :, i : i + min_frames + 1, :, :]
        if self.use_tiling and (
            tile.shape[-1] > min_width or tile.shape[-2] > min_height
        ):
            decoded = self.tiled_decode(
                tile, temb, causal=causal, return_dict=True
            ).sample
        else:
            decoded = self.decoder(tile, temb, causal=causal)
        if previous is None:
            frames, previous = (
                decoded[:, :, : self.tile_sample_stride_num_frames + 1],
                decoded,
            )
        elif decoded.shape[2] <= 1:
            # Upstream drops this chunk's last frame, which leaves it nothing to add.
            continue
        else:
            key = ("temporal", tuple(previous.shape), tuple(decoded.shape))
            step = _assembly(self, key, lambda: TemporalStep(self), previous, decoded)
            frames, previous = step(previous, decoded)
        frames = frames.cpu() if _resident(frames) else frames
        count = min(frames.shape[2], num_sample_frames - written)
        if output is None:
            output = frames.new_empty(
                (*frames.shape[:2], num_sample_frames, *frames.shape[3:])
            )
        output[:, :, written : written + count] = frames[:, :, :count]
        written += count

    return DecoderOutput(sample=output) if return_dict else (output,)


TARGETS = {TILED_DECODE: tiled_decode, TEMPORAL_TILED_DECODE: temporal_tiled_decode}


def register() -> tuple[str, ...]:
    for target, replacement in TARGETS.items():
        HookRegistry.register(target, replacement, HookType.REPLACE)
    return tuple(TARGETS)

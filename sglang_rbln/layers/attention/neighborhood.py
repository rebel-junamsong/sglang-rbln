# SPDX-License-Identifier: Apache-2.0
"""Neighborhood attention as tiled, masked SDPA over static gathers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F

__all__ = ["neighborhood_attention"]


class AxisTiles(NamedTuple):
    queries: torch.Tensor
    halo: torch.Tensor
    mask: torch.Tensor
    count: int
    size: int
    span: int


def axis_tiles(
    length: int, kernel: int, causal: bool, size: int, device: torch.device
) -> AxisTiles:
    """Query tiles of one axis, the key halo of each tile, and the window mask."""
    kernel, size = min(kernel, length), min(size, length)
    span = min(size + kernel - 1, length)
    count = -(-length // size)
    rows = torch.arange(count * size, device=device).view(count, size)
    rows = rows.clamp(max=length - 1)
    if causal:
        low, high = (rows - kernel + 1).clamp(min=0), rows + 1
    else:
        low = (rows - kernel // 2).clamp(0, length - kernel)
        high = low + kernel
    halo = low[:, :1].clamp(max=length - span) + torch.arange(span, device=device)
    keys = halo[:, None, :]
    mask = (keys >= low[..., None]) & (keys < high[..., None])
    return AxisTiles(rows.flatten(), halo.flatten(), mask, count, size, span)


def _gather(x: torch.Tensor, axes: Sequence[AxisTiles], field: str) -> torch.Tensor:
    batch, *grid, heads, dim = x.shape
    x = x.reshape(batch, -1, dim)
    for axis, tiles in enumerate(axes):
        done = x.shape[1 : 1 + axis]
        x = x.reshape(x.shape[0], *done, grid[axis], -1, dim)
        x = x.index_select(1 + axis, getattr(tiles, field))
        length = tiles.size if field == "queries" else tiles.span
        x = x.unflatten(1 + axis, (tiles.count, length))
        x = x.movedim(1 + axis, 1).flatten(0, 1)
    return x.reshape(x.shape[0], -1, heads, dim).transpose(1, 2)


def _scatter(
    x: torch.Tensor, axes: Sequence[AxisTiles], grid: Sequence[int], heads: int
) -> torch.Tensor:
    x = x.transpose(1, 2)
    dim = x.shape[-1]
    x = x.reshape(x.shape[0], *[tiles.size for tiles in axes], -1)
    for axis in reversed(range(len(axes))):
        x = x.unflatten(0, (-1, axes[axis].count)).movedim(1, 1 + axis)
        x = x.flatten(1 + axis, 2 + axis).narrow(1 + axis, 0, grid[axis])
        x = x.flatten(1 + axis, -1)
    return x.reshape(x.shape[0], *grid, heads, dim)


def _window_mask(axes: Sequence[AxisTiles]) -> torch.Tensor:
    mask = axes[0].mask
    for tiles in axes[1:]:
        mask = mask[:, None, :, None, :, None] & tiles.mask[None, :, None, :, None, :]
        mask = mask.flatten(4, 5).flatten(2, 3).flatten(0, 1)
    return mask[:, None]


def default_tile_size(rank: int) -> list[int]:
    return [1, 4, 8] if rank == 3 else [8] * rank


def neighborhood_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel_size: Sequence[int],
    is_causal: Sequence[bool] | None = None,
    tile_size: Sequence[int] | None = None,
) -> torch.Tensor:
    """NATTEN ``na2d`` / ``na3d`` on ``(B, *grid, heads, head_dim)``."""
    batch, *grid, heads, _ = q.shape
    causal = list(is_causal or [False] * len(grid))
    sizes = list(tile_size or default_tile_size(len(grid)))
    axes = [
        axis_tiles(length, kernel, axis_causal, size, q.device)
        for length, kernel, axis_causal, size in zip(
            grid, kernel_size, causal, sizes, strict=True
        )
    ]
    mask = _window_mask(axes).repeat(batch, 1, 1, 1)
    out = F.scaled_dot_product_attention(
        _gather(q, axes, "queries"),
        _gather(k, axes, "halo"),
        _gather(v, axes, "halo"),
        attn_mask=mask,
    )
    return _scatter(out, axes, grid, heads)

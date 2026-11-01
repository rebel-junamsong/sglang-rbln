# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F

__all__ = ["neighborhood_attention"]

SINGLE_TILE_KERNELS = 3
MAX_SCORES_PER_CALL = 1 << 30


class AxisTiles(NamedTuple):
    queries: tuple[int, ...]
    halo: tuple[int, ...]
    mask: torch.Tensor
    positions: tuple[int, ...]
    size: int
    span: int

    @property
    def count(self) -> int:
        return len(self.positions)


def axis_tiles(
    length: int, kernel: int, causal: bool, size: int, device: torch.device
) -> list[AxisTiles]:
    kernel, size = min(kernel, length), min(size, length)
    span = min(size + kernel - 1, length)
    classes: dict[tuple, tuple[list[int], list[int], list[int]]] = {}
    for position in range(-(-length // size)):
        rows = [min(position * size + row, length - 1) for row in range(size)]
        if causal:
            windows = [(max(row - kernel + 1, 0), row + 1) for row in rows]
        else:
            low = [min(max(row - kernel // 2, 0), length - kernel) for row in rows]
            windows = [(start, start + kernel) for start in low]
        first = min(windows[0][0], length - span)
        key = tuple((low - first, high - first) for low, high in windows)
        positions, queries, halo = classes.setdefault(key, ([], [], []))
        positions.append(position)
        queries.extend(rows)
        halo.extend(range(first, first + span))
    return [
        AxisTiles(
            tuple(queries),
            tuple(halo),
            torch.tensor(
                [[low <= key < high for key in range(span)] for low, high in windows],
                device=device,
            ),
            tuple(positions),
            size,
            span,
        )
        for windows, (positions, queries, halo) in classes.items()
    ]


def _chunk(tiles: AxisTiles, length: int) -> list[AxisTiles]:
    return [
        tiles._replace(
            queries=tiles.queries[start * tiles.size : (start + length) * tiles.size],
            halo=tiles.halo[start * tiles.span : (start + length) * tiles.span],
            positions=tiles.positions[start : start + length],
        )
        for start in range(0, tiles.count, length)
    ]


def _take(x: torch.Tensor, dim: int, indices: Sequence[int]) -> torch.Tensor:
    runs: list[list[int]] = []
    for index in indices:
        if runs and index == runs[-1][0] + runs[-1][1]:
            runs[-1][1] += 1
        else:
            runs.append([index, 1])
    if len(runs) == 1 and runs[0] == [0, x.shape[dim]]:
        return x
    return torch.cat([x.narrow(dim, start, length) for start, length in runs], dim)


def _bounded(classes: list[list[AxisTiles]], scores_per_tile: int):
    counts = [max(tiles.count for tiles in axis) for axis in classes]
    while scores_per_tile * math.prod(counts) > MAX_SCORES_PER_CALL:
        axis = max(range(len(counts)), key=counts.__getitem__)
        if counts[axis] == 1:
            break
        counts[axis] = -(-counts[axis] // 2)
    return [
        [chunk for tiles in axis for chunk in _chunk(tiles, length)]
        for axis, length in zip(classes, counts, strict=True)
    ]


def _gather(x: torch.Tensor, axes: Sequence[AxisTiles], field: str) -> torch.Tensor:
    batch, *grid, heads, dim = x.shape
    x = x.reshape(batch, -1, dim)
    for axis, tiles in enumerate(axes):
        done = x.shape[1 : 1 + axis]
        x = x.reshape(x.shape[0], *done, grid[axis], -1, dim)
        x = _take(x, 1 + axis, getattr(tiles, field))
        length = tiles.size if field == "queries" else tiles.span
        x = x.unflatten(1 + axis, (tiles.count, length))
        x = x.movedim(1 + axis, 1).flatten(0, 1)
    return x.reshape(x.shape[0], -1, heads, dim).transpose(1, 2)


def _scatter(
    x: torch.Tensor, sizes: Sequence[int], grid: Sequence[int]
) -> torch.Tensor:
    rank = len(grid)
    batch, *counts, heads, _, dim = x.shape
    x = x.unflatten(-2, tuple(sizes))
    order = [i for axis in range(rank) for i in (1 + axis, 2 + rank + axis)]
    x = x.permute(0, *order, 1 + rank, -1)
    x = x.reshape(
        batch, *(c * s for c, s in zip(counts, sizes, strict=True)), heads, dim
    )
    for axis, length in enumerate(grid):
        x = x.narrow(1 + axis, 0, length)
    return x


def _assemble(blocks: dict[tuple[int, ...], torch.Tensor], shape: Sequence[int]):
    for axis in reversed(range(len(shape))):
        blocks = {
            key: torch.cat([blocks[(*key, c)] for c in range(shape[axis])], 1 + axis)
            for key in itertools.product(*(range(n) for n in shape[:axis]))
        }
    return blocks[()]


def _window_mask(axes: Sequence[AxisTiles]) -> torch.Tensor:
    mask = axes[0].mask
    for tiles in axes[1:]:
        mask = mask[:, None, :, None] & tiles.mask[None, :, None, :]
        mask = mask.flatten(2, 3).flatten(0, 1)
    return mask[None, None]


def default_tile_size(rank: int) -> list[int]:
    return [1, 4, 8] if rank == 3 else [8] * rank


def default_tile_lengths(grid: Sequence[int], kernel_size: Sequence[int]) -> list[int]:
    return [
        length if length <= SINGLE_TILE_KERNELS * kernel else size
        for length, kernel, size in zip(
            grid, kernel_size, default_tile_size(len(grid)), strict=True
        )
    ]


def neighborhood_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel_size: Sequence[int],
    is_causal: Sequence[bool] | None = None,
    tile_size: Sequence[int] | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    batch, *grid, heads, _ = q.shape
    causal = list(is_causal or [False] * len(grid))
    sizes = list(tile_size or default_tile_lengths(grid, kernel_size))
    classes = [
        axis_tiles(length, kernel, axis_causal, size, q.device)
        for length, kernel, axis_causal, size in zip(
            grid, kernel_size, causal, sizes, strict=True
        )
    ]
    tile_scores = batch * heads
    for axis in classes:
        tile_scores *= axis[0].size * axis[0].span
    classes = _bounded(classes, tile_scores)
    blocks = {}
    for key in itertools.product(*(range(len(axis)) for axis in classes)):
        axes = [axis[index] for axis, index in zip(classes, key, strict=True)]
        out = F.scaled_dot_product_attention(
            _gather(q, axes, "queries").flatten(0, 1)[None],
            _gather(k, axes, "halo").flatten(0, 1)[None],
            _gather(v, axes, "halo").flatten(0, 1)[None],
            attn_mask=_window_mask(axes),
            scale=scale,
        )
        counts = (batch, *(tiles.count for tiles in axes), heads)
        blocks[key] = out[0].unflatten(0, counts)
    out = _assemble(blocks, [len(axis) for axis in classes])
    for axis, tiles in enumerate(classes):
        order = [position for cls in tiles for position in cls.positions]
        out = _take(out, 1 + axis, sorted(range(len(order)), key=order.__getitem__))
    return _scatter(out, [axis[0].size for axis in classes], grid)

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from sglang_rbln.compilation.regions import RegionDoesNotFit, _does_not_fit

logger = logging.getLogger(__name__)

# A VAE coder is one graph per tile, so it cannot be split the way a block
# region is. Its own knob is the tile: a smaller tile is a smaller graph.
TILE_ATTRIBUTES = (
    "tile_sample_min_height",
    "tile_sample_min_width",
    "tile_sample_stride_height",
    "tile_sample_stride_width",
)
# Below this the tiles stop paying for themselves: the seams cost more than the
# graph saves, and upstream's own minimum is 256.
MIN_TILE = 128


def _shrink_factor(error: BaseException) -> float:
    """How much smaller the tile has to be, from what the fit check measured.

    A tile's activations scale with its area, so the side shrinks by the square
    root of the memory ratio. Without numbers (a bare SYS_ENOMEM from the
    runtime) fall back to halving the area.
    """
    if not isinstance(error, RegionDoesNotFit):
        return math.sqrt(2)
    ratios = [
        need / room
        for need, room in zip(error.needed, error.free, strict=True)
        if room > 0 and need > room
    ]
    return math.sqrt(max(ratios, default=2.0))


def _rounded(value: float) -> int:
    # The coders halve the spatial axes per down/up block, so keep tiles on a
    # multiple of 32; a ragged tile adds a partial one and another graph.
    return max(MIN_TILE, int(value) // 32 * 32)


def shrink_tiles(module: Any, error: BaseException) -> bool:
    """Make `module`'s tiles smaller. Returns False when there is no room left."""
    sizes = {name: getattr(module, name, None) for name in TILE_ATTRIBUTES}
    if any(size is None for size in sizes.values()):
        return False
    factor = _shrink_factor(error)
    shrunk = {name: _rounded(size / factor) for name, size in sizes.items()}
    if all(
        new >= old for new, old in zip(shrunk.values(), sizes.values(), strict=True)
    ):
        return False
    if shrunk["tile_sample_min_height"] == sizes["tile_sample_min_height"] and (
        shrunk["tile_sample_min_width"] == sizes["tile_sample_min_width"]
    ):
        return False
    module.enable_tiling(**shrunk)
    logger.info(
        "Tiles did not fit (%s); retrying at %s",
        error,
        {
            name: shrunk[name]
            for name in ("tile_sample_min_height", "tile_sample_min_width")
        },
    )
    return True


def with_smaller_tiles(
    call: Callable[[], Any], module: Any, release: Callable[[], None] = lambda: None
) -> Any:
    """Run `call`, shrinking `module`'s tiles while they do not fit the device.

    `release` drops what the given-up tile size left on the device (its graphs'
    runtimes) before the smaller size compiles beside it.
    """
    while True:
        try:
            return call()
        except Exception as error:
            if not _does_not_fit(error) or not shrink_tiles(module, error):
                raise
            release()

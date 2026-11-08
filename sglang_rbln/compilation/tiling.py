# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

from sglang_rbln.compilation.backend import forget_frames
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
# The temporal windows a video decoder falls back to, in sample frames, largest
# first. Each blends into the next over OVERLAP frames. Every blended frame is an
# average of two decodes and loses detail (upstream's 16/8 windows blend half the
# frames: 0.77x the Laplacian variance of an untiled decode), so a decode starts
# untiled and takes the largest window that fits. A fixed ladder keeps the window,
# and so the graphs, the same from one request to the next.
WINDOWS = (64, 32, 16)
OVERLAP = 8


def _shortfall(error: BaseException) -> float | None:
    """How many times the room the graph needed, on its tightest node; None when
    the error carries no numbers (a bare SYS_ENOMEM from the runtime)."""
    if not isinstance(error, RegionDoesNotFit):
        return None
    ratios = [
        need / room
        for need, room in zip(error.needed, error.free, strict=True)
        if room > 0 and need > room
    ]
    return max(ratios, default=None)


def _shrink_factor(error: BaseException) -> float:
    """How much smaller the tile's side has to be: its activations scale with its
    area, so by the square root of the shortfall, or of 2 without one."""
    return math.sqrt(_shortfall(error) or 2.0)


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


def shrink_window(module: Any, error: BaseException, frames: int) -> bool:
    """Decode `module`'s `frames` sample frames in a smaller temporal window.

    It is the largest of WINDOWS the shortfall allows, as activations scale with
    the frames, or without a shortfall the next smaller. Returns False when the
    window is already the smallest, or the module does not decode in windows.
    """
    if not hasattr(module, "use_framewise_decoding"):
        return False
    current = (
        module.tile_sample_min_num_frames if module.use_framewise_decoding else frames
    )
    smaller = [window for window in WINDOWS if window < current]
    if not smaller:
        return False
    shortfall = _shortfall(error)
    target = current / shortfall if shortfall else smaller[0]
    window = next((window for window in smaller if window <= target), smaller[-1])
    module.use_framewise_decoding = True
    module.tile_sample_min_num_frames = window
    module.tile_sample_stride_num_frames = window - OVERLAP
    logger.info(
        "%s frames did not fit (%s); decoding in %s-frame windows",
        current,
        error,
        window,
    )
    return True


def with_smaller_tiles(
    call: Callable[[], Any],
    module: Any,
    release: Callable[[], None] = lambda: None,
    frames: int | None = None,
) -> Any:
    """Run `call`, shrinking `module`'s tiles while they do not fit the device.

    A video decode of `frames` sample frames gives up frames before area: its
    temporal window shrinks to the smallest of WINDOWS before its spatial tiles do.
    `release` drops what the given-up size left on the device (its graphs'
    runtimes) before the smaller size compiles beside it.
    """
    while True:
        try:
            return call()
        except Exception as error:
            if not _does_not_fit(error):
                raise
            shrunk = frames is not None and shrink_window(module, error, frames)
            if not (shrunk or shrink_tiles(module, error)):
                raise
            forget_frames(error)
            release()

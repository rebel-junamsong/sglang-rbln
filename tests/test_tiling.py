# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_rbln.compilation import regions, tiling

MiB = 2**20


class Coder:
    """The tiling surface `AutoencoderKLLTX2Video` offers."""

    def __init__(self, height: int = 512, width: int = 512) -> None:
        self.tile_sample_min_height = height
        self.tile_sample_min_width = width
        self.tile_sample_stride_height = height - 64
        self.tile_sample_stride_width = width - 64
        self.enabled: list[dict] = []

    def enable_tiling(self, **sizes: int) -> None:
        self.enabled.append(sizes)
        for name, size in sizes.items():
            setattr(self, name, size)


def does_not_fit(needed: list[int], free: list[int]) -> regions.RegionDoesNotFit:
    return regions.RegionDoesNotFit(needed, free)


def test_shrinks_the_tile_by_the_measured_shortfall():
    coder = Coder()
    # Four times the room it has: the area must come down 4x, the side 2x.
    assert tiling.shrink_tiles(coder, does_not_fit([400 * MiB], [100 * MiB]))
    assert (coder.tile_sample_min_height, coder.tile_sample_min_width) == (256, 256)
    assert coder.enabled[0]["tile_sample_stride_height"] == 224


def test_shrinks_the_strides_with_the_tile():
    coder = Coder()
    tiling.shrink_tiles(coder, does_not_fit([200 * MiB], [100 * MiB]))
    assert coder.tile_sample_min_height > coder.tile_sample_stride_height


def test_halves_the_area_when_the_error_carries_no_numbers():
    coder = Coder()
    assert tiling.shrink_tiles(coder, RuntimeError("SYS_ENOMEM"))
    assert coder.tile_sample_min_height == 352


def test_judges_every_node_by_its_own_shortfall():
    coder = Coder()
    # Node 1 is the tight one; the tile has to suit it, not the average.
    tiling.shrink_tiles(
        coder, does_not_fit([100 * MiB, 900 * MiB], [200 * MiB, 100 * MiB])
    )
    assert coder.tile_sample_min_height == 160


def test_refuses_to_shrink_below_the_floor():
    coder = Coder(height=tiling.MIN_TILE, width=tiling.MIN_TILE)
    assert not tiling.shrink_tiles(coder, does_not_fit([900 * MiB], [1 * MiB]))
    assert coder.enabled == []


def test_leaves_a_module_that_does_not_tile():
    module = SimpleNamespace()
    assert not tiling.shrink_tiles(module, does_not_fit([900 * MiB], [1 * MiB]))


def test_retries_the_call_on_smaller_tiles_until_it_fits():
    coder = Coder()
    attempts = []

    def call():
        attempts.append(coder.tile_sample_min_height)
        if coder.tile_sample_min_height > 256:
            raise does_not_fit([400 * MiB], [100 * MiB])
        return "decoded"

    assert tiling.with_smaller_tiles(call, coder) == "decoded"
    assert attempts == [512, 256]


def test_gives_up_when_the_tile_cannot_shrink_further():
    coder = Coder(height=tiling.MIN_TILE, width=tiling.MIN_TILE)

    def call():
        raise does_not_fit([900 * MiB], [1 * MiB])

    with pytest.raises(regions.RegionDoesNotFit):
        tiling.with_smaller_tiles(call, coder)


def test_lets_an_unrelated_failure_through():
    coder = Coder()

    def call():
        raise ValueError("not a memory problem")

    with pytest.raises(ValueError):
        tiling.with_smaller_tiles(call, coder)
    assert coder.enabled == []

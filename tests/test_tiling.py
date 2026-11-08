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

    released = []
    release = lambda: released.append(coder.tile_sample_min_height)  # noqa: E731
    assert tiling.with_smaller_tiles(call, coder, release) == "decoded"
    assert attempts == [512, 256]
    assert released == [256]  # once, after shrinking and before the retry


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


class VideoCoder(Coder):
    """A coder that also decodes in temporal windows, as the LTX-2 VAE does."""

    def __init__(self) -> None:
        super().__init__()
        self.use_framewise_decoding = False
        self.tile_sample_min_num_frames = 16
        self.tile_sample_stride_num_frames = 8


def window(coder: VideoCoder) -> tuple[int, int] | None:
    if not coder.use_framewise_decoding:
        return None
    return coder.tile_sample_min_num_frames, coder.tile_sample_stride_num_frames


def test_a_decode_that_fits_keeps_every_frame_in_one_window():
    coder = VideoCoder()
    assert tiling.with_smaller_tiles(lambda: "video", coder, frames=121) == "video"
    assert window(coder) is None


def test_takes_the_largest_window_the_shortfall_allows():
    coder = VideoCoder()
    # 121 frames needed 1.5x the room: 80 would fit, the ladder's largest is 64.
    assert tiling.shrink_window(coder, does_not_fit([150 * MiB], [100 * MiB]), 121)
    assert window(coder) == (64, 56)
    # 4x: 30 frames would fit, so 16, not the next rung down.
    coder = VideoCoder()
    assert tiling.shrink_window(coder, does_not_fit([400 * MiB], [100 * MiB]), 121)
    assert window(coder) == (16, 8)


@pytest.mark.parametrize(
    ("start", "after"), [(None, (64, 56)), ((64, 56), (32, 24)), ((32, 24), (16, 8))]
)
def test_a_window_that_does_not_fit_takes_the_next_rung(start, after):
    coder = VideoCoder()
    if start is not None:
        coder.use_framewise_decoding = True
        coder.tile_sample_min_num_frames, coder.tile_sample_stride_num_frames = start
    # A bare SYS_ENOMEM carries no shortfall: one rung at a time.
    assert tiling.shrink_window(coder, RuntimeError("SYS_ENOMEM"), 121)
    assert window(coder) == after


def test_frames_shrink_before_the_spatial_tiles():
    coder = VideoCoder()
    attempts = []

    def call():
        attempts.append((window(coder), coder.tile_sample_min_height))
        if coder.tile_sample_min_height > 256:
            raise RuntimeError("SYS_ENOMEM")
        return "video"

    assert tiling.with_smaller_tiles(call, coder, frames=121) == "video"
    assert attempts == [
        (None, 512),
        ((64, 56), 512),
        ((32, 24), 512),
        ((16, 8), 512),
        ((16, 8), 352),
        ((16, 8), 224),
    ]


def test_the_window_holds_for_the_next_request():
    coder = VideoCoder()
    sizes = iter([does_not_fit([150 * MiB], [100 * MiB])])

    def call():
        error = next(sizes, None)
        if error is not None:
            raise error
        return window(coder)

    assert tiling.with_smaller_tiles(call, coder, frames=121) == (64, 56)
    assert tiling.with_smaller_tiles(call, coder, frames=121) == (64, 56)


def test_a_short_video_never_takes_a_window_as_long_as_itself():
    coder = VideoCoder()
    # 33 frames: only the 32 and 16 rungs are shorter.
    assert tiling.shrink_window(coder, does_not_fit([150 * MiB], [100 * MiB]), 33)
    assert window(coder) == (16, 8)


def test_an_eager_tensor_without_room_shrinks_the_window():
    # The LTX 1280x704 decode failed this way: an eager tensor of the untiled decode
    # found no room even after the runtime evicted idle programs.
    error = RuntimeError(
        "(Runtime) code=202 INIT_INVALID_ARGUMENT "
        "(RBLNCachingAllocator::malloc block != nullptr)"
    )
    coder = VideoCoder()
    attempts = []

    def call():
        attempts.append(window(coder))
        if not coder.use_framewise_decoding:
            raise error
        return "video"

    assert tiling.with_smaller_tiles(call, coder, frames=121) == "video"
    assert attempts == [None, (64, 56)]

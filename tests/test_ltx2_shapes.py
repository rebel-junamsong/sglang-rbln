# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector import (
    LTX2RotaryPosEmbed1d,
)
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    LTX2AudioVideoRotaryPosEmbed,
)

from sglang_rbln.patches.models import ltx_2


def assert_same_contiguous(hooked, expected):
    for got, want in zip(hooked, expected, strict=True):
        assert got.is_contiguous()
        assert got.dtype == want.dtype
        assert torch.equal(got, want)


@pytest.mark.parametrize("rope_type", ["interleaved", "split"])
@pytest.mark.parametrize(
    ("frames", "height", "width"),
    [
        pytest.param(1, 17, 30, id="image-landscape"),
        pytest.param(1, 30, 17, id="image-portrait"),
        pytest.param(2, 13, 23, id="short-odd"),
        pytest.param(16, 17, 30, id="default-960x544x121"),
        pytest.param(16, 30, 17, id="default-portrait"),
        pytest.param(31, 7, 11, id="long-odd"),
    ],
)
def test_video_rotary_tables_match_upstream(rope_type, frames, height, width):
    rope = LTX2AudioVideoRotaryPosEmbed(
        dim=128, rope_type=rope_type, num_attention_heads=4
    )
    coords = rope.prepare_video_coords(
        2, frames, height, width, torch.device("cpu"), fps=24.0
    )
    expected = rope(coords, out_dtype=torch.bfloat16)
    assert_same_contiguous(ltx_2.contiguous_rotary_tables(expected), expected)


@pytest.mark.parametrize("rope_type", ["interleaved", "split"])
@pytest.mark.parametrize("frames", [1, 7, 121, 257])
def test_audio_rotary_tables_match_upstream(rope_type, frames):
    rope = LTX2AudioVideoRotaryPosEmbed(
        dim=64,
        patch_size=1,
        scale_factors=(4,),
        modality="audio",
        rope_type=rope_type,
        num_attention_heads=2,
    )
    coords = rope.prepare_audio_coords(2, frames, torch.device("cpu"))
    expected = rope(coords, out_dtype=torch.bfloat16)
    assert_same_contiguous(ltx_2.contiguous_rotary_tables(expected), expected)


@pytest.mark.parametrize("rope_type", ["interleaved", "split"])
@pytest.mark.parametrize("tokens", [1, 77, 128, 1023])
def test_connector_rotary_tables_match_upstream(rope_type, tokens):
    rope = LTX2RotaryPosEmbed1d(128, rope_type=rope_type, num_attention_heads=4)
    expected = rope(2, tokens, torch.device("cpu"), torch.bfloat16)
    assert_same_contiguous(ltx_2.contiguous_rotary_tables(expected), expected)


class _CountingRope(LTX2AudioVideoRotaryPosEmbed):
    calls = 0

    def forward(self, *args, **kwargs):
        type(self).calls += 1
        return super().forward(*args, **kwargs)


def test_rotary_tables_are_built_once_per_position_set():
    rope = _CountingRope(dim=128, rope_type="split", num_attention_heads=4)
    forward = _CountingRope.forward

    def tables(frames):
        coords = rope.prepare_video_coords(
            2, frames, 17, 30, torch.device("cpu"), fps=24.0
        )
        return ltx_2.cached_rotary_tables(
            forward, rope, coords, out_dtype=torch.bfloat16
        )

    first = tables(16)
    assert tables(16) is first  # a new coords tensor of the same positions: no rebuild
    assert _CountingRope.calls == 1
    other = tables(15)
    assert _CountingRope.calls == 2 and other[0].shape != first[0].shape
    expected = forward(
        rope,
        rope.prepare_video_coords(2, 16, 17, 30, torch.device("cpu"), fps=24.0),
        out_dtype=torch.bfloat16,
    )
    assert_same_contiguous(first, expected)
    assert all(table.is_contiguous() for table in first)
    for frames in range(ltx_2.ROTARY_TABLES_KEPT + 2):
        tables(frames + 1)
    assert len(rope.__dict__[ltx_2.ROTARY_TABLES]) == ltx_2.ROTARY_TABLES_KEPT

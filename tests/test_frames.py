# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest
import torch
from sglang.multimodal_gen.runtime.entrypoints import utils
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.patches.entrypoints import frames

UPSTREAM = utils._sample_to_uint8_frames
EDGES = [0, 1, -0.0, 1e-9, 1 - 1e-7, 0.5 / 255, 254.5 / 255, 2, -3, np.inf, -np.inf]


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_frames_match_upstream_byte_for_byte(channels):
    rng = np.random.default_rng(0)
    sample = rng.random((5, 17, 33, channels), dtype=np.float32) * 1.2 - 0.1
    sample.reshape(-1)[: len(EDGES)] = EDGES
    expected = UPSTREAM(sample)
    actual = frames.sample_to_uint8_frames(UPSTREAM, sample)
    assert len(actual) == len(expected)
    assert all(
        a.dtype == np.uint8 and np.array_equal(a, e)
        for a, e in zip(actual, expected, strict=True)
    )


def test_every_value_in_range_rounds_as_upstream():
    # 12M evenly spaced values just past [0, 1], as four 1000x1000 RGB frames
    sample = np.linspace(-0.01, 1.01, 3 * 4_000_000, dtype=np.float32)
    sample = sample.reshape(-1, 1000, 1000, 3)
    expected = UPSTREAM(sample)
    actual = frames.sample_to_uint8_frames(UPSTREAM, sample)
    assert all(np.array_equal(a, e) for a, e in zip(actual, expected, strict=True))


@pytest.mark.parametrize(
    "sample",
    [
        torch.rand(3, 2, 8, 8),
        np.zeros((2, 8, 8, 3), np.uint8),
        np.zeros((2, 8, 8, 3), np.float64),
        np.zeros((3, 2, 8, 8), np.float32),
        np.zeros((8, 8, 3), np.float32),
    ],
    ids=["tensor", "uint8", "float64", "channels-first", "one-frame"],
)
def test_other_sample_types_and_layouts_go_to_upstream(sample):
    def upstream(value):
        assert value is sample
        return "upstream"

    assert frames.sample_to_uint8_frames(upstream, sample) == "upstream"


def test_wraps_the_upstream_conversion():
    assert frames.register() == frames.HOOK_TARGET
    assert any(
        hook is frames.sample_to_uint8_frames and kind is HookType.AROUND
        for kind, hook, _ in HookRegistry._hooks[frames.HOOK_TARGET]
    )

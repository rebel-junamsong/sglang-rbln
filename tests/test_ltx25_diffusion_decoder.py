# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from sglang.multimodal_gen.runtime.models.decoders import ltx_2_5_diffusion_decoder
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry

from sglang_rbln.layers.attention.neighborhood import neighborhood_attention
from sglang_rbln.patches.models import ltx_2


def test_hands_the_decoder_the_tiled_neighborhood_attention():
    assert ltx_2.DIFFUSION_DECODER_NA3D in ltx_2.register()
    hooks = [hook for _, hook, _ in HookRegistry._hooks[ltx_2.DIFFUSION_DECODER_NA3D]]
    assert ltx_2.na3d in hooks
    assert ltx_2.na3d() is neighborhood_attention


@pytest.mark.parametrize(
    ("grid", "kernel"),
    [
        ((5, 9, 10), (3, 7, 7)),
        ((4, 7, 9), (3, 5, 5)),
        ((7, 6, 11), (3, 5, 5)),
        ((13, 14, 16), (11, 11, 11)),
        ((11, 23, 12), (11, 11, 11)),
    ],
    ids=str,
)
def test_matches_the_flex_attention_decoder(monkeypatch, grid, kernel):
    torch.manual_seed(0)
    attention = ltx_2_5_diffusion_decoder.LTX2VideoVaeNeighborhoodAttention(
        128, kernel
    ).eval()
    hidden_states = torch.randn(1, *grid, 128)
    with torch.no_grad():
        expected = attention(hidden_states)
        monkeypatch.setattr(ltx_2_5_diffusion_decoder, "_na3d", ltx_2.na3d)
        actual = attention(hidden_states)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import AutoencoderKLLTX2Video
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.patches.models import ltx_2

UPSTREAM = {
    name: getattr(AutoencoderKLLTX2Video, name)
    for name in ("blend_v", "blend_h", "blend_t")
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("name", ["blend_v", "blend_h", "blend_t"])
@pytest.mark.parametrize("extent", [0, 1, 7, 64, 200])
def test_blend_matches_upstream_bit_for_bit(name, dtype, extent):
    assert UPSTREAM[name].__module__ == AutoencoderKLLTX2Video.__module__
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(1, 3, 9, 96, 80, generator=generator).to(dtype)
    b = torch.randn(1, 3, 9, 96, 80, generator=generator).to(dtype)
    expected = UPSTREAM[name](None, a.clone(), b.clone(), extent)
    actual = getattr(ltx_2, name)(None, a.clone(), b.clone(), extent)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("target", ltx_2.BLENDS)
def test_replaces_the_upstream_blend(target):
    assert target in ltx_2.register()
    assert any(
        hook is ltx_2.BLENDS[target] and kind is HookType.REPLACE
        for kind, hook, _ in HookRegistry._hooks[target]
    )

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextvars
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.models.vaes import wanvae as upstream_vae

from sglang_rbln.patches.models import cosmos3video, wanvae

pytestmark = pytest.mark.usefixtures("rbln_platform")

GRIDS = pytest.mark.parametrize(
    ("frames", "height", "width"),
    [
        pytest.param(1, 20, 20, id="image-square"),
        pytest.param(1, 15, 26, id="image-landscape"),
        pytest.param(1, 26, 15, id="image-portrait"),
        pytest.param(2, 13, 7, id="short-odd"),
        pytest.param(21, 15, 26, id="long-landscape"),
        pytest.param(21, 26, 15, id="long-portrait"),
        pytest.param(31, 9, 11, id="longest-odd"),
    ],
)


@pytest.fixture
def transformer() -> SimpleNamespace:
    return SimpleNamespace(
        temporal_margin=15,
        base_fps=24.0,
        temporal_compression_factor=4,
        sound_latent_fps=25.0,
        temporal_compression_factor_sound=1,
    )


@GRIDS
@pytest.mark.parametrize("fps", [None, 16.0])
def test_rope_positions_match_upstream(transformer, frames, height, width, fps):
    from sglang.multimodal_gen.runtime.models.dits import cosmos3video as dit

    text_mask = (torch.arange(13) < torch.tensor([[7], [13]])).long()
    args = (text_mask, frames, height, width, fps, torch.device("cpu"))
    expected = dit.Cosmos3OmniTransformer._compute_rope_position_ids(transformer, *args)
    actual = cosmos3video.rope_position_ids(transformer, *args)
    for got, want in zip(actual, expected, strict=True):
        assert got.dtype == want.dtype
        assert torch.equal(got, want)


@GRIDS
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rope_from_cache_matches_upstream(frames, height, width, dtype):
    from sglang.multimodal_gen.runtime.models.dits import cosmos3video as dit

    generator = torch.Generator().manual_seed(0)
    tokens, head_dim = frames * height * width, 128
    q = torch.randn(1, tokens, 4, head_dim, generator=generator).to(dtype)
    k = torch.randn(1, tokens, 2, head_dim, generator=generator).to(dtype)
    cos_sin_cache = torch.randn(tokens, head_dim, generator=generator)
    expected = dit._apply_qwen3_rope_from_cache(q, k, cos_sin_cache)
    actual = cosmos3video.rope_from_cache(q, k, cos_sin_cache)
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize(
    ("frames", "height", "width"),
    [
        pytest.param(1, 4, 4, id="image-square"),
        pytest.param(1, 3, 7, id="image-landscape"),
        pytest.param(1, 7, 3, id="image-portrait"),
        pytest.param(2, 5, 9, id="short-landscape"),
        pytest.param(6, 9, 5, id="long-portrait"),
    ],
)
def test_traceable_vae_decode_matches_upstream(
    monkeypatch, tiny_wan_vae, frames, height, width
):
    for name in wanvae.CONTEXT_VARS:
        monkeypatch.setattr(
            upstream_vae, name, contextvars.ContextVar(name, default=None)
        )
    latents = torch.randn(
        1, 4, frames, height, width, generator=torch.Generator().manual_seed(0)
    )
    with torch.no_grad():
        expected = tiny_wan_vae.decode(latents)
        wanvae.register()
        actual = tiny_wan_vae.decode(latents)
    assert torch.equal(actual, expected)

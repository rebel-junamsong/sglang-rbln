# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import AutoencoderKLLTX2Video
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.patches.models import ltx_2_vae_assembly as assembly

UPSTREAM = {
    name: getattr(AutoencoderKLLTX2Video, name)
    for name in (
        "tiled_decode",
        "_temporal_tiled_decode",
        "blend_v",
        "blend_h",
        "blend_t",
    )
}


def fake_vae(
    dtype, tiled_decode, temporal_tiled_decode, *, frames=24, size=96, stride=64
):
    """A VAE whose decoder is a fixed random upsampling, tiled like LTX-2's."""
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(3, 4, generator=generator).to(dtype)

    def decoder(z, temb, causal=None):
        x = torch.einsum("oc,bcthw->bothw", weight, z)
        x = x.repeat_interleave(8, 3).repeat_interleave(8, 4)
        return torch.cat([x[:, :, :1], x.repeat_interleave(4, 2)[:, :, 4:]], 2)

    vae = SimpleNamespace(
        decoder=decoder,
        use_tiling=True,
        spatial_compression_ratio=8,
        temporal_compression_ratio=4,
        tile_sample_min_height=size,
        tile_sample_min_width=size,
        tile_sample_stride_height=stride,
        tile_sample_stride_width=stride,
        tile_sample_min_num_frames=frames,
        tile_sample_stride_num_frames=frames - 8,
    )
    for name in ("blend_v", "blend_h", "blend_t"):
        setattr(vae, name, MethodType(UPSTREAM[name], vae))
    vae.tiled_decode = MethodType(tiled_decode, vae)
    vae._temporal_tiled_decode = MethodType(temporal_tiled_decode, vae)
    return vae


def latent(dtype, frames=9, height=20, width=28):
    generator = torch.Generator().manual_seed(1)
    return torch.randn(1, 4, frames, height, width, generator=generator).to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("name", ["blend_v", "blend_h", "blend_t"])
@pytest.mark.parametrize("extent", [0, 1, 7, 64, 200])
def test_blend_matches_upstream_bit_for_bit(name, dtype, extent):
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(1, 3, 9, 96, 80, generator=generator).to(dtype)
    b = torch.randn(1, 3, 9, 96, 80, generator=generator).to(dtype)
    dim = {"blend_v": 3, "blend_h": 4, "blend_t": 2}[name]
    expected = UPSTREAM[name](None, a.clone(), b.clone(), extent)
    actual = assembly.blend(a, b, extent, dim)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape", [(9, 20, 28), (9, 8, 8), (5, 12, 20)], ids=["2x3", "1x1", "short"]
)
def test_spatial_join_matches_upstream_bit_for_bit(dtype, shape):
    z = latent(dtype, *shape)
    upstream = fake_vae(
        dtype, UPSTREAM["tiled_decode"], UPSTREAM["_temporal_tiled_decode"]
    )
    ours = fake_vae(dtype, assembly.tiled_decode, assembly.temporal_tiled_decode)
    expected = upstream.tiled_decode(z, None).sample
    actual = ours.tiled_decode(z, None).sample
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("frames", [9, 13, 16], ids=["chunks", "short-last", "uneven"])
def test_temporal_join_matches_upstream_bit_for_bit(dtype, frames):
    z = latent(dtype, frames)
    upstream = fake_vae(
        dtype, UPSTREAM["tiled_decode"], UPSTREAM["_temporal_tiled_decode"], frames=16
    )
    ours = fake_vae(
        dtype, assembly.tiled_decode, assembly.temporal_tiled_decode, frames=16
    )
    expected = upstream._temporal_tiled_decode(z, None).sample
    actual = ours._temporal_tiled_decode(z, None).sample
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_host_tiles_are_not_compiled(monkeypatch):
    def compile_called(*args, **kwargs):
        raise AssertionError("host tiles must not compile an assembly")

    monkeypatch.setattr(torch.nn.Module, "compile", compile_called)
    ours = fake_vae(
        torch.float32, assembly.tiled_decode, assembly.temporal_tiled_decode
    )
    ours._temporal_tiled_decode(latent(torch.float32), None)


@pytest.mark.parametrize("target", assembly.TARGETS)
def test_replaces_the_upstream_joins(target):
    assert target in assembly.register()
    assert any(
        hook is assembly.TARGETS[target] and kind is HookType.REPLACE
        for kind, hook, _ in HookRegistry._hooks[target]
    )

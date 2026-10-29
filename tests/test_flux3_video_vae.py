# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln.layers.attention.neighborhood import neighborhood_attention
from sglang_rbln.patches.models import flux3_video_vae

pytestmark = pytest.mark.usefixtures("rbln_platform")


class _Block(nn.Linear):
    pass


def _coder() -> nn.Module:
    coder = nn.Module()
    coder.proj = nn.Linear(4, 4)
    coder.features = nn.Sequential(
        nn.Sequential(_Block(4, 4), _Block(4, 4)), _Block(4, 4)
    )
    return coder


@pytest.fixture
def vae():
    vae = nn.Module()
    vae.model = nn.Module()
    vae.model.encoder, vae.model.decoder = _coder(), _coder().float()
    return vae


@pytest.fixture
def compile_vae(monkeypatch):
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_VAE", "1")


@pytest.fixture
def compiled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs))
    )
    return calls


def test_leaves_the_host_vae_alone_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_RBLN_COMPILE_VAE", raising=False)
    assert flux3_video_vae.register() == ()


def test_registers_the_attention_and_the_compile_hooks(compile_vae):
    targets = flux3_video_vae.register()
    assert targets == (flux3_video_vae.ATTENTION_TARGET, flux3_video_vae.LOAD_TARGET)
    hooks = {
        target: [hook for _, hook, _ in HookRegistry._hooks[target]]
        for target in targets
    }
    assert neighborhood_attention in hooks[flux3_video_vae.ATTENTION_TARGET]
    assert flux3_video_vae.compile_regions in hooks[flux3_video_vae.LOAD_TARGET]


def test_regions_flatten_sequential_stages():
    regions = list(flux3_video_vae.regions(_coder()))
    assert [type(region) for region in regions] == [nn.Linear, _Block, _Block, _Block]


def test_compiles_every_region_of_both_coders_in_bfloat16(vae, compiled):
    server_args = SimpleNamespace(enable_torch_compile=True)
    assert flux3_video_vae.compile_regions(vae, server_args, None) is vae
    assert len(compiled) == 8
    assert all(kwargs["backend"] == "rbln" for _, kwargs in compiled)
    assert all(kwargs["dynamic"] is False for _, kwargs in compiled)
    assert vae.model.decoder.proj.weight.dtype == torch.bfloat16


def test_compiled_rotary_tables_stay_float32(vae, compiled):
    from sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae import (
        RotaryPositionEmbedding3D,
    )

    for coder in (vae.model.encoder, vae.model.decoder):
        coder.rope = RotaryPositionEmbedding3D(64).to(coder.proj.weight.dtype)
    expected = vae.model.encoder.rope.inv_freq.float()
    flux3_video_vae.compile_regions(
        vae, SimpleNamespace(enable_torch_compile=True), None
    )
    for coder in (vae.model.encoder, vae.model.decoder):
        assert coder.rope.inv_freq.dtype == torch.float32
        assert "inv_freq" in dict(coder.rope.named_buffers())
        torch.testing.assert_close(coder.rope.inv_freq, expected, rtol=0, atol=0)
    assert vae.model.decoder.proj.weight.dtype == torch.bfloat16


def test_spreads_decoder_regions_over_the_tensor_parallel_npus(
    vae, compiled, monkeypatch
):
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "2")
    flux3_video_vae.compile_regions(
        vae, SimpleNamespace(enable_torch_compile=True), None
    )
    encoder = list(flux3_video_vae.regions(vae.model.encoder))
    devices = {id(module): kwargs["options"]["device"] for module, kwargs in compiled}
    assert all(kwargs["options"]["num_devices"] == 1 for _, kwargs in compiled)
    assert [devices[id(region)] for region in encoder] == [0] * len(encoder)
    assert [
        devices[id(region)] for region in flux3_video_vae.regions(vae.model.decoder)
    ] == [0, 1, 0, 1]


def test_keeps_the_float32_decoder_without_compile(vae, compiled):
    server_args = SimpleNamespace(enable_torch_compile=False)
    flux3_video_vae.compile_regions(vae, server_args, None)
    assert compiled == []
    assert vae.model.decoder.proj.weight.dtype == torch.float32


def test_compiles_the_decoder_regions_largest_first(monkeypatch):
    from sglang.multimodal_gen.runtime.models.vaes import flux3_video_vae as upstream

    monkeypatch.setattr(upstream, "neighborhood_attention", neighborhood_attention)

    decoder = upstream.DecoderSwin3D(
        z_ch=4,
        patch_size=[1, 2, 2],
        embed_dim=8,
        depths=[1, 1],
        temporal=[True, False],
        num_heads=[1, 1],
        window_size=[3, 3, 3],
        max_t=3,
    ).eval()
    calls = []
    order = [
        decoder.proj_in,
        *flux3_video_vae.regions(decoder.features),
        decoder.proj_out,
    ]
    for region in order:
        region.register_forward_pre_hook(
            lambda module, args: calls.append((module, args[0].shape))
        )
    decoder.compiled_shapes = set()
    decoder.register_forward_pre_hook(flux3_video_vae.compile_largest_first)
    latents = torch.randn(1, 4, 4, 4, 4)

    with torch.no_grad():
        decoder(latents)
        warmup, decode = calls[: len(order)], calls[len(order) :]
        calls.clear()
        decoder(latents)

    assert [module for module, _ in decode] == order
    assert warmup == decode[::-1]
    assert calls == decode

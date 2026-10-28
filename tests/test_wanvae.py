# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextvars

import pytest
import torch
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.runtime.models.vaes import wanvae as upstream

from sglang_rbln.patches.models import wanvae

pytestmark = pytest.mark.usefixtures("rbln_platform")


@pytest.fixture
def context_vars(monkeypatch):
    for name in wanvae.CONTEXT_VARS:
        monkeypatch.setattr(upstream, name, contextvars.ContextVar(name, default=None))


@pytest.fixture
def tiny_vae() -> upstream.AutoencoderKLWan:
    config = WanVAEConfig(load_encoder=False, use_parallel_decode=False)
    arch = config.arch_config
    arch.base_dim = arch.decoder_base_dim = 16
    arch.z_dim, arch.is_residual, arch.patch_size = 4, True, 2
    arch.in_channels = arch.out_channels = 12
    arch.latents_mean, arch.latents_std = arch.latents_mean[:4], arch.latents_std[:4]
    arch.__post_init__()
    return upstream.AutoencoderKLWan(config).eval()


def decode_graphs(vae: upstream.AutoencoderKLWan) -> int:
    torch._dynamo.reset()
    with torch.no_grad():
        explained = torch._dynamo.explain(vae.decode)(torch.randn(1, 4, 3, 4, 6))
    return explained.graph_count


def test_traceable_var_behaves_like_a_context_var():
    var = wanvae.TraceableContextVar("feat_idx", 0)
    token = var.set(3)
    assert var.get() == 3
    var.reset(token)
    assert var.get() == 0


@pytest.mark.usefixtures("context_vars")
def test_register_replaces_each_context_var_once():
    assert wanvae.register() == list(wanvae.CONTEXT_VARS)
    assert wanvae.register() == []
    assert all(
        isinstance(getattr(upstream, name), wanvae.TraceableContextVar)
        for name in wanvae.CONTEXT_VARS
    )


@pytest.mark.usefixtures("context_vars")
def test_context_vars_split_the_decoder(tiny_vae):
    assert decode_graphs(tiny_vae) > 100


@pytest.mark.usefixtures("context_vars")
def test_traceable_feature_cache_keeps_the_decoder_whole(tiny_vae):
    latents = torch.randn(1, 4, 3, 4, 6, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = tiny_vae.decode(latents)
        wanvae.register()
        assert decode_graphs(tiny_vae) <= 5
        assert torch.equal(tiny_vae.decode(latents), expected)

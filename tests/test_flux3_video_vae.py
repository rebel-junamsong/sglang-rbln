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


@pytest.fixture
def compile_vae(monkeypatch):
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_VAE", "1")


def test_leaves_the_host_vae_alone_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_RBLN_COMPILE_VAE", raising=False)
    assert flux3_video_vae.register() == ()


def test_registers_the_attention_and_the_compile_hooks(compile_vae):
    targets = flux3_video_vae.register()
    assert targets == (
        flux3_video_vae.ATTENTION_TARGET,
        flux3_video_vae.NATTEN_TARGET,
        flux3_video_vae.LOAD_TARGET,
    )
    hooks = {
        target: [hook for _, hook, _ in HookRegistry._hooks[target]]
        for target in targets
    }
    assert neighborhood_attention in hooks[flux3_video_vae.ATTENTION_TARGET]
    assert flux3_video_vae.natten_available in hooks[flux3_video_vae.NATTEN_TARGET]
    assert flux3_video_vae.compile_coders in hooks[flux3_video_vae.LOAD_TARGET]


@pytest.mark.parametrize(("causal", "max_t"), [(False, 4), (True, 3), (False, None)])
def test_a_looped_block_traces_as_one_graph(monkeypatch, causal, max_t):
    from sglang.multimodal_gen.runtime.models.vaes import flux3_video_vae as upstream

    monkeypatch.setattr(upstream, "neighborhood_attention", neighborhood_attention)
    monkeypatch.setattr(upstream, "natten_available", flux3_video_vae.natten_available)
    torch.manual_seed(0)
    block = upstream.SwinTransformerBlock(
        64, 2, [3, 5, 5], causal=causal, qk_norm=True, max_t=max_t
    ).eval()
    x = torch.randn(1, 9, 6, 8, 64)

    with torch.no_grad():
        explanation = torch._dynamo.explain(block)(x.clone())
        expected = block(x.clone())
        compiled = torch.compile(block, backend="eager", fullgraph=True)(x.clone())

    assert (explanation.graph_count, explanation.graph_break_count) == (1, 0)
    torch.testing.assert_close(compiled, expected, rtol=0, atol=0)


def _upstream_vae(monkeypatch):
    from sglang.multimodal_gen.runtime.models.vaes import flux3_video_vae as upstream

    monkeypatch.setattr(upstream, "neighborhood_attention", neighborhood_attention)
    monkeypatch.setattr(upstream, "natten_available", flux3_video_vae.natten_available)
    common = dict(
        patch_size=[1, 2, 2],
        embed_dim=8,
        depths=[1, 2],
        temporal=[True, False],
        num_heads=[1, 2],
        window_size=[3, 3, 3],
    )
    torch.manual_seed(0)
    vae = nn.Module()
    vae.model = nn.Module()
    vae.model.encoder = upstream.EncoderSwin3D(z_ch=8, **common).eval()
    vae.model.decoder = upstream.DecoderSwin3D(z_ch=4, max_t=3, **common).eval()
    return vae


@pytest.fixture
def compiled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs))
    )
    return calls


def test_compiles_each_coder_as_one_graph_on_the_last_tensor_parallel_npu(
    monkeypatch, compiled
):
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    vae = _upstream_vae(monkeypatch)
    server_args = SimpleNamespace(enable_torch_compile=True)

    assert flux3_video_vae.compile_coders(vae, server_args, None) is vae

    assert [module for module, _ in compiled] == [vae.model.encoder, vae.model.decoder]
    options = [kwargs["options"] for _, kwargs in compiled]
    assert all((o["num_devices"], o["device"]) == (1, 3) for o in options)
    assert all(kwargs["dynamic"] is False for _, kwargs in compiled)
    assert vae.model.decoder.proj_in.weight.dtype == torch.bfloat16


def test_coders_keep_the_devices_declared_for_their_component(monkeypatch, compiled):
    from sglang_rbln.compilation import placement

    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    vae = _upstream_vae(monkeypatch)
    monkeypatch.setattr(placement, "devices_of", lambda module: (1, 2))
    flux3_video_vae.compile_coders(
        vae, SimpleNamespace(enable_torch_compile=True), None
    )
    options = [kwargs["options"] for _, kwargs in compiled]
    assert all((o["num_devices"], o["device"]) == (2, [1, 2]) for o in options)


def test_keeps_the_float32_decoder_without_compile(monkeypatch, compiled):
    vae = _upstream_vae(monkeypatch)
    flux3_video_vae.compile_coders(
        vae, SimpleNamespace(enable_torch_compile=False), None
    )
    assert compiled == []
    assert vae.model.decoder.proj_in.weight.dtype == torch.float32


@pytest.mark.parametrize(
    ("coder", "shape"),
    [
        ("decoder", (1, 4, 3, 3, 5)),
        ("decoder", (1, 4, 4, 5, 3)),
        ("decoder", (1, 4, 2, 2, 2)),
        ("encoder", (1, 3, 5, 12, 10)),
        ("encoder", (1, 3, 1, 6, 14)),
    ],
    ids=str,
)
def test_a_coder_traces_as_one_graph(monkeypatch, coder, shape):
    module = getattr(_upstream_vae(monkeypatch).model, coder)
    x = torch.randn(shape)
    with torch.no_grad():
        explanation = torch._dynamo.explain(module)(x.clone())
        expected = module(x.clone())
        compiled = torch.compile(module, backend="eager", fullgraph=True)(x.clone())

    assert (explanation.graph_count, explanation.graph_break_count) == (1, 0)
    torch.testing.assert_close(compiled, expected, rtol=0, atol=0)

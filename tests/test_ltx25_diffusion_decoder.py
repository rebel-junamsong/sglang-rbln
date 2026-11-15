# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import pytest
import torch
from sglang.multimodal_gen.configs.models.decoders.ltx_2_5_diffusion_decoder import (
    LTX25DiffusionDecoderConfig,
)
from sglang.multimodal_gen.runtime.models.decoders import ltx_2_5_diffusion_decoder
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry

from sglang_rbln.compilation import backend as compilation
from sglang_rbln.patches.models import ltx_2


def test_decoder_keeps_upstream_attention():
    """The decoder keeps upstream's flex_attention neighborhood attention; the
    plugin does not replace it."""
    targets = ltx_2.register()
    assert ltx_2.DIFFUSION_DECODER_INIT in targets
    assert not [t for t in targets if t.endswith("._na3d")]
    hooks = [hook for _, hook, _ in HookRegistry._hooks[ltx_2.DIFFUSION_DECODER_INIT]]
    assert ltx_2.keyframe_tag in hooks


def test_decoder_loads_the_checkpoint_keyframe_tag():
    config = LTX25DiffusionDecoderConfig()
    arch = config.arch_config
    arch.decoder_stage_channels = [64, 32, 16, 16, 8]
    arch.decoder_head_dim = 8
    decoder = ltx_2_5_diffusion_decoder.LTX2VideoDiffusionDecoder3d(config)
    ltx_2.keyframe_tag(None, decoder, config)
    state = decoder.state_dict()
    assert torch.equal(state["type_emb"], torch.zeros(arch.latent_channels))
    tag = torch.randn(arch.latent_channels)
    decoder.load_state_dict({**state, "type_emb": tag})
    assert torch.equal(decoder.type_emb.detach(), tag)


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
def test_neighborhood_attention_compiled_by_the_backend_matches_eager(grid, kernel):
    """The decoder's neighborhood attention, compiled through the plugin backend,
    matches eager."""
    lower = pytest.importorskip("rebel.core._flex_attention").lower_flex_attention

    def lowered(graph_module, _):
        lower(graph_module)
        return graph_module.forward

    torch.manual_seed(0)
    attention = ltx_2_5_diffusion_decoder.LTX2VideoVaeNeighborhoodAttention(
        128, kernel
    ).eval()
    hidden_states = torch.randn(1, *grid, 128)
    torch._dynamo.reset()
    with torch.no_grad():
        expected = attention(hidden_states)
        actual = torch.compile(attention, backend=lowered)(hidden_states)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


@pytest.mark.npu
def test_stage_3_attention_on_two_npus_matches_eager(tmp_path):
    """The stage-3 decoder attention at this grid, compiled for two NPUs, runs and
    matches eager."""
    visible = [
        npu for npu in os.environ.get("RBLN_VISIBLE_DEVICES", "").split(",") if npu
    ]
    if len(visible) < 2:
        pytest.skip("needs RBLN_VISIBLE_DEVICES to name two NPUs of its own")
    compilation.ensure_compile_backend()
    torch.manual_seed(0)
    attention = ltx_2_5_diffusion_decoder.LTX2VideoVaeNeighborhoodAttention(
        512, (3, 5, 5)
    )
    attention = attention.to(torch.bfloat16).eval()
    hidden_states = torch.randn(1, 18, 68, 120, 512, dtype=torch.bfloat16)
    options = {"num_devices": 2, "cache_dir": str(tmp_path), "mode": ["strict"]}
    with torch.no_grad():
        reference = attention(hidden_states)
        compiled = torch.compile(attention, backend="rbln", options=options)
        output = compiled(hidden_states)
    error = (output.float() - reference.float()).abs().mean() / reference.abs().mean()
    assert error < 3e-2

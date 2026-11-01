# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

import pytest
import torch
from sglang.multimodal_gen.configs.models.decoders.ltx_2_5_diffusion_decoder import (
    LTX25DiffusionDecoderConfig,
)
from sglang.multimodal_gen.runtime.models.decoders import ltx_2_5_diffusion_decoder
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry

from sglang_rbln.compilation import backend as compilation
from sglang_rbln.layers.attention.neighborhood import neighborhood_attention
from sglang_rbln.patches.models import ltx_2
from sglang_rbln.patches.stages.ltx_2 import decoding_av


def test_hands_the_decoder_the_tiled_neighborhood_attention():
    targets = ltx_2.register()
    assert {ltx_2.DIFFUSION_DECODER_NA3D, ltx_2.DIFFUSION_DECODER_INIT} <= set(targets)
    hooks = [hook for _, hook, _ in HookRegistry._hooks[ltx_2.DIFFUSION_DECODER_NA3D]]
    assert ltx_2.na3d in hooks
    hooks = [hook for _, hook, _ in HookRegistry._hooks[ltx_2.DIFFUSION_DECODER_INIT]]
    assert ltx_2.keyframe_tag in hooks
    assert ltx_2.na3d() is neighborhood_attention


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


@pytest.mark.npu
def test_diffusion_decoder_runs_on_the_npu(caplog, monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compilation.ensure_compile_backend()
    torch.manual_seed(0)
    config = LTX25DiffusionDecoderConfig()
    arch = config.arch_config
    arch.decoder_stage_channels = [128, 64, 32, 32, 16]
    arch.decoder_stage_depths = [2, 1, 1, 1, 2]
    arch.decoder_head_dim = 16
    model = ltx_2_5_diffusion_decoder.LTX2VideoDiffusionDecoderModel(config)
    model = model.to(torch.bfloat16).eval()
    latents = torch.randn(1, 128, 3, 7, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        reference = model(latents, generator=torch.Generator().manual_seed(0))
        monkeypatch.setattr(ltx_2_5_diffusion_decoder, "_na3d", ltx_2.na3d)
        decoding_av.compile_diffusion_decoder(model)
        with caplog.at_level(logging.WARNING, logger="rebel-compiler"):
            for _ in range(2):
                output = model(latents, generator=torch.Generator().manual_seed(0))
    assert "allback to eager" not in caplog.text
    assert output.shape == reference.shape == (1, 3, 17, 224, 256)
    error = (output.float() - reference.float()).abs().mean() / reference.abs().mean()
    assert error < 3e-2

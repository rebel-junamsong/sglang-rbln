# SPDX-License-Identifier: Apache-2.0
import logging
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import LTX2VideoDecoder3d
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln.compilation import backend as compilation
from sglang_rbln.patches.stages.ltx_2 import decoding_av

pytestmark = pytest.mark.usefixtures("rbln_platform")


@pytest.fixture
def stage():
    return SimpleNamespace(
        vae=SimpleNamespace(decoder=nn.Linear(4, 4), use_framewise_decoding=False)
    )


@pytest.fixture
def compiled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs))
    )
    return calls


def decode_video(stage, batch, server_args, on_frames=None):
    return "video", on_frames


def test_registers_around_the_video_decode():
    target = decoding_av.register()
    assert target == decoding_av.HOOK_TARGET
    hooks = [hook for _, hook, _ in HookRegistry._hooks[target]]
    assert decoding_av.compile_video_decoder in hooks


def test_compiles_the_decoder_once(stage, compiled):
    batch = SimpleNamespace(use_diffusion_decoder=False)
    server_args = SimpleNamespace(enable_torch_compile=True)
    outputs = [
        decoding_av.compile_video_decoder(
            decode_video, stage, batch, server_args, on_frames=print
        )
        for _ in range(2)
    ]
    assert outputs == [("video", print)] * 2
    assert [module for module, _ in compiled] == [stage.vae.decoder]
    assert compiled[0][1]["backend"] == "rbln"
    assert compiled[0][1]["dynamic"] is False
    assert stage.vae.use_framewise_decoding


@pytest.mark.parametrize(
    ("enable_torch_compile", "use_diffusion_decoder"),
    [(False, False), (True, True)],
    ids=["compile-disabled", "diffusion-decoder"],
)
def test_leaves_the_decoder_eager(
    stage, compiled, enable_torch_compile, use_diffusion_decoder
):
    batch = SimpleNamespace(use_diffusion_decoder=use_diffusion_decoder)
    server_args = SimpleNamespace(enable_torch_compile=enable_torch_compile)
    decoding_av.compile_video_decoder(decode_video, stage, batch, server_args)
    assert compiled == []
    assert not stage.vae.use_framewise_decoding


@pytest.mark.npu
def test_video_decoder_runs_on_the_npu(caplog, monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_STRICT", "1")
    compilation.ensure_compile_backend()
    torch.manual_seed(0)
    decoder = (
        LTX2VideoDecoder3d(
            block_out_channels=(64, 128, 128, 256),
            spatio_temporal_scaling=(True, True, True, True),
            layers_per_block=(1, 1, 1, 1, 1),
            inject_noise=(False,) * 5,
            upsample_residual=(False,) * 4,
            upsample_factor=(2, 2, 1, 2),
            upsample_type=("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
            spatial_padding_mode="zeros",
        )
        .to(torch.bfloat16)
        .eval()
    )
    latents = torch.randn(1, 128, 2, 2, 3, dtype=torch.bfloat16)
    with torch.no_grad():
        reference = decoder(latents)
        compiled = torch.compile(
            decoder,
            backend=compilation.COMPILE_BACKEND_NAME,
            options=compilation.compile_options(decoder),
            fullgraph=True,
            dynamic=False,
        )
        with caplog.at_level(logging.WARNING, logger="rebel-compiler"):
            output = compiled(latents)
    assert "Fallback to eager" not in caplog.text
    assert output.shape == reference.shape == (1, 3, 9, 64, 96)
    error = (output.float() - reference.float()).abs().mean() / reference.abs().mean()
    assert error < 2e-2

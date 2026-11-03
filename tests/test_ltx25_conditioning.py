# SPDX-License-Identifier: Apache-2.0
import logging
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.configs.models.adapter.ltx_2_duration_head import (
    LTX2DurationHeadConfig,
)
from sglang.multimodal_gen.runtime.models.adapter.ltx_2_duration_head import (
    LTX2DurationHead,
)
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import LTX2VideoEncoder3d
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln.compilation import backend as compilation
from sglang_rbln.compilation import placement
from sglang_rbln.patches.stages.ltx_2 import duration, image_encoding, upsampling

pytestmark = pytest.mark.usefixtures("rbln_platform")

COMPILE = SimpleNamespace(enable_torch_compile=True)


@pytest.fixture
def compiled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs))
    )
    return calls


def forward(stage, batch, server_args):
    return "output"


@pytest.fixture
def encode_stage(monkeypatch):
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_VAE_ENCODER", "1")
    return SimpleNamespace(
        vae=SimpleNamespace(encoder=nn.Linear(4, 4), decoder=nn.Linear(4, 4))
    )


@pytest.fixture
def duration_stage(monkeypatch):
    monkeypatch.setenv("SGLANG_RBLN_DURATION_HEAD_ON_NPU", "1")
    return SimpleNamespace(duration_head=nn.Linear(4, 4))


AUTO = SimpleNamespace(auto_duration=True)


@pytest.fixture
def upsample_stage(monkeypatch):
    monkeypatch.setenv("SGLANG_RBLN_UPSAMPLER_ON_NPU", "1")
    return SimpleNamespace(spatial_upsampler=nn.Conv3d(4, 4, 3))


@pytest.mark.parametrize(
    ("module", "hook"),
    [
        (image_encoding, image_encoding.compile_video_encoder),
        (duration, duration.compile_duration_head),
        (upsampling, upsampling.compile_upsampler),
    ],
    ids=["encoder", "duration-head", "upsampler"],
)
def test_registers_around_the_stage(module, hook):
    target = module.register()
    assert target == module.HOOK_TARGET
    assert hook in [registered for _, registered, _ in HookRegistry._hooks[target]]


def test_compiles_only_the_encoder_once(encode_stage, compiled):
    outputs = [
        image_encoding.compile_video_encoder(forward, encode_stage, None, COMPILE)
        for _ in range(2)
    ]
    assert outputs == ["output", "output"]
    assert [module for module, _ in compiled] == [encode_stage.vae.encoder]
    kwargs = compiled[0][1]
    assert (kwargs["backend"], kwargs["dynamic"]) == ("rbln", False)


def test_encoder_spans_the_devices_of_the_vae(encode_stage, compiled, monkeypatch):
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    placement.place(encode_stage.vae.encoder, (4, 5, 6, 7))
    image_encoding.compile_video_encoder(forward, encode_stage, None, COMPILE)
    options = compiled[0][1]["options"]
    assert (options["num_devices"], options["device"]) == (4, [4, 5, 6, 7])


@pytest.mark.parametrize(
    ("requested", "server_args"),
    [("", COMPILE), ("1", SimpleNamespace(enable_torch_compile=False))],
    ids=["not-requested", "compile-disabled"],
)
def test_leaves_the_encoder_eager(
    encode_stage, compiled, monkeypatch, requested, server_args
):
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_VAE_ENCODER", requested)
    image_encoding.compile_video_encoder(forward, encode_stage, None, server_args)
    assert compiled == []


def test_compiles_the_duration_head_once_on_the_last_device(
    duration_stage, compiled, monkeypatch
):
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    outputs = [
        duration.compile_duration_head(forward, duration_stage, AUTO, COMPILE)
        for _ in range(2)
    ]
    assert outputs == ["output", "output"]
    assert [module for module, _ in compiled] == [duration_stage.duration_head]
    options = compiled[0][1]["options"]
    assert (options["num_devices"], options["device"]) == (1, 3)


@pytest.mark.parametrize(
    ("requested", "batch", "server_args"),
    [
        ("", AUTO, COMPILE),
        ("1", SimpleNamespace(auto_duration=False), COMPILE),
        ("1", AUTO, SimpleNamespace(enable_torch_compile=False)),
    ],
    ids=["not-requested", "fixed-duration", "compile-disabled"],
)
def test_leaves_the_duration_head_eager(
    duration_stage, compiled, monkeypatch, requested, batch, server_args
):
    monkeypatch.setenv("SGLANG_RBLN_DURATION_HEAD_ON_NPU", requested)
    duration.compile_duration_head(forward, duration_stage, batch, server_args)
    assert compiled == []


@pytest.mark.parametrize(
    ("placed", "device"), [(None, 3), ((0, 1, 2, 3), 3)], ids=["tp-last", "placed"]
)
def test_compiles_the_upsampler_once_on_one_device(
    upsample_stage, compiled, monkeypatch, placed, device
):
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    if placed:
        placement.place(upsample_stage.spatial_upsampler, placed)
    outputs = [
        upsampling.compile_upsampler(forward, upsample_stage, None, COMPILE)
        for _ in range(2)
    ]
    assert outputs == ["output", "output"]
    assert [module for module, _ in compiled] == [upsample_stage.spatial_upsampler]
    options = compiled[0][1]["options"]
    assert (options["num_devices"], options["device"]) == (1, device)


@pytest.mark.parametrize(
    ("requested", "server_args"),
    [("", COMPILE), ("1", SimpleNamespace(enable_torch_compile=False))],
    ids=["not-requested", "compile-disabled"],
)
def test_leaves_the_upsampler_eager(
    upsample_stage, compiled, monkeypatch, requested, server_args
):
    monkeypatch.setenv("SGLANG_RBLN_UPSAMPLER_ON_NPU", requested)
    upsampling.compile_upsampler(forward, upsample_stage, None, server_args)
    assert compiled == []


def _run_on_npu(module, inputs, caplog, monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compilation.ensure_compile_backend()
    with torch.no_grad():
        reference = module(*inputs)
        compiled = torch.compile(
            module,
            backend=compilation.COMPILE_BACKEND_NAME,
            options=compilation.compile_options(module),
            fullgraph=True,
            dynamic=False,
        )
        with caplog.at_level(logging.WARNING, logger="rebel-compiler"):
            output = compiled(*inputs)
    assert "Fallback to eager" not in caplog.text
    assert output.shape == reference.shape
    return output.float(), reference.float()


@pytest.mark.npu
def test_video_encoder_runs_on_the_npu(caplog, monkeypatch, tmp_path):
    torch.manual_seed(0)
    encoder = (
        LTX2VideoEncoder3d(
            block_out_channels=(64, 128, 128, 128),
            down_block_types=("LTX2VideoDownBlock3D",) * 4,
            spatio_temporal_scaling=(True, True, True, True),
            layers_per_block=(1, 1, 1, 1, 1),
            downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
            spatial_padding_mode="zeros",
        )
        .to(torch.bfloat16)
        .eval()
    )
    image = torch.randn(1, 3, 1, 64, 96, dtype=torch.bfloat16)
    output, reference = _run_on_npu(encoder, (image,), caplog, monkeypatch, tmp_path)
    error = (output - reference).abs().mean() / reference.abs().mean()
    assert error < 2e-2


@pytest.mark.npu
def test_duration_head_runs_on_the_npu(caplog, monkeypatch, tmp_path):
    torch.manual_seed(0)
    head = LTX2DurationHead(LTX2DurationHeadConfig()).to(torch.bfloat16).eval()
    video = torch.randn(1, 1024, 4096, dtype=torch.bfloat16)
    audio = torch.randn(1, 1024, 2048, dtype=torch.bfloat16)
    output, reference = _run_on_npu(head, (video, audio), caplog, monkeypatch, tmp_path)
    assert torch.allclose(output, reference, rtol=2e-2)

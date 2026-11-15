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
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

from sglang_rbln.compilation import backend as compilation
from sglang_rbln.compilation import placement
from sglang_rbln.compilation.regions import RegionDoesNotFit
from sglang_rbln.patches.stages.ltx_2 import duration, image_encoding

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
def encode_stage():
    return SimpleNamespace(
        vae=SimpleNamespace(encoder=nn.Linear(4, 4), decoder=nn.Linear(4, 4))
    )


@pytest.fixture
def duration_stage():
    return SimpleNamespace(duration_head=nn.Linear(4, 4))


AUTO = SimpleNamespace(auto_duration=True)


@pytest.mark.parametrize(
    ("module", "hook", "kind"),
    [
        (image_encoding, image_encoding.compile_video_encoder, HookType.AROUND),
        (duration, duration.compile_duration_head, HookType.AROUND),
    ],
    ids=["encoder", "duration-head"],
)
def test_registers_on_the_stage(module, hook, kind):
    target = module.register()
    assert target == module.HOOK_TARGET
    assert (kind, hook) in [
        (registered_kind, registered)
        for registered_kind, registered, _ in HookRegistry._hooks[target]
    ]


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


def test_leaves_the_encoder_eager_without_torch_compile(encode_stage, compiled):
    server_args = SimpleNamespace(enable_torch_compile=False)
    image_encoding.compile_video_encoder(forward, encode_stage, None, server_args)
    assert compiled == []


def test_compiles_the_duration_head_once_on_the_dit_set(
    duration_stage, compiled, monkeypatch
):
    """With no spare NPUs for a text encoder the head runs across the DiT's set."""
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    monkeypatch.setattr(placement, "visible_npus", lambda: 4)
    outputs = [
        duration.compile_duration_head(forward, duration_stage, AUTO, COMPILE)
        for _ in range(2)
    ]
    assert outputs == ["output", "output"]
    assert [module for module, _ in compiled] == [duration_stage.duration_head]
    options = compiled[0][1]["options"]
    assert (options["num_devices"], options.get("device")) == (4, None)


@pytest.mark.parametrize(
    ("placed", "devices"),
    [(None, (4, None)), ((4, 5, 6, 7), (4, [4, 5, 6, 7]))],
    ids=["dit-set", "placed"],
)
def test_duration_head_compiles_for_the_set_of_its_inputs(
    duration_stage, compiled, monkeypatch, placed, devices
):
    # the head compiles for the NPU set that holds its inputs
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    monkeypatch.setattr(placement, "text_encoder_devices", lambda: placed)
    duration.compile_duration_head(forward, duration_stage, AUTO, COMPILE)
    options = compiled[0][1]["options"]
    assert (options["num_devices"], options.get("device")) == devices


@pytest.mark.parametrize(
    ("batch", "server_args"),
    [
        (SimpleNamespace(auto_duration=False), COMPILE),
        (AUTO, SimpleNamespace(enable_torch_compile=False)),
    ],
    ids=["fixed-duration", "compile-disabled"],
)
def test_leaves_the_duration_head_eager(duration_stage, compiled, batch, server_args):
    duration.compile_duration_head(forward, duration_stage, batch, server_args)
    assert compiled == []


def test_duration_head_follows_the_text_encoder(duration_stage, compiled, monkeypatch):
    """With no placement of its own the head goes where the text encoder is."""
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    monkeypatch.setattr(placement, "visible_npus", lambda: 8)
    duration.compile_duration_head(forward, duration_stage, AUTO, COMPILE)
    options = compiled[0][1]["options"]
    assert (options["num_devices"], options["device"]) == (4, [4, 5, 6, 7])


def _refusing_once(error):
    """A stage forward that fails the first call the way the fit check does."""
    calls = []

    def stage_forward(stage, batch, server_args):
        calls.append(None)
        if len(calls) == 1:
            raise error
        return "output"

    return stage_forward, calls


@pytest.mark.parametrize(
    ("hook", "stage_fixture", "module_of", "batch", "what"),
    [
        (
            image_encoding.compile_video_encoder,
            "encode_stage",
            lambda stage: stage.vae.encoder,
            None,
            "video encoder",
        ),
        (
            duration.compile_duration_head,
            "duration_stage",
            lambda stage: stage.duration_head,
            AUTO,
            "duration head",
        ),
    ],
    ids=["encoder", "duration-head"],
)
def test_stays_on_the_cpu_when_it_does_not_fit(
    request, caplog, compiled, hook, stage_fixture, module_of, batch, what
):
    stage = request.getfixturevalue(stage_fixture)
    module = module_of(stage)
    module._compiled_call_impl = object()
    stage_forward, calls = _refusing_once(RegionDoesNotFit([9], [4]))

    with caplog.at_level(logging.WARNING):
        output = hook(stage_forward, stage, batch, COMPILE)

    assert output == "output", "the request still has to be served"
    assert len(calls) == 2, "it has to be retried eager, not just logged"
    assert module._compiled_call_impl is None, "the compiled call has to be dropped"
    assert compiled, "it is compiled first; the CPU is the fallback"
    logged = caplog.text
    assert what in logged and "stays on the CPU" in logged
    assert "needs [9] bytes per node, [4] free" in logged, "log needed/free per node"


@pytest.mark.parametrize(
    ("hook", "stage_fixture", "batch"),
    [
        (image_encoding.compile_video_encoder, "encode_stage", None),
        (duration.compile_duration_head, "duration_stage", AUTO),
    ],
    ids=["encoder", "duration-head"],
)
def test_an_unrelated_failure_is_not_swallowed(
    request, compiled, hook, stage_fixture, batch
):
    stage = request.getfixturevalue(stage_fixture)
    stage_forward, calls = _refusing_once(ValueError("something else"))
    with pytest.raises(ValueError, match="something else"):
        hook(stage_forward, stage, batch, COMPILE)
    assert len(calls) == 1, "only a memory failure falls back"


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

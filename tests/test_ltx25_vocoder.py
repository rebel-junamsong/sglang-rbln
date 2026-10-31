# SPDX-License-Identifier: Apache-2.0
import logging
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.configs.models.vocoder.ltx_vocoder import (
    LTXVocoderArchConfig,
    LTXVocoderConfig,
)
from sglang.multimodal_gen.runtime.models.vocoder.ltx_2_vocoder import LTX2Vocoder
from torch import nn

from sglang_rbln.compilation import backend as compilation
from sglang_rbln.patches.stages.ltx_2 import vocoder

pytestmark = pytest.mark.usefixtures("rbln_platform")


@pytest.fixture
def stage():
    return SimpleNamespace(vocoder=nn.Linear(4, 4))


@pytest.fixture
def compiled(monkeypatch):
    monkeypatch.setenv("SGLANG_RBLN_VOCODER_ON_NPU", "1")
    calls = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs))
    )
    return calls


def forward(stage, batch, server_args):
    return "output"


def test_compiles_the_vocoder_once(stage, compiled):
    server_args = SimpleNamespace(enable_torch_compile=True)
    outputs = [
        vocoder.compile_vocoder(forward, stage, None, server_args) for _ in range(2)
    ]
    assert outputs == ["output", "output"]
    assert [module for module, _ in compiled] == [stage.vocoder]
    kwargs = compiled[0][1]
    assert (kwargs["backend"], kwargs["dynamic"]) == ("rbln", False)
    assert kwargs["options"]["compile_context"].state.use_weight_sharing


def test_runs_the_vocoder_on_the_last_tensor_parallel_device(
    stage, compiled, monkeypatch
):
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    server_args = SimpleNamespace(enable_torch_compile=True)
    vocoder.compile_vocoder(forward, stage, None, server_args)
    options = compiled[0][1]["options"]
    assert (options["num_devices"], options["device"]) == (1, 3)


@pytest.mark.parametrize(
    ("on_npu", "enable_torch_compile"),
    [("", True), ("1", False)],
    ids=["not-requested", "compile-disabled"],
)
def test_leaves_the_vocoder_eager(
    stage, compiled, monkeypatch, on_npu, enable_torch_compile
):
    monkeypatch.setenv("SGLANG_RBLN_VOCODER_ON_NPU", on_npu)
    server_args = SimpleNamespace(enable_torch_compile=enable_torch_compile)
    vocoder.compile_vocoder(forward, stage, None, server_args)
    assert compiled == []


def small_bwe_vocoder() -> LTX2Vocoder:
    arch = LTXVocoderArchConfig(
        hidden_channels=64,
        upsample_kernel_sizes=[11, 4],
        upsample_factors=[5, 2],
        resnet_kernel_sizes=[3, 7],
        resnet_dilations=[[1, 3, 5], [1, 3, 5]],
        act_fn="snakebeta",
        antialias=True,
        input_sampling_rate=16000,
        output_sampling_rate=48000,
        filter_length=32,
        window_length=32,
        hop_length=10,
        num_mel_channels=64,
        bwe_act_fn="snakebeta",
        bwe_final_bias=False,
        bwe_hidden_channels=32,
        bwe_upsample_factors=[6, 5],
        bwe_upsample_kernel_sizes=[12, 11],
        bwe_resnet_kernel_sizes=[3],
        bwe_resnet_dilations=[[1, 3, 5]],
    )
    model = LTX2Vocoder(LTXVocoderConfig(arch_config=arch))
    for buffer in model.mel_stft.buffers():
        buffer.copy_(torch.rand_like(buffer))
    return model.eval()


@pytest.mark.npu
def test_vocoder_runs_on_the_npu(caplog, monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compilation.ensure_compile_backend()
    torch.manual_seed(0)
    model = small_bwe_vocoder()
    spectrogram = torch.randn(1, 2, 64, 64)
    with torch.no_grad():
        reference = model(spectrogram)
        compiled = torch.compile(
            model,
            backend=compilation.COMPILE_BACKEND_NAME,
            options=compilation.compile_options(model),
            fullgraph=True,
            dynamic=False,
        )
        with caplog.at_level(logging.WARNING, logger="rebel-compiler"):
            output = compiled(spectrogram)
    assert "Fallback to eager" not in caplog.text
    assert output.shape == reference.shape == (1, 2, 1920)
    snr = 10 * torch.log10(reference.pow(2).sum() / (output - reference).pow(2).sum())
    assert snr > 30

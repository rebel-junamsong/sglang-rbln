# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    ltx_2,
)
from torch import nn

from sglang_rbln import device
from sglang_rbln.compilation import placement, residency
from sglang_rbln.patches.stages.ltx_2 import upsampling

pytestmark = pytest.mark.usefixtures("rbln_platform")

STAGE = ltx_2.upsampling.LTX2UpsampleStage
UPSAMPLE = STAGE._upsample_video_latents
PACK = STAGE.__dict__["_pack_video_latents"].__func__


@pytest.fixture
def on_the_npus(monkeypatch):
    """Take the device path on the host: every tensor counts as resident, and each
    module runs eagerly instead of compiling."""
    built = []

    def resident_module(cache, key, build, tensors, devices=None):
        built.append(key[0])
        return build()

    monkeypatch.setattr(residency, "_is_resident", lambda value: True)
    monkeypatch.setattr(residency, "resident_module", resident_module)
    monkeypatch.setattr(residency, "npus_of", lambda value: (0, 1, 2, 3))
    return built


@pytest.fixture
def stage():
    torch.manual_seed(0)

    @contextlib.contextmanager
    def use_declared_component(component_name, module, target_dtype):
        yield module

    vae = SimpleNamespace(latents_mean=torch.randn(4), latents_std=torch.rand(4) + 0.5)
    return SimpleNamespace(
        vae=vae,
        spatial_upsampler=nn.Conv3d(4, 4, 3, padding=1),
        use_declared_component=use_declared_component,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_upsampling_on_the_npus_matches_upstream_bit_for_bit(on_the_npus, stage, dtype):
    stage.spatial_upsampler.to(dtype)
    latents = torch.randn(1, 4, 3, 6, 10).to(dtype)
    with torch.no_grad():
        expected = UPSAMPLE(stage, latents, None, torch.device("cpu"))
        actual = upsampling.upsample_video_latents(
            UPSAMPLE, stage, latents, None, torch.device("cpu")
        )
    assert on_the_npus == ["upsample"]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_packing_on_the_npus_matches_upstream_bit_for_bit(on_the_npus, dtype):
    server_args = SimpleNamespace(pipeline_config=LTX2PipelineConfig())
    latents = torch.randn(1, 128, 3, 6, 10).to(dtype)
    expected, actual = SimpleNamespace(), SimpleNamespace()
    PACK(expected, latents, server_args)
    upsampling.pack_video_latents(PACK, actual, latents, server_args)
    assert on_the_npus == ["pack"]
    assert torch.equal(actual.latents, expected.latents)
    assert actual.raw_latent_shape == expected.raw_latent_shape


def test_host_latents_take_upstream(stage, monkeypatch):
    calls = []
    monkeypatch.setattr(
        residency, "resident_module", lambda *args, **kwargs: calls.append(args)
    )
    latents = torch.randn(1, 4, 2, 4, 4)
    with torch.no_grad():
        expected = UPSAMPLE(stage, latents, None, torch.device("cpu"))
        actual = upsampling.upsample_video_latents(
            UPSAMPLE, stage, latents, None, torch.device("cpu")
        )
    assert torch.equal(actual, expected)
    assert calls == []


@pytest.mark.parametrize("resident", [True, False], ids=["resident", "host"])
def test_resident_regions_compile_the_upsampler_only_inside_its_graph(
    stage, monkeypatch, resident
):
    monkeypatch.setenv("SGLANG_RBLN_UPSAMPLER_ON_NPU", "1")
    monkeypatch.setattr(device, "resident_regions", lambda: resident)
    monkeypatch.setattr(placement, "text_encoder_devices", lambda: (0, 1, 2, 3))
    compiled = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda module, **kwargs: compiled.append(module)
    )
    upsampling.compile_upsampler(
        stage, None, SimpleNamespace(enable_torch_compile=True)
    )
    assert placement.devices_of(stage.spatial_upsampler) == (0, 1, 2, 3)
    assert compiled == ([] if resident else [stage.spatial_upsampler])

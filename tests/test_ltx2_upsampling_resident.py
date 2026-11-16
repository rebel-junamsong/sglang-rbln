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

from sglang_rbln.compilation import residency
from sglang_rbln.patches.stages.ltx_2 import upsampling

pytestmark = pytest.mark.usefixtures("rbln_platform")

STAGE = ltx_2.upsampling.LTX2UpsampleStage
UPSAMPLE = STAGE._upsample_video_latents
PACK = STAGE.__dict__["_pack_video_latents"].__func__


@pytest.fixture
def built_on_the_npus(monkeypatch):
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
        # as upstream's component manager does: the module takes the use's dtype
        yield module.to(dtype=target_dtype)

    vae = SimpleNamespace(latents_mean=torch.randn(4), latents_std=torch.rand(4) + 0.5)
    return SimpleNamespace(
        vae=vae,
        spatial_upsampler=nn.Conv3d(4, 4, 3, padding=1),
        use_declared_component=use_declared_component,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_upsampling_on_the_npus_matches_upstream_bit_for_bit(
    built_on_the_npus, stage, dtype
):
    stage.spatial_upsampler.to(dtype)
    latents = torch.randn(1, 4, 3, 6, 10).to(dtype)
    with torch.no_grad():
        expected = UPSAMPLE(stage, latents, None, torch.device("cpu"))
        actual = upsampling.upsample_video_latents(
            UPSAMPLE, stage, latents, None, torch.device("cpu")
        )
    assert built_on_the_npus == ["upsample"]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
def test_packing_on_the_npus_matches_upstream_bit_for_bit(built_on_the_npus, dtype):
    server_args = SimpleNamespace(pipeline_config=LTX2PipelineConfig())
    latents = torch.randn(1, 128, 3, 6, 10).to(dtype)
    expected, actual = SimpleNamespace(), SimpleNamespace()
    PACK(expected, latents, server_args)
    upsampling.pack_video_latents(PACK, actual, latents, server_args)
    assert built_on_the_npus == ["pack"]
    assert torch.equal(actual.latents, expected.latents)
    assert actual.raw_latent_shape == expected.raw_latent_shape


def test_host_latents_go_through_upstream_as_is(stage, monkeypatch):
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


def test_fp32_latents_upsample_on_the_device_with_a_bf16_upsampler(
    dummy_device, monkeypatch, tmp_path
):
    """Two-stage LTX: the stage-1 latents come out of the denoise loop in float32
    while the upsampler loads in bfloat16. As upstream, the upsampler runs in the
    latents' dtype; the graph compiled with a bfloat16 upsampler failed to trace."""
    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compile_module = torch.nn.Module.compile

    def compile_resident(module, *args, **kwargs):
        # as the server's residency hook does: the call takes its inputs to the NPUs
        compile_module(module, *args, **kwargs)
        residency.install_resident_call(module)

    monkeypatch.setattr(torch.nn.Module, "compile", compile_resident)
    torch.manual_seed(0)

    @contextlib.contextmanager
    def use_declared_component(component_name, module, target_dtype):
        yield module.to(dtype=target_dtype)

    upsampler = nn.Conv3d(128, 128, 3, padding=1).to(torch.bfloat16)
    stage = SimpleNamespace(
        vae=SimpleNamespace(
            latents_mean=torch.randn(128), latents_std=torch.rand(128) + 0.5
        ),
        spatial_upsampler=upsampler,
        use_declared_component=use_declared_component,
    )
    latents = torch.randn(1, 128, 3, 8, 16).to("rbln:0")
    upsampled = upsampling.upsample_video_latents(
        UPSAMPLE, stage, latents, None, torch.device("cpu")
    )
    assert (upsampled.device.type, upsampled.dtype) == ("rbln", torch.float32)
    assert upsampled.shape == latents.shape
    assert next(upsampler.parameters()).dtype == torch.float32

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.configs.pipeline_configs import ltx_2 as configs
from sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector import (
    LTX2ConnectorTransformer1d,
    LTX2TextConnectors,
)
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.compilation import residency
from sglang_rbln.patches.models import ltx_2_text_features as features

POSTPROCESS = configs._gemma_postprocess_func
RESCALE = LTX2TextConnectors.__dict__["_rescale_v2_features"].__func__
FORWARD = LTX2ConnectorTransformer1d.forward
LENGTH = 16
MASKS = {
    "none": [0] * LENGTH,
    "left-padded": [0] * 5 + [1] * (LENGTH - 5),
    "scattered": [1, 0, 1, 1, 0, 0, 1, 0] * 2,
    "all": [1] * LENGTH,
}


def pipeline_config(v2: bool):
    arch = SimpleNamespace(caption_proj_before_connector=v2)
    return SimpleNamespace(dit_config=SimpleNamespace(arch_config=arch))


@pytest.fixture
def on_the_npus(monkeypatch):
    """Take the device path on the host: every tensor counts as resident, and each
    module runs eagerly instead of compiling."""
    monkeypatch.setattr(residency, "_is_resident", lambda value: True)
    monkeypatch.setattr(
        residency,
        "resident_module",
        lambda cache, key, build, tensors, devices=None: build(),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("v2", [True, False], ids=["v2", "v1"])
@pytest.mark.parametrize("mask", ["left-padded", "all"])
def test_packing_matches_upstream_bit_for_bit(on_the_npus, dtype, v2, mask):
    generator = torch.Generator().manual_seed(0)
    hidden = tuple(
        torch.randn(1, LENGTH, 8, generator=generator).to(dtype) for _ in range(5)
    )
    attention_mask = torch.tensor([MASKS[mask]])
    config = pipeline_config(v2)
    expected = POSTPROCESS(
        SimpleNamespace(hidden_states=hidden),
        {"attention_mask": attention_mask},
        config,
    )
    actual = features.gemma_postprocess(
        POSTPROCESS,
        SimpleNamespace(hidden_states=hidden),
        {"attention_mask": attention_mask},
        config,
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("target", [4096, 2048])
def test_rescale_matches_upstream_bit_for_bit(on_the_npus, dtype, target):
    x = torch.randn(1, LENGTH, 32, generator=torch.Generator().manual_seed(1)).to(dtype)
    expected = RESCALE(x, target, 3840)
    assert torch.equal(features.rescale_v2_features(RESCALE, x, target, 3840), expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mask", sorted(MASKS))
def test_register_step_matches_upstream_bit_for_bit(on_the_npus, dtype, mask):
    torch.manual_seed(0)
    connector = LTX2ConnectorTransformer1d(
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=1,
        num_learnable_registers=8,
    ).to(dtype)
    hidden = torch.randn(1, LENGTH, 32).to(dtype)
    additive = (torch.tensor([MASKS[mask]]) - 1).to(dtype) * torch.finfo(dtype).max
    with torch.no_grad():
        expected = FORWARD(connector, hidden.clone(), additive.clone())
        actual = features.connector_forward(
            FORWARD, connector, hidden.clone(), additive.clone()
        )
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)


def test_host_inputs_take_upstream_and_compile_nothing(monkeypatch):
    def compile_called(*args, **kwargs):
        raise AssertionError("host inputs must not compile a module")

    monkeypatch.setattr(torch.nn.Module, "compile", compile_called)
    hidden = tuple(torch.randn(1, LENGTH, 8) for _ in range(3))
    inputs = {"attention_mask": torch.tensor([MASKS["left-padded"]])}
    config = pipeline_config(True)
    expected = POSTPROCESS(SimpleNamespace(hidden_states=hidden), inputs, config)
    actual = features.gemma_postprocess(
        POSTPROCESS, SimpleNamespace(hidden_states=hidden), inputs, config
    )
    assert torch.equal(actual, expected)
    x = torch.randn(1, LENGTH, 32)
    assert torch.equal(
        features.rescale_v2_features(RESCALE, x, 4096, 3840), RESCALE(x, 4096, 3840)
    )


@pytest.mark.parametrize("target", features.TARGETS)
def test_wraps_upstream_around(target):
    assert target in features.register()
    assert any(
        hook is features.TARGETS[target] and kind is HookType.AROUND
        for kind, hook, _ in HookRegistry._hooks[target]
    )

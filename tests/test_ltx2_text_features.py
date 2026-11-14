# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector import (
    LTX2ConnectorTransformer1d,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators,
)
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.compilation import placement, residency
from sglang_rbln.patches.models import ltx_2_text_features as features

FORWARD = LTX2ConnectorTransformer1d.forward
LENGTH = 16
MASKS = {
    "none": [0] * LENGTH,
    "left-padded": [0] * 5 + [1] * (LENGTH - 5),
    "scattered": [1, 0, 1, 1, 0, 0, 1, 0] * 2,
    "all": [1] * LENGTH,
}


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


def test_the_output_norm_runs_as_a_module_on_the_npus(monkeypatch):
    built = []

    def resident_module(cache, key, build, tensors, devices=None):
        module = build()
        built.append((key[0], type(module).__name__))
        return module

    monkeypatch.setattr(residency, "_is_resident", lambda value: True)
    monkeypatch.setattr(residency, "resident_module", resident_module)
    torch.manual_seed(0)
    connector = LTX2ConnectorTransformer1d(
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=1,
        num_learnable_registers=8,
    )
    additive = torch.zeros(1, LENGTH)
    with torch.no_grad():
        features.connector_forward(
            FORWARD, connector, torch.randn(1, LENGTH, 32), additive
        )
    assert ("norm_out", "Normalize") in built


@pytest.mark.parametrize(
    ("placed", "given", "expected"),
    [
        ((4, 5, 6, 7), None, torch.device("rbln", 1)),
        ((4, 5, 6, 7), "cpu", "cpu"),
        (None, None, None),
    ],
    ids=["placed", "explicit", "unplaced"],
)
def test_text_features_are_returned_on_the_connectors_npus(
    monkeypatch, placed, given, expected
):
    # Upstream returned them on the platform device, the host: 367 MiB down, and up
    # again into each connector.
    monkeypatch.setattr(placement, "_components", {"connectors": placed})
    monkeypatch.setattr(residency, "logical_device_on", lambda devices: 1)
    seen = []

    def upstream(stage, text, server_args, **kwargs):
        seen.append(kwargs.get("device"))

    features.encode_text(upstream, None, "a yak", None, device=given)
    assert seen == [expected]


VALIDATE = StageValidators.__dict__["list_of_tensors_with_min_dims"].__func__


@pytest.mark.parametrize("poisoned", [False, True], ids=["clean", "nan"])
def test_resident_text_features_are_checked_for_nan_on_the_npus(monkeypatch, poisoned):
    # Upstream's isnan downloaded the whole 367 MiB feature to look.
    checked = []

    def resident_module(cache, key, build, tensors, devices=None):
        checked.append(key[0])
        return build()

    monkeypatch.setattr(residency, "_is_resident", lambda value: True)
    monkeypatch.setattr(residency, "npus_of", lambda device: (4, 5, 6, 7))
    monkeypatch.setattr(residency, "resident_module", resident_module)
    features_ = torch.randn(1, LENGTH, 8)
    if poisoned:
        features_[0, 3, 5] = float("nan")
    valid = features.tensors_with_min_dims(VALIDATE, [features_], 2)
    assert valid is not poisoned
    assert valid == VALIDATE([features_], 2)
    assert checked == ["nan"]


@pytest.mark.parametrize("target", features.TARGETS)
def test_wraps_upstream_around(target):
    assert target in features.register()
    assert any(
        hook is features.TARGETS[target] and kind is HookType.AROUND
        for kind, hook, _ in HookRegistry._hooks[target]
    )

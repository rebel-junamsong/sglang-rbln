# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators,
)
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.compilation import placement, residency
from sglang_rbln.patches.models import ltx_2_text_features as features

LENGTH = 16


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

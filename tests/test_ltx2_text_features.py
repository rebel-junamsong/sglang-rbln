# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
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


@pytest.mark.parametrize("target", features.TARGETS)
def test_wraps_upstream_around(target):
    assert target in features.register()
    assert any(
        hook is features.TARGETS[target] and kind is HookType.AROUND
        for kind, hook, _ in HookRegistry._hooks[target]
    )

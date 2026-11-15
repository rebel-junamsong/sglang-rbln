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


def test_host_hidden_states_are_packed_by_upstream_as_is():
    seen = []

    def upstream(hidden_states, attention_mask):
        seen.append(hidden_states.device.type)
        return "packed"

    hidden = torch.zeros(1, LENGTH, 8, 3)
    assert features.pack_text_features(upstream, hidden, torch.ones(1, LENGTH)) == (
        "packed"
    )
    assert seen == ["cpu"]
    assert features._pack_graphs == {}


def test_resident_hidden_states_are_packed_in_one_graph_on_the_device(
    dummy_device, monkeypatch, tmp_path
):
    """On device tensors the packing runs as one compiled graph of upstream's code,
    with nothing taken to the host."""
    import rebel
    from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
        pack_text_embeds_v2,
    )

    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(features, "_pack_graphs", {})
    hidden = torch.randn(1, 128, 64, 49, dtype=torch.bfloat16).to("rbln:0")
    mask = torch.ones(1, 128, dtype=torch.long).to("rbln:0")

    outputs = [
        features.pack_text_features(pack_text_embeds_v2, hidden, mask) for _ in range(2)
    ]
    assert len(features._pack_graphs) == 1
    assert {(out.device.type, tuple(out.shape)) for out in outputs} == {
        ("rbln", (1, 128, 64 * 49))
    }
    (artifact,) = tmp_path.rglob("*.rbln")
    stats = rebel.RBLNCompiledModel.inspect(str(artifact))["subgraph"]["stats"]
    assert [stat["device_func_count"] for stat in stats] == [1]
    host_ops = {
        op
        for stat in stats
        for ops in stat["host_op_reason"].values()
        for op in ops
        if not op.startswith("args_")
    }
    assert host_ops == set()

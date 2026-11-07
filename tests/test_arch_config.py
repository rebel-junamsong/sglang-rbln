# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from sglang.multimodal_gen.configs.models.base import ArchConfig

from sglang_rbln.patches.configs import arch_config


@pytest.fixture
def traceable_config(monkeypatch):
    monkeypatch.setattr(ArchConfig, "__getattr__", arch_config.extra_attr)


@pytest.mark.parametrize(
    ("extras", "expected"), [({}, 4.0), ({"variant": "scaled"}, 6.0)]
)
def test_extra_attrs_trace_without_graph_breaks(traceable_config, extras, expected):
    config = ArchConfig()
    for name, value in extras.items():
        setattr(config, name, value)

    def forward(x):
        if getattr(config, "variant", "plain") == "scaled":
            return x * 2
        return x + 1

    torch._dynamo.reset()
    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    assert compiled(torch.full((2,), 3.0)).tolist() == [expected, expected]


def test_missing_extra_attr_still_raises(traceable_config):
    with pytest.raises(AttributeError, match="no attribute 'variant'"):
        _ = ArchConfig().variant


def test_registers_a_replace_hook():
    assert arch_config.register() == arch_config.HOOK_TARGET

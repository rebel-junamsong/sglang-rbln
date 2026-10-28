# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln.patches.stages import flux3_action

pytestmark = pytest.mark.usefixtures("rbln_platform")


class _Block(nn.Linear):
    pass


@pytest.fixture
def stage():
    transformer = nn.Module()
    transformer.blocks = nn.ModuleList([_Block(4, 4), _Block(4, 4)])
    transformer.head = nn.Linear(4, 4)
    transformer._compile_conditions = [lambda name, module: isinstance(module, _Block)]
    return SimpleNamespace(transformer=transformer)


@pytest.fixture
def compiled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs))
    )
    return calls


def forward(stage, batch, server_args):
    return "denoised"


def test_compiles_before_the_observation_stage():
    targets = flux3_action.register()
    assert targets == (
        flux3_action.HOOK_TARGET,
        flux3_action.OBSERVATION_HOOK_TARGET,
    )
    for target in targets:
        hooks = [hook for _, hook, _ in HookRegistry._hooks[target]]
        assert flux3_action.compile_blocks in hooks


def test_compiles_each_block_once(stage, compiled):
    server_args = SimpleNamespace(enable_torch_compile=True)
    outputs = [
        flux3_action.compile_blocks(forward, stage, None, server_args) for _ in range(2)
    ]
    assert outputs == ["denoised", "denoised"]
    assert [type(module) for module, _ in compiled] == [_Block, _Block]
    assert all(kwargs["backend"] == "rbln" for _, kwargs in compiled)
    assert all(kwargs["dynamic"] is False for _, kwargs in compiled)


def test_respects_disabled_compile(stage, compiled):
    server_args = SimpleNamespace(enable_torch_compile=False)
    flux3_action.compile_blocks(forward, stage, None, server_args)
    assert compiled == []

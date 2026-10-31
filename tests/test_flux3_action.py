# SPDX-License-Identifier: Apache-2.0
import importlib
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

from sglang_rbln.patches.stages import flux3_action, flux3_action_step

pytestmark = pytest.mark.usefixtures("rbln_platform")


class _Block(nn.Linear):
    pass


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(4, 4), _Block(4, 4)])
        self.head = nn.Linear(4, 4)
        self._compile_conditions = [lambda name, module: isinstance(module, _Block)]

    def encode_context(self, x):
        return x


class Flux3ActionTextEncodingStage(SimpleNamespace):
    pass


class Flux3ActionObservationEncodingStage(SimpleNamespace):
    def _encode_streams(self, x):
        return x


class Flux3ActionDenoisingStage(SimpleNamespace):
    def _denoise_step(self, x):
        return x


STAGES = (
    Flux3ActionTextEncodingStage,
    Flux3ActionObservationEncodingStage,
    Flux3ActionDenoisingStage,
)


@pytest.fixture
def transformer():
    return _Transformer()


@pytest.fixture
def compiled(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs))
    )
    return calls


def forward(stage, batch, server_args):
    return "ran"


def test_hooks_every_stage_that_runs_the_dit():
    targets = flux3_action.register()
    assert targets == (
        flux3_action.DENOISE_TARGET,
        flux3_action.OBSERVATION_TARGET,
        flux3_action.TEXT_TARGET,
    )
    for target in targets:
        hooks = [hook for _, hook, _ in HookRegistry._hooks[target]]
        assert flux3_action.compile_dit in hooks


@pytest.mark.parametrize("one_graph", [True, False])
def test_backports_the_tensor_only_step_with_one_graph(monkeypatch, one_graph):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", "" if one_graph else "4")
    stages = importlib.import_module(flux3_action_step.STAGES)
    methods = [
        (getattr(stages, stage), method)
        for stage, backported in flux3_action_step.BACKPORTED.items()
        for method in backported
    ]
    for stage, method in methods:
        monkeypatch.setattr(stage, method.__name__, None, raising=False)
    registered = []
    monkeypatch.setattr(
        HookRegistry,
        "register",
        classmethod(lambda cls, *args: registered.append(args)),
    )
    targets = flux3_action_step.register()
    if not one_graph:
        assert (targets, registered) == ((), [])
        return
    assert registered == [
        (target, forward, HookType.REPLACE)
        for target, forward in flux3_action_step.FORWARDS.items()
    ]
    assert targets == (
        flux3_action_step.OBSERVATION_TARGET,
        flux3_action_step.DENOISE_TARGET,
    )
    assert all(getattr(stage, method.__name__) is method for stage, method in methods)


@pytest.mark.parametrize("stage_type", STAGES)
def test_compiles_each_entry_point_once_as_one_graph(
    monkeypatch, stage_type, transformer, compiled
):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    stage = stage_type(transformer=transformer)
    server_args = SimpleNamespace(enable_torch_compile=True)
    outputs = [
        flux3_action.compile_dit(forward, stage, None, server_args) for _ in range(2)
    ]
    assert outputs == ["ran", "ran"]
    assert len(compiled) == 1
    fn, kwargs = compiled[0]
    assert fn.__name__ in ("encode_context", "_encode_streams", "_denoise_step")
    assert kwargs["fullgraph"] is True
    assert kwargs["dynamic"] is False
    assert kwargs["backend"] == "rbln"


def test_regions_compile_each_block_once(monkeypatch, transformer, compiled):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", "4")
    server_args = SimpleNamespace(enable_torch_compile=True)
    for stage_type in STAGES:
        flux3_action.compile_dit(
            forward, stage_type(transformer=transformer), None, server_args
        )
    assert [type(module) for module, _ in compiled] == [_Block, _Block]
    assert all(kwargs["fullgraph"] is False for _, kwargs in compiled)


@pytest.mark.parametrize("stage_type", STAGES)
def test_respects_disabled_compile(stage_type, transformer, compiled):
    server_args = SimpleNamespace(enable_torch_compile=False)
    flux3_action.compile_dit(
        forward, stage_type(transformer=transformer), None, server_args
    )
    assert compiled == []

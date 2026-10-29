# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.configs.pipeline_configs.flux3_action import (
    Flux3ActionPipelineConfig,
)
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

from sglang_rbln.compilation import backend, shapes
from sglang_rbln.compilation.shapes import ShapeKey
from sglang_rbln.patches.managers import gpu_worker


class LTX2Pipeline:
    pass


class Flux3ActionPipeline:
    pass


@pytest.mark.parametrize(
    ("height", "width", "num_frames"),
    [
        pytest.param(544, 960, 121, id="ltx-default"),
        pytest.param(960, 544, 121, id="portrait"),
        pytest.param(370, 500, 1, id="unaligned-image"),
        pytest.param(240, 416, 9, id="short-unaligned"),
    ],
)
def test_request_pipelines_key_on_the_requested_size(height, width, num_frames):
    request = SimpleNamespace(height=height, width=width, num_frames=num_frames)
    key = shapes.shape_key(LTX2Pipeline(), SimpleNamespace(), request)
    assert key == ShapeKey(
        "LTX2Pipeline",
        (("height", height), ("width", width), ("num_frames", num_frames)),
    )
    assert str(key) == (
        f"LTX2Pipeline(height={height}, width={width}, num_frames={num_frames})"
    )


@pytest.mark.parametrize(
    ("canvas_hw", "horizon"),
    [((544, 736), 32), ((480, 352), 16), ((500, 370), 7)],
    ids=["default", "portrait", "unaligned"],
)
def test_action_pipeline_keys_on_its_config(canvas_hw, horizon):
    config = Flux3ActionPipelineConfig(canvas_hw=canvas_hw, action_horizon=horizon)
    request = SimpleNamespace(height=None, width=None, num_frames=None)
    key = shapes.shape_key(Flux3ActionPipeline(), config, request)
    assert key.dims == (
        ("height", canvas_hw[0]),
        ("width", canvas_hw[1]),
        ("action_horizon", horizon),
    )


@pytest.fixture
def worker() -> SimpleNamespace:
    return SimpleNamespace(
        pipeline=LTX2Pipeline(), server_args=SimpleNamespace(pipeline_config=None)
    )


def test_request_runs_under_its_key(worker):
    request = SimpleNamespace(height=64, width=96, num_frames=5)
    output = object()

    def execute_forward(worker, batch):
        execute_forward.key = shapes.current_shape_key.get()
        return output

    assert gpu_worker.key_request_shape(execute_forward, worker, [request]) is output
    assert execute_forward.key == shapes.shape_key(worker.pipeline, None, request)
    assert shapes.current_shape_key.get() is None


def test_key_is_cleared_when_the_request_fails(worker):
    def execute_forward(worker, batch):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        gpu_worker.key_request_shape(
            execute_forward, worker, [SimpleNamespace(height=1, width=1)]
        )
    assert shapes.current_shape_key.get() is None


def test_registers_the_key_around_execute_forward():
    gpu_worker.register()
    assert any(
        hook is gpu_worker.key_request_shape and kind is HookType.AROUND
        for kind, hook, _ in HookRegistry._hooks[gpu_worker.HOOK_TARGET]
    )


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16)

    def forward(self, hidden):
        return hidden + self.proj(hidden)


@pytest.fixture
def recording(monkeypatch) -> Iterator[SimpleNamespace]:
    monkeypatch.setattr(shapes, "_compiled", set())
    torch._dynamo.reset()
    repeats: list[bool] = []

    def compile_backend(graph, example_inputs):
        repeats.append(shapes.repeats_compile(graph, example_inputs))
        return graph.forward

    with torch._dynamo.config.patch(automatic_dynamic_shapes=False):
        yield SimpleNamespace(backend=compile_backend, repeats=repeats)


def run_under(key, block, hidden):
    token = shapes.current_shape_key.set(key)
    try:
        with torch.no_grad():
            return block(hidden)
    finally:
        shapes.current_shape_key.reset(token)


KEY = ShapeKey("LTX2Pipeline", (("height", 64), ("width", 96), ("num_frames", 5)))
OTHER_KEY = ShapeKey("LTX2Pipeline", (("height", 96), ("width", 64), ("num_frames", 5)))


def test_new_shapes_and_new_blocks_are_not_repeats(recording):
    blocks = [_Block() for _ in range(3)]
    for block in blocks:
        block.compile(backend=recording.backend, dynamic=False)
    for key, tokens in [(KEY, 8), (OTHER_KEY, 12)]:
        hidden = torch.randn(1, tokens, 16)
        for block in blocks:
            hidden = run_under(key, block, hidden)
    assert recording.repeats == [False] * 6


def test_a_stride_recompile_at_the_same_key_is_a_repeat(recording):
    block = _Block()
    hidden = torch.randn(1, 8, 16)
    with torch.no_grad():
        expected = block(hidden)
    block.compile(backend=recording.backend, dynamic=False)
    contiguous = run_under(KEY, block, hidden)
    strided = run_under(KEY, block, hidden.transpose(1, 2).contiguous().transpose(1, 2))
    assert recording.repeats == [False, True]
    assert torch.equal(contiguous, expected)
    assert torch.equal(strided, expected)


def test_backend_warns_on_a_repeat_with_the_key(monkeypatch, caplog):
    monkeypatch.setattr(shapes, "_compiled", set())
    for flag in ("SGLANG_RBLN_RUNTIME_REPORTS", "SGLANG_RBLN_CHECK_GRAPHS"):
        monkeypatch.delenv(flag, raising=False)
    instrumented = backend._instrumented_backend(lambda graph, inputs: graph.forward)
    graph = torch.fx.symbolic_trace(_Block())
    inputs = [torch.randn(1, 8, 16)]
    token = shapes.current_shape_key.set(KEY)
    try:
        with caplog.at_level(logging.WARNING, logger=backend.__name__):
            instrumented(graph, inputs)
            assert not caplog.records
            instrumented(graph, inputs)
    finally:
        shapes.current_shape_key.reset(token)
    (record,) = caplog.records
    assert str(KEY) in record.getMessage()

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sglang.multimodal_gen.configs.pipeline_configs.flux3_action import (
    Flux3ActionPipelineConfig,
)
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.compilation import shapes
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

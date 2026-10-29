# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import itertools
from dataclasses import dataclass
from typing import Any

import torch

REQUEST_DIMS = ("height", "width", "num_frames")


@dataclass(frozen=True)
class ShapeKey:
    pipeline: str
    dims: tuple[tuple[str, int | None], ...]

    def __str__(self) -> str:
        dims = ", ".join(f"{name}={value}" for name, value in self.dims)
        return f"{self.pipeline}({dims})"


current_shape_key: contextvars.ContextVar[ShapeKey | None] = contextvars.ContextVar(
    "sglang_rbln_shape_key", default=None
)


def shape_key(pipeline: Any, pipeline_config: Any, request: Any) -> ShapeKey:
    if canvas_hw := getattr(pipeline_config, "canvas_hw", None):
        height, width = canvas_hw
        dims = (
            ("height", height),
            ("width", width),
            ("action_horizon", pipeline_config.action_horizon),
        )
    else:
        dims = tuple((name, getattr(request, name, None)) for name in REQUEST_DIMS)
    return ShapeKey(type(pipeline).__name__, dims)


def _region(graph: torch.fx.GraphModule, example_inputs) -> tuple[int, ...] | str:
    lifted = (
        value for value in example_inputs if isinstance(value, torch.nn.Parameter)
    )
    weights = tuple(
        map(id, itertools.chain(lifted, graph.parameters(), graph.buffers()))
    )
    return weights or graph.code


def _input_signature(example_inputs) -> tuple:
    return tuple(
        (tuple(value.shape), value.dtype)
        if isinstance(value, torch.Tensor)
        else type(value).__name__
        for value in example_inputs
        if not isinstance(value, torch.nn.Parameter)
    )


_compiled: set[tuple] = set()


def repeats_compile(graph: torch.fx.GraphModule, example_inputs) -> bool:
    entry = (
        current_shape_key.get(),
        _region(graph, example_inputs),
        _input_signature(example_inputs),
    )
    if entry in _compiled:
        return True
    _compiled.add(entry)
    return False

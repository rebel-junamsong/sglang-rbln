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


def shape_key(pipeline: Any, request: Any) -> ShapeKey:
    dims = tuple((name, getattr(request, name, None)) for name in REQUEST_DIMS)
    return ShapeKey(type(pipeline).__name__, dims)


def _graph_identity(
    graph: torch.fx.GraphModule, example_inputs
) -> tuple[int, ...] | str:
    """What tells one compiled module from another: its weights, or without any, its
    code."""
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


# (shape key, graph identity, input signature) of every graph compiled so far
_seen_compiles: set[tuple] = set()


def repeats_compile(graph: torch.fx.GraphModule, example_inputs) -> bool:
    """Whether this graph was compiled before for the same request shape and inputs."""
    entry = (
        current_shape_key.get(),
        _graph_identity(graph, example_inputs),
        _input_signature(example_inputs),
    )
    if entry in _seen_compiles:
        return True
    _seen_compiles.add(entry)
    return False

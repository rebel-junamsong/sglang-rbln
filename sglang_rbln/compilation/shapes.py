# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any

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

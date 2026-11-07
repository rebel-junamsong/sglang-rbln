# SPDX-License-Identifier: Apache-2.0
"""Where a pipeline stage's wall time goes: inside compiled graph runs, or on the host
between them.

Each stage logs its wall time, its graph runs and the time spent in them, and the
host stretches between them, the largest first, each named by the graph that ran
after it. Unlike the profiler this adds no time of its own, so the host stretches
are the ones a request really pays.
"""

from __future__ import annotations

import collections
import contextvars
import dataclasses
import functools
import logging
import time
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)

STAGE_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.PipelineStage.__call__"
)
SHOWN = 5


@dataclasses.dataclass
class StageTimeline:
    started: float
    last: float
    runs: int = 0
    in_graphs: float = 0.0
    # host seconds and count, by the graph that ran after them
    stretches: collections.Counter = dataclasses.field(
        default_factory=collections.Counter
    )
    counts: collections.Counter = dataclasses.field(default_factory=collections.Counter)

    def host(self, label: str, now: float) -> None:
        self.stretches[label] += now - self.last
        self.counts[label] += 1

    def report(self, name: str, ended: float) -> str:
        self.host("stage end", ended)
        wall = ended - self.started
        host = wall - self.in_graphs
        top = ", ".join(
            f"before {label} {seconds:.3f} s x{self.counts[label]}"
            for label, seconds in self.stretches.most_common(SHOWN)
        )
        return (
            f"rbln timeline in {name}: wall {wall:.3f} s, {self.runs} graph runs "
            f"{self.in_graphs:.3f} s, host {host:.3f} s ({top})"
        )


current: contextvars.ContextVar[StageTimeline | None] = contextvars.ContextVar(
    "rbln_stage_timeline", default=None
)


def graph_label(example_inputs) -> str:
    """The graph's largest tensor input, which tells the DiT, the VAE tiles and the
    text encoder apart."""
    shapes = [
        tuple(value.shape)
        for value in example_inputs
        if isinstance(value, torch.Tensor)
    ]
    largest = max(shapes, key=lambda shape: (len(shape), shape), default=())
    return f"graph{list(largest)}x{len(shapes)}"


def timed_callable(compiled: Callable, label: str) -> Callable:
    @functools.wraps(compiled)
    def run(*args, **kwargs):
        stage = current.get()
        if stage is None:
            return compiled(*args, **kwargs)
        started = time.perf_counter()
        stage.host(label, started)
        try:
            return compiled(*args, **kwargs)
        finally:
            stage.last = time.perf_counter()
            stage.runs += 1
            stage.in_graphs += stage.last - started

    return run


def timed_stage(call: Callable, stage: Any, *args, **kwargs):
    started = time.perf_counter()
    timeline = StageTimeline(started, started)
    token = current.set(timeline)
    try:
        return call(stage, *args, **kwargs)
    finally:
        current.reset(token)
        logger.info(timeline.report(type(stage).__name__, time.perf_counter()))


def register() -> str:
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    HookRegistry.register(STAGE_TARGET, timed_stage, HookType.AROUND)
    return STAGE_TARGET

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.compilation import backend, timeline


class Stage:
    """Host work, a graph run, more host work, two runs of another graph."""

    def __init__(self, dit, vae) -> None:
        self.dit, self.vae = dit, vae

    def __call__(self):
        self.dit()
        self.vae()
        self.vae()
        return "done"


@pytest.fixture
def clock(monkeypatch):
    """perf_counter steps one second per reading."""
    ticks = itertools.count()
    monkeypatch.setattr(timeline.time, "perf_counter", lambda: float(next(ticks)))


def test_a_stage_logs_its_host_time_by_the_graph_that_ran_next(clock, caplog):
    dit = timeline.timed_callable(lambda: None, "dit")
    vae = timeline.timed_callable(lambda: None, "vae")
    with caplog.at_level("INFO", logger=timeline.__name__):
        assert timeline.timed_stage(lambda stage: stage(), Stage(dit, vae)) == "done"
    # readings: stage 0 | dit 1-2 | vae 3-4 | vae 5-6 | end 7
    assert (
        "rbln timeline in Stage: wall 7.000 s, 3 graph runs 3.000 s, host 4.000 s "
        "(before vae 2.000 s x2, before dit 1.000 s x1, before stage end 1.000 s x1)"
    ) in caplog.text


def test_a_graph_run_outside_a_stage_is_not_timed(clock):
    seen = []
    run = timeline.timed_callable(lambda value: seen.append(value) or value, "dit")
    assert run(3) == 3
    assert seen == [3]


def test_a_graph_is_named_by_its_largest_input():
    inputs = [torch.zeros(4), torch.zeros(1, 14080, 128), 2.0, torch.zeros(1, 8)]
    assert timeline.graph_label(inputs) == "graph[1, 14080, 128]x3"


def test_the_backend_times_each_graph_run_when_asked(monkeypatch, clock, caplog):
    monkeypatch.setenv("SGLANG_RBLN_LOG_TIMELINE", "1")
    graph = torch.fx.symbolic_trace(lambda x: x + 1)
    compiled = backend._instrumented_backend(lambda graph, inputs: graph)(
        graph, [torch.zeros(2, 3)]
    )
    with caplog.at_level("INFO", logger=timeline.__name__):
        timeline.timed_stage(lambda stage: compiled(torch.zeros(2, 3)), Stage(0, 0))
    assert "1 graph runs 1.000 s" in caplog.text
    assert "before graph[2, 3]x1" in caplog.text


def test_wraps_every_stage_around():
    target = timeline.register()
    assert any(
        hook is timeline.timed_stage and kind is HookType.AROUND
        for kind, hook, _ in HookRegistry._hooks[target]
    )

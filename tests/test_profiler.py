# SPDX-License-Identifier: Apache-2.0
import contextlib
import itertools
from types import SimpleNamespace

import pytest
import rebel.profiler

from sglang_rbln.patches.managers import gpu_worker


@pytest.fixture
def fake_profiler(monkeypatch):
    state = SimpleNamespace(active=False, dirs=[], profiled=[])

    @contextlib.contextmanager
    def profile(output_dir=""):
        state.dirs.append(output_dir)
        state.active = True
        yield
        state.active = False

    monkeypatch.setattr(rebel.profiler, "profile", profile)
    monkeypatch.setattr(gpu_worker, "_request_counter", itertools.count(1))
    return state


def serve_requests(fake_profiler, *warmups):
    def execute_forward(worker, batch):
        if fake_profiler.active:
            fake_profiler.profiled.append(batch[0].request_id)
        return batch[0].request_id

    return [
        gpu_worker.profile_request(
            execute_forward, None, [SimpleNamespace(is_warmup=w, request_id=f"r{i}")]
        )
        for i, w in enumerate(warmups)
    ]


def test_no_profile_dir_profiles_nothing(monkeypatch, fake_profiler):
    monkeypatch.delenv("SGLANG_RBLN_PROFILE_DIR", raising=False)
    assert serve_requests(fake_profiler, False, False, False, False) == [
        "r0",
        "r1",
        "r2",
        "r3",
    ]
    assert fake_profiler.dirs == []


@pytest.mark.parametrize(
    ("request_index", "expected"),
    [("1", "r1"), ("", "r3"), ("3", "r3")],
    ids=["first", "unset-means-third", "third"],
)
def test_profiles_only_the_chosen_request_after_warmup(
    monkeypatch, tmp_path, fake_profiler, request_index, expected
):
    trace_dir = tmp_path / "trace"
    monkeypatch.setenv("SGLANG_RBLN_PROFILE_DIR", str(trace_dir))
    monkeypatch.setenv("SGLANG_RBLN_PROFILE_REQUEST", request_index)
    assert serve_requests(fake_profiler, True, False, False, False, False) == [
        "r0",
        "r1",
        "r2",
        "r3",
        "r4",
    ]
    assert fake_profiler.profiled == [expected]
    assert fake_profiler.dirs == [str(trace_dir)]
    assert trace_dir.is_dir()

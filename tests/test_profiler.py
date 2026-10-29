# SPDX-License-Identifier: Apache-2.0
import contextlib
import itertools
from types import SimpleNamespace

import pytest
import rebel.profiler
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.patches.managers import gpu_worker


@pytest.fixture
def window(monkeypatch):
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


def serve(window, *warmups):
    def execute_forward(worker, batch):
        if window.active:
            window.profiled.append(batch[0].request_id)
        return batch[0].request_id

    return [
        gpu_worker.profile_request(
            execute_forward, None, [SimpleNamespace(is_warmup=w, request_id=f"r{i}")]
        )
        for i, w in enumerate(warmups)
    ]


def test_registers_an_around_hook():
    assert gpu_worker.register() == gpu_worker.HOOK_TARGET
    assert any(
        hook is gpu_worker.profile_request and kind is HookType.AROUND
        for kind, hook, _ in HookRegistry._hooks[gpu_worker.HOOK_TARGET]
    )


def test_does_nothing_without_a_profile_dir(monkeypatch, window):
    monkeypatch.delenv("SGLANG_RBLN_PROFILE_DIR", raising=False)
    assert serve(window, False, False, False, False) == ["r0", "r1", "r2", "r3"]
    assert window.dirs == []


@pytest.mark.parametrize(
    ("request_index", "expected"), [("1", "r1"), ("", "r3"), ("3", "r3")]
)
def test_profiles_only_the_chosen_request_after_warmup(
    monkeypatch, tmp_path, window, request_index, expected
):
    trace_dir = tmp_path / "trace"
    monkeypatch.setenv("SGLANG_RBLN_PROFILE_DIR", str(trace_dir))
    monkeypatch.setenv("SGLANG_RBLN_PROFILE_REQUEST", request_index)
    assert serve(window, True, False, False, False, False) == [
        "r0",
        "r1",
        "r2",
        "r3",
        "r4",
    ]
    assert window.profiled == [expected]
    assert window.dirs == [str(trace_dir)]
    assert trace_dir.is_dir()

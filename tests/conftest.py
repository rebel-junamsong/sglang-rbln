# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable

import pytest
from sglang.multimodal_gen.runtime import platforms

from sglang_rbln import device, envs


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    unmet = {
        "npu": (not device.device_nodes(), "no RBLN device nodes on host"),
        "device_mode": (
            device.execution_mode() is not device.ExecutionMode.DEVICE,
            "execution mode is host",
        ),
    }
    for item in items:
        for marker, (missing, reason) in unmet.items():
            if missing and item.get_closest_marker(marker):
                item.add_marker(pytest.mark.xfail(reason=reason, strict=True))


@pytest.fixture(scope="session")
def npu_name() -> str:
    if forced := envs.RBLN_FORCE_NPU_NAME:
        return forced
    if device.device_nodes():
        return device.npu_name()
    raise pytest.UsageError(
        "no RBLN NPU on this host; set RBLN_FORCE_NPU_NAME to the NPU "
        "the compiler should target (as printed by rbln-smi)"
    )


@pytest.fixture
def fresh_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    for attr in ("_current_platform", "_current_platform_selection"):
        monkeypatch.setattr(platforms, attr, None)
    monkeypatch.setattr(platforms, "_backend_init_done", False)


@pytest.fixture
def rbln_override(monkeypatch: pytest.MonkeyPatch, npu_name: str) -> dict[str, str]:
    env = {
        "RBLN_FORCE_NPU_NAME": npu_name,
        "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "rbln",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return env


@pytest.fixture
def rbln_platform(fresh_platform: None, rbln_override: dict[str, str]):
    platform = platforms.current_platform
    assert platform.device_name == "rbln"
    return platform


@pytest.fixture
def run_probe(rbln_override: dict[str, str]) -> Callable[[str, str], dict]:
    def run(code: str, execution_mode: str) -> dict:
        probe_env = {
            **os.environ,
            **rbln_override,
            "SGLANG_RBLN_EXECUTION_MODE": execution_mode,
            "PYTHONWARNINGS": "ignore",
        }
        completed = subprocess.run(
            [sys.executable, "-c", code],
            env=probe_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        return json.loads(completed.stdout.strip().splitlines()[-1])

    return run

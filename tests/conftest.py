# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable

import pytest
from sglang.multimodal_gen.runtime import platforms

from sglang_rbln import device

FORCED_NPU_NAME = "RBLN-CA25"


@pytest.fixture
def fresh_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    for attr in ("_current_platform", "_current_platform_selection"):
        monkeypatch.setattr(platforms, attr, None)
    monkeypatch.setattr(platforms, "_backend_init_done", False)


@pytest.fixture
def rbln_override(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    env = {
        device.FORCE_NPU_NAME_ENV: FORCED_NPU_NAME,
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
            device.EXECUTION_MODE_ENV: execution_mode,
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

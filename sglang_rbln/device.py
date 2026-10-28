# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import psutil

from sglang_rbln import envs

DEVICE_NODE_DIR = Path("/dev")
DEVICE_NODE_PATTERN = "rbln[0-9]*"


class MemoryInfo(NamedTuple):
    total: int
    used: int
    free: int


class ExecutionMode(StrEnum):
    AUTO = "auto"
    HOST = "host"
    DEVICE = "device"


def device_nodes() -> list[Path]:
    return sorted(DEVICE_NODE_DIR.glob(DEVICE_NODE_PATTERN))


def _distribution_installed(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def rbln_available() -> bool:
    if envs.RBLN_FORCE_NPU_NAME:
        return True
    return _distribution_installed("rebel") and bool(device_nodes())


def torch_rbln_available() -> bool:
    return _distribution_installed("torch_rbln")


def execution_mode() -> ExecutionMode:
    requested = envs.SGLANG_RBLN_EXECUTION_MODE
    try:
        mode = ExecutionMode(requested)
    except ValueError:
        choices = ", ".join(ExecutionMode)
        raise ValueError(
            f"SGLANG_RBLN_EXECUTION_MODE={requested!r}; expected one of {choices}"
        ) from None
    if mode is ExecutionMode.AUTO:
        return ExecutionMode.DEVICE if torch_rbln_available() else ExecutionMode.HOST
    return mode


def npu_name(device_id: int = 0) -> str:
    import rebel

    name = rebel.get_npu_name(device_id) or envs.RBLN_FORCE_NPU_NAME
    if not name:
        raise RuntimeError(
            f"Could not determine the RBLN NPU name for device {device_id}; "
            "set RBLN_FORCE_NPU_NAME to the target NPU on a host without one."
        )
    return str(name)


def npu_memory_info(device_id: int = 0) -> MemoryInfo:
    import rebel

    info = rebel.get_device_memory_info(device_id)
    return MemoryInfo(int(info.total), int(info.used), int(info.free))


def host_memory_info() -> MemoryInfo:
    memory = psutil.virtual_memory()
    return MemoryInfo(int(memory.total), int(memory.used), int(memory.available))

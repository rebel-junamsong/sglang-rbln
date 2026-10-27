# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
from enum import StrEnum
from pathlib import Path

import psutil

FORCE_NPU_NAME_ENV = "RBLN_FORCE_NPU_NAME"
EXECUTION_MODE_ENV = "SGLANG_RBLN_EXECUTION_MODE"
DEVICE_NODE_DIR = Path("/dev")
DEVICE_NODE_PATTERN = "rbln[0-9]*"


class ExecutionMode(StrEnum):
    AUTO = "auto"
    HOST = "host"
    DEVICE = "device"


def forced_npu_name() -> str | None:
    return os.environ.get(FORCE_NPU_NAME_ENV) or None


def device_nodes() -> list[Path]:
    return sorted(DEVICE_NODE_DIR.glob(DEVICE_NODE_PATTERN))


def _distribution_installed(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def rbln_available() -> bool:
    if forced_npu_name():
        return True
    return _distribution_installed("rebel") and bool(device_nodes())


def torch_rbln_available() -> bool:
    return _distribution_installed("torch_rbln")


def execution_mode() -> ExecutionMode:
    requested = os.environ.get(EXECUTION_MODE_ENV, "").strip().lower() or "auto"
    try:
        mode = ExecutionMode(requested)
    except ValueError:
        choices = ", ".join(ExecutionMode)
        raise ValueError(
            f"{EXECUTION_MODE_ENV}={requested!r}; expected one of {choices}"
        ) from None
    if mode is ExecutionMode.AUTO:
        return ExecutionMode.DEVICE if torch_rbln_available() else ExecutionMode.HOST
    return mode


def npu_name(device_id: int = 0) -> str:
    import rebel

    name = rebel.get_npu_name(device_id) or forced_npu_name()
    if not name:
        raise RuntimeError(
            f"Could not determine the RBLN NPU name for device {device_id}; "
            f"set {FORCE_NPU_NAME_ENV} (e.g. RBLN-CA25) on a host without an NPU."
        )
    return str(name)


def npu_memory_info(device_id: int = 0) -> tuple[int, int, int]:
    import rebel

    info = rebel.get_device_memory_info(device_id)
    return int(info.total), int(info.used), int(info.free)


def host_memory_info() -> tuple[int, int, int]:
    memory = psutil.virtual_memory()
    return int(memory.total), int(memory.used), int(memory.available)

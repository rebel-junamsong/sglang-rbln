# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import NamedTuple

from sglang_rbln import envs

DEVICE_NODE_DIR = Path("/dev")
DEVICE_NODE_PATTERN = "rbln[0-9]*"


class MemoryInfo(NamedTuple):
    total: int
    used: int
    free: int


def device_nodes() -> list[Path]:
    return sorted(DEVICE_NODE_DIR.glob(DEVICE_NODE_PATTERN))


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def rbln_available() -> bool:
    """Whether to select the platform: a forced NPU name, or rebel-compiler and an
    NPU device node. Imports nothing, since SGLang asks in every process it spawns
    and importing rebel is slow."""
    if envs.RBLN_FORCE_NPU_NAME:
        return True
    return _module_available("rebel") and bool(device_nodes())


def npu_name(device_id: int = 0) -> str:
    import rebel

    name = rebel.get_npu_name(device_id) or envs.RBLN_FORCE_NPU_NAME
    if not name:
        raise RuntimeError(
            f"Could not determine the RBLN NPU name for device {device_id}; "
            "set RBLN_FORCE_NPU_NAME to the target NPU on a host without one."
        )
    return str(name)


def npu_memory_info(device_id: int) -> MemoryInfo:
    import rebel

    info = rebel.get_device_memory_info(device_id)
    return MemoryInfo(int(info.total), int(info.used), int(info.free))

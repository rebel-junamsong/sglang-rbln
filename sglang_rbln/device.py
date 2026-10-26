# SPDX-License-Identifier: Apache-2.0
"""NPU presence and ``rebel`` device queries."""

from __future__ import annotations

import glob
import importlib.util
import os

FORCE_NPU_NAME_ENV = "RBLN_FORCE_NPU_NAME"
DEVICE_NODE_GLOB = "/dev/rbln[0-9]*"


def forced_npu_name() -> str | None:
    return os.environ.get(FORCE_NPU_NAME_ENV) or None


def device_nodes() -> list[str]:
    return sorted(glob.glob(DEVICE_NODE_GLOB))


def rbln_available() -> bool:
    """Import-free: ``activate()`` runs in every spawned process."""
    if forced_npu_name():
        return True
    try:
        if importlib.util.find_spec("rebel") is None:
            return False
    except (ImportError, ValueError):
        return False
    return bool(device_nodes())


def npu_name(device_id: int = 0) -> str:
    import rebel

    name = rebel.get_npu_name(device_id) or forced_npu_name()
    if not name:
        raise RuntimeError(
            f"Could not determine the RBLN NPU name for device {device_id}; "
            f"set {FORCE_NPU_NAME_ENV} (e.g. RBLN-CA25) on a host without an NPU."
        )
    return str(name)


def memory_info(device_id: int = 0) -> tuple[int, int, int]:
    """``(total, used, free)`` in bytes."""
    import rebel

    info = rebel.get_device_memory_info(device_id)
    return int(info.total), int(info.used), int(info.free)

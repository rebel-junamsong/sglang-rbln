# SPDX-License-Identifier: Apache-2.0
"""NPU presence, execution mode, and ``rebel`` device queries."""

from __future__ import annotations

import glob
import importlib.util
import os

import psutil

FORCE_NPU_NAME_ENV = "RBLN_FORCE_NPU_NAME"
EXECUTION_MODE_ENV = "SGLANG_RBLN_EXECUTION_MODE"
EXECUTION_MODES = ("auto", "host", "device")
DEVICE_NODE_GLOB = "/dev/rbln[0-9]*"


def forced_npu_name() -> str | None:
    return os.environ.get(FORCE_NPU_NAME_ENV) or None


def device_nodes() -> list[str]:
    return sorted(glob.glob(DEVICE_NODE_GLOB))


def _distribution_installed(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def rbln_available() -> bool:
    """Import-free: ``activate()`` runs in every spawned process."""
    if forced_npu_name():
        return True
    return _distribution_installed("rebel") and bool(device_nodes())


def torch_rbln_available() -> bool:
    return _distribution_installed("torch_rbln")


def execution_mode() -> str:
    """``device``: tensors live on the ``rbln`` torch device; ``host``: on CPU."""
    requested = os.environ.get(EXECUTION_MODE_ENV, "auto").strip().lower() or "auto"
    if requested not in EXECUTION_MODES:
        raise ValueError(
            f"{EXECUTION_MODE_ENV}={requested!r}; expected one of {EXECUTION_MODES}"
        )
    if requested == "auto":
        return "device" if torch_rbln_available() else "host"
    return requested


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
    """``(total, used, free)`` NPU memory in bytes."""
    import rebel

    info = rebel.get_device_memory_info(device_id)
    return int(info.total), int(info.used), int(info.free)


def host_memory_info() -> tuple[int, int, int]:
    """``(total, used, free)`` host memory in bytes."""
    memory = psutil.virtual_memory()
    return int(memory.total), int(memory.used), int(memory.available)

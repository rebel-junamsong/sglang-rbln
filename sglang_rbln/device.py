# SPDX-License-Identifier: Apache-2.0
"""Device discovery and memory queries over the ``rebel`` runtime.

:func:`rbln_available` is import-free so ``activate()`` stays cheap; everything
else imports ``rebel`` lazily and only runs inside a worker.
"""

from __future__ import annotations

import glob
import importlib.util
import os

# Set on hosts without an NPU (e.g. a CPU-only compile worker) to name the
# target NPU so the platform still activates and compiles for it.
FORCE_NPU_NAME_ENV = "RBLN_FORCE_NPU_NAME"

# The driver exposes one character device per NPU.
DEVICE_NODE_GLOB = "/dev/rbln[0-9]*"


def forced_npu_name() -> str | None:
    return os.environ.get(FORCE_NPU_NAME_ENV) or None


def device_nodes() -> list[str]:
    return sorted(glob.glob(DEVICE_NODE_GLOB))


def rbln_available() -> bool:
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
            f"Could not determine the RBLN NPU name for device {device_id} "
            f"(rebel.get_npu_name returned None). On a host without an NPU set "
            f"{FORCE_NPU_NAME_ENV} to the target NPU, e.g. RBLN-CA25."
        )
    return str(name)


def memory_info(device_id: int = 0) -> tuple[int, int, int]:
    """``(total, used, free)`` device memory in bytes."""
    import rebel

    info = rebel.get_device_memory_info(device_id)
    return int(info.total), int(info.used), int(info.free)

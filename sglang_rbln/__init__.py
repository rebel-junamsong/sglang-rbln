# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from sglang_rbln import device

logger = logging.getLogger(__name__)

__all__ = ["activate", "PLATFORM_CLASS_PATH"]

PLATFORM_CLASS_PATH = "sglang_rbln.platform:RBLNPlatform"


def activate() -> str | None:
    if not device.rbln_available():
        return None
    logger.info("RBLN out-of-tree platform selected (%s).", PLATFORM_CLASS_PATH)
    return PLATFORM_CLASS_PATH

# SPDX-License-Identifier: Apache-2.0
"""RBLN NPU as an out-of-tree SGLang Diffusion platform."""

from __future__ import annotations

import logging

from sglang_rbln import device

logger = logging.getLogger(__name__)

__all__ = ["activate", "ENTRY_POINT_NAME", "PLATFORM_CLASS_PATH"]

ENTRY_POINT_NAME = "rbln"
PLATFORM_CLASS_PATH = "sglang_rbln.platform:RblnPlatform"


def activate() -> str | None:
    """Platform class path when an RBLN NPU is present, else None."""
    if not device.rbln_available():
        return None
    logger.info("RBLN out-of-tree platform selected (%s).", PLATFORM_CLASS_PATH)
    return PLATFORM_CLASS_PATH

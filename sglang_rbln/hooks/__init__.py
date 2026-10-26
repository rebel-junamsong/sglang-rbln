# SPDX-License-Identifier: Apache-2.0
"""Hook and CustomOp registrations for the ``rbln_hooks`` plugin entry point."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register() -> None:
    logger.debug("sglang_rbln hooks registered.")

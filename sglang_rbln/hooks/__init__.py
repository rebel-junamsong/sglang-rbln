# SPDX-License-Identifier: Apache-2.0
"""Hook and CustomOp registrations for the ``rbln_hooks`` plugin entry point."""

from __future__ import annotations

import logging

from sglang_rbln.hooks import triton_norm

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register() -> None:
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if current_platform.device_name != "rbln":
        return
    bound = triton_norm.seed()
    logger.debug("sglang_rbln hooks registered (triton_norm: %s).", bound)

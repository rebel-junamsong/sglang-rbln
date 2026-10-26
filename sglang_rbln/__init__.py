# SPDX-License-Identifier: Apache-2.0
"""SGLang Diffusion's RBLN backend (torch-rbln) as an out-of-tree platform plugin.

``SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rbln`` selects
:class:`sglang_rbln.platform.RblnPlatform`; with the override unset the loader
calls every installed ``activate()`` and takes the one that answers.
"""

from __future__ import annotations

import logging

from sglang_rbln import device

logger = logging.getLogger(__name__)

__all__ = ["activate", "ENTRY_POINT_NAME", "PLATFORM_CLASS_PATH"]

ENTRY_POINT_NAME = "rbln"
PLATFORM_CLASS_PATH = "sglang_rbln.platform:RblnPlatform"


def activate() -> str | None:
    """Return the platform class path, or None on a host without an RBLN NPU.

    Runs in every spawned process, so it must stay cheap: presence is read from
    the device nodes and the installed distribution, never by importing
    ``rebel`` (seconds) or ``torch_rbln``. Raising here aborts startup, so a
    failed probe answers None.
    """
    if not device.rbln_available():
        return None
    logger.info("RBLN out-of-tree platform selected (%s).", PLATFORM_CLASS_PATH)
    return PLATFORM_CLASS_PATH

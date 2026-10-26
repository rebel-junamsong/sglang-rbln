# SPDX-License-Identifier: Apache-2.0
"""Hook and CustomOp registrations (``sglang.multimodal_gen.plugins`` entry point).

SGLang runs :func:`register` once per process, before the worker builds the
scheduler, and applies the registered hooks afterwards.

Every hook is tracked debt and carries a header: Why / How (hook type and
target) / Upstream PR or issue / Remove when. Prefer a platform seam
(``Platform`` method, ``CustomOp.register_oot_forward``) over a hook, and a
hook over a REPLACE of anything longer than ~30 lines.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register() -> None:
    """Register RBLN hooks and out-of-tree CustomOp forwards.

    Nothing yet: the platform seams cover the skeleton. Entries are added here
    as the Cosmos3 path needs them.
    """
    logger.debug("sglang_rbln hooks registered.")

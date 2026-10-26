# SPDX-License-Identifier: Apache-2.0
"""Hook and CustomOp registrations for the ``rbln_hooks`` plugin entry point."""

from __future__ import annotations

import logging

from sglang_rbln import compilation
from sglang_rbln.hooks import cosmos3_compile, triton_norm, vsa_import

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register() -> None:
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if current_platform.device_name != "rbln":
        return
    vsa_blocked = vsa_import.block_without_cuda()
    bound_exports = triton_norm.bind_missing_exports()
    compile_backend = compilation.ensure_compile_backend()
    cosmos3_hook_target = cosmos3_compile.register()
    logger.debug(
        "sglang_rbln hooks registered (vsa blocked: %s, triton_norm: %s, "
        "compile backend: %s, cosmos3 hook: %s).",
        vsa_blocked,
        bound_exports,
        compile_backend,
        cosmos3_hook_target,
    )

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_rbln.patches.cuda_imports import block_without_cuda

BLOCKED_MODULE = "flashinfer"


def register() -> bool:
    return block_without_cuda(BLOCKED_MODULE)

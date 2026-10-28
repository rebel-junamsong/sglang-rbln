# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from sglang_rbln import device
from sglang_rbln.compilation import backend as compilation
from sglang_rbln.patches.layers import layernorm
from sglang_rbln.patches.layers.attention import video_sparse_attn
from sglang_rbln.patches.models import ltx_2
from sglang_rbln.patches.srt.layers.quantization import fp4_utils
from sglang_rbln.patches.stages import cosmos3
from sglang_rbln.patches.torch_rbln import ops_utils

logger = logging.getLogger(__name__)

__all__ = ["register"]


def register() -> None:
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if current_platform.device_name != "rbln":
        return
    vsa_blocked = video_sparse_attn.register()
    flashinfer_blocked = fp4_utils.register()
    bound_exports = layernorm.bind_missing_exports()
    compile_backend = compilation.ensure_compile_backend()
    cosmos3_target = cosmos3.register()
    ltx_2_target = ltx_2.register()
    promotion_patched = device.torch_rbln_available() and ops_utils.register()
    logger.debug(
        "sglang_rbln patches registered (vsa blocked: %s, flashinfer blocked: %s, "
        "triton_norm: %s, compile backend: %s, cosmos3: %s, ltx_2: %s, "
        "torch_rbln promotion: %s).",
        vsa_blocked,
        flashinfer_blocked,
        bound_exports,
        compile_backend,
        cosmos3_target,
        ltx_2_target,
        promotion_patched,
    )

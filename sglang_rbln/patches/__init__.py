# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_rbln import device
from sglang_rbln.compilation import residency
from sglang_rbln.patches.layers import layernorm
from sglang_rbln.patches.layers.attention import video_sparse_attn
from sglang_rbln.patches.managers import gpu_worker
from sglang_rbln.patches.models import cosmos3video, ltx_2
from sglang_rbln.patches.srt.layers.quantization import fp4_utils
from sglang_rbln.patches.stages import (
    cosmos3,
    flux3_action,
    flux3_action_video,
    text_encoding,
)
from sglang_rbln.patches.torch_rbln import ops_utils

__all__ = ["register"]


def register() -> None:
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if current_platform.device_name != "rbln":
        return
    video_sparse_attn.register()
    fp4_utils.register()
    layernorm.register()
    cosmos3.register()
    cosmos3video.register()
    flux3_action.register()
    flux3_action_video.register()
    ltx_2.register()
    text_encoding.register()
    gpu_worker.register()
    if device.torch_rbln_available():
        ops_utils.register()
    if device.resident_regions():
        residency.register()

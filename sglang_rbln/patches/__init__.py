# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_rbln import device
from sglang_rbln.compilation import regions, residency
from sglang_rbln.patches import ffmpeg_spawn
from sglang_rbln.patches.configs import arch_config
from sglang_rbln.patches.layers import layernorm
from sglang_rbln.patches.layers.attention import video_sparse_attn
from sglang_rbln.patches.managers import gpu_worker
from sglang_rbln.patches.models import (
    cosmos3video,
    flux3,
    flux3_video_vae,
    ltx_2,
    wanvae,
)
from sglang_rbln.patches.srt.layers.quantization import fp4_utils
from sglang_rbln.patches.stages import (
    cosmos3,
    flux3_action,
    flux3_action_step,
    flux3_action_video,
    text_encoding,
)
from sglang_rbln.patches.stages.ltx_2 import decoding_av
from sglang_rbln.patches.stages.ltx_2 import vocoder as ltx_2_vocoder
from sglang_rbln.patches.torch_rbln import ops_utils

__all__ = ["register"]


def register() -> None:
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if current_platform.device_name != "rbln":
        return
    arch_config.register()
    video_sparse_attn.register()
    regions.register()
    fp4_utils.register()
    layernorm.register()
    cosmos3.register()
    cosmos3video.register()
    flux3.register()
    flux3_action_step.register()
    flux3_action.register()
    flux3_action_video.register()
    flux3_video_vae.register()
    ltx_2.register()
    text_encoding.register()
    ltx_2_vocoder.register()
    decoding_av.register()
    gpu_worker.register()
    ffmpeg_spawn.register()
    wanvae.register()
    if device.torch_rbln_available():
        ops_utils.register()
    if device.resident_regions():
        residency.register()

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang_rbln import device, envs
from sglang_rbln.compilation import placement, regions, residency, timeline
from sglang_rbln.patches import ffmpeg_spawn
from sglang_rbln.patches.configs import arch_config
from sglang_rbln.patches.entrypoints import prompt_enhancement
from sglang_rbln.patches.layers import layernorm
from sglang_rbln.patches.layers.attention import video_sparse_attn
from sglang_rbln.patches.managers import gpu_worker
from sglang_rbln.patches.models import (
    ltx_2,
    ltx_2_text_features,
    ltx_2_tile_ranks,
    ltx_2_vae_assembly,
)
from sglang_rbln.patches.srt.layers.quantization import fp4_utils
from sglang_rbln.patches.stages import compile_only, text_encoding
from sglang_rbln.patches.stages.ltx_2 import (
    decoding_av,
    detailing,
    duration,
    image_encoding,
    upsampling,
)
from sglang_rbln.patches.torch_rbln import ops_utils

__all__ = ["register"]


def register() -> None:
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if current_platform.device_name != "rbln":
        return
    arch_config.register()
    regions.install_fit_check()
    placement.register()
    video_sparse_attn.register()
    regions.register()
    fp4_utils.register()
    layernorm.register()
    ltx_2.register()
    ltx_2_tile_ranks.register()
    text_encoding.register()
    decoding_av.register()
    image_encoding.register()
    duration.register()
    upsampling.register()
    detailing.register()
    gpu_worker.register()
    prompt_enhancement.register()
    ffmpeg_spawn.register()
    if device.torch_rbln_available():
        ops_utils.register()
    if envs.SGLANG_RBLN_COMPILE_ONLY:
        compile_only.register()
    if envs.SGLANG_RBLN_LOG_TIMELINE:
        timeline.register()
    if device.resident_regions():
        residency.register()
        ltx_2_vae_assembly.register()
        ltx_2_text_features.register()
        upsampling.register_resident()
        ltx_2.register_resident()

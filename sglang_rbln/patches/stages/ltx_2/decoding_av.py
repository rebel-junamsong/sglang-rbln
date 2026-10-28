# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

logger = logging.getLogger(__name__)

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".ltx_2.decoding_av.LTX2AVDecodingStage._decode_video"
)
COMPILED = "_rbln_decoder_compiled"


def compile_video_decoder(decode_video, stage, batch, server_args, *args, **kwargs):
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    vae = stage.vae
    if (
        server_args.enable_torch_compile
        and not batch.use_diffusion_decoder
        and not getattr(vae, COMPILED, False)
    ):
        compile_kwargs = build_torch_compile_kwargs(mode=None, module=vae.decoder)
        compile_kwargs["dynamic"] = False
        vae.decoder.compile(**compile_kwargs)
        vae.use_framewise_decoding = True
        setattr(vae, COMPILED, True)
        logger.info("Compiling the LTX-2 video decoder: %s", compile_kwargs)
    return decode_video(stage, batch, server_args, *args, **kwargs)


def register() -> str:
    HookRegistry.register(HOOK_TARGET, compile_video_decoder, HookType.AROUND)
    return HOOK_TARGET

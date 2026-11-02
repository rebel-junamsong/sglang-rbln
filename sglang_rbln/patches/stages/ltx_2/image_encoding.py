# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import envs
from sglang_rbln.compilation.backend import static_compile_kwargs

logger = logging.getLogger(__name__)

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding"
    ".LTX2ImageEncodingStage.forward"
)
COMPILED = "_rbln_encoder_compiled"


def compile_video_encoder(forward, stage, batch, server_args):
    encoder = getattr(stage.vae, "encoder", None)
    if (
        envs.SGLANG_RBLN_COMPILE_VAE_ENCODER
        and server_args.enable_torch_compile
        and encoder is not None
        and not getattr(encoder, COMPILED, False)
    ):
        compile_kwargs = static_compile_kwargs(encoder)
        encoder.compile(**compile_kwargs)
        setattr(encoder, COMPILED, True)
        logger.info("Compiling the LTX-2 video encoder: %s", compile_kwargs)
    return forward(stage, batch, server_args)


def register() -> str:
    HookRegistry.register(HOOK_TARGET, compile_video_encoder, HookType.AROUND)
    return HOOK_TARGET

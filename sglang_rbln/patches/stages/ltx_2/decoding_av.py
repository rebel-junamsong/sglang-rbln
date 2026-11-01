# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.compilation import tiling
from sglang_rbln.compilation.backend import (
    single_device_compile_kwargs,
    static_compile_kwargs,
)
from sglang_rbln.patches.stages.ltx_2 import vocoder

logger = logging.getLogger(__name__)

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".ltx_2.decoding_av.LTX2AVDecodingStage.forward"
)
COMPILED = "_rbln_decoder_compiled"
DIFFUSION_DECODER_GRAPHS = (
    "forward_stages_1_to_3",
    "forward_stage_4",
    "forward_diffusion_step",
)


def compile_diffusion_decoder(model) -> None:
    compile_kwargs = single_device_compile_kwargs(model)
    decoder = model.decoder
    for name in DIFFUSION_DECODER_GRAPHS:
        setattr(decoder, name, torch.compile(getattr(decoder, name), **compile_kwargs))
    logger.info("Compiling the LTX-2.5 diffusion decoder: %s", compile_kwargs)


def compile_video_decoder(forward, stage, batch, server_args, *args, **kwargs):
    module = stage.diffusion_decoder if batch.use_diffusion_decoder else stage.vae
    if (
        server_args.enable_torch_compile
        and module is not None
        and not getattr(module, COMPILED, False)
    ):
        if batch.use_diffusion_decoder:
            compile_diffusion_decoder(module)
        else:
            compile_kwargs = static_compile_kwargs(module.decoder)
            module.decoder.compile(**compile_kwargs)
            module.use_framewise_decoding = True
            logger.info("Compiling the LTX-2 video decoder: %s", compile_kwargs)
        setattr(module, COMPILED, True)
    return forward(stage, batch, server_args, *args, **kwargs)


def decode_within_memory(forward, stage, batch, server_args, *args, **kwargs):
    """Retry the decode on smaller tiles when its graph does not fit.

    One tile is one graph, so a decoder cannot be split like a block region;
    the tile size is its knob.
    """
    return tiling.with_smaller_tiles(
        lambda: forward(stage, batch, server_args, *args, **kwargs), stage.vae
    )


def compile_decoders(forward, stage, batch, server_args):
    audio = functools.partial(
        vocoder.compile_audio,
        functools.partial(vocoder.compile_vocoder, forward),
    )
    return compile_video_decoder(
        functools.partial(decode_within_memory, audio), stage, batch, server_args
    )


def register() -> str:
    HookRegistry.register(HOOK_TARGET, compile_decoders, HookType.AROUND)
    return HOOK_TARGET

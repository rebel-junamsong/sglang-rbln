# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.compilation import tiling
from sglang_rbln.compilation.backend import COMPILED, compile_whole, release_graphs
from sglang_rbln.patches.stages.ltx_2 import vocoder

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".ltx_2.decoding_av.LTX2AVDecodingStage.forward"
)
DECODE_TARGET = (
    "sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae.AutoencoderKLLTX2Video.decode"
)


def compile_video_decoder(forward, stage, batch, server_args, *args, **kwargs):
    vae = stage.vae
    if (
        server_args.enable_torch_compile
        and not batch.use_diffusion_decoder
        and vae is not None
        and not getattr(vae.decoder, COMPILED, False)
    ):
        compile_whole(vae.decoder, "LTX-2 video decoder")
    return forward(stage, batch, server_args, *args, **kwargs)


def decode_within_memory(decode, vae, *args, **kwargs):
    """Retry the VAE decode on smaller tiles when a tile's graph does not fit.

    One tile is one graph, so a decoder cannot be split like a block region; the
    tile size is its knob. The decode starts on all of the video's frames at once,
    as upstream's does: a temporal window costs detail in every frame it blends.
    Only when that does not fit does it take the largest window that does, and only
    past the smallest window do the spatial tiles shrink. Only the VAE decode
    retries: another graph of the stage that does not fit has nothing to gain from
    smaller tiles. Before a retry the given-up size's graphs leave the device.
    """
    z = args[0] if args else kwargs["z"]
    frames = (z.shape[2] - 1) * vae.temporal_compression_ratio + 1
    return tiling.with_smaller_tiles(
        lambda: decode(vae, *args, **kwargs),
        vae,
        lambda: release_graphs(vae.decoder),
        frames,
    )


def compile_decoders(forward, stage, batch, server_args):
    audio = functools.partial(vocoder.compile_audio, forward)
    output = compile_video_decoder(audio, stage, batch, server_args)
    # The VAE's tiles and the video they join are a request's largest device tensors
    # (2.2 GB on the VAE's first NPU at 1280x704x121). With the video on the host,
    # their freed memory goes back, so every request's decode finds the free memory
    # the first one did.
    torch.rbln.empty_cache()
    return output


def register() -> tuple[str, ...]:
    HookRegistry.register(HOOK_TARGET, compile_decoders, HookType.AROUND)
    HookRegistry.register(DECODE_TARGET, decode_within_memory, HookType.AROUND)
    return HOOK_TARGET, DECODE_TARGET

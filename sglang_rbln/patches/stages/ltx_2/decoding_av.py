# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import itertools
import logging

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import envs
from sglang_rbln.compilation import placement, regions, tiling
from sglang_rbln.compilation.backend import device_set_context, static_compile_kwargs
from sglang_rbln.patches.models import ltx_2_tile_ranks
from sglang_rbln.patches.stages.ltx_2 import vocoder

logger = logging.getLogger(__name__)

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".ltx_2.decoding_av.LTX2AVDecodingStage.forward"
)
DIFFUSION_DECODE_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".ltx_2.decoding_av.LTX2AVDecodingStage._decode_with_diffusion_decoder"
)
COMPILED = "_rbln_decoder_compiled"
DIFFUSION_DECODER_GRAPHS = ("forward_stage_4", "forward_diffusion_step")
DIFFUSION_DECODER_TILE_TOKENS = 1 << 18


def diffusion_decoder_tile_tokens(model) -> int:
    patch = model.decoder.patch_size
    return (
        model.tile_sample_min_num_frames
        * (model.tile_sample_min_height // patch)
        * (model.tile_sample_min_width // patch)
    )


def fit_diffusion_decoder_tiles(model) -> None:
    axes = itertools.cycle((("height", "width"), ("num_frames",)))
    while diffusion_decoder_tile_tokens(model) > DIFFUSION_DECODER_TILE_TOKENS:
        for axis in next(axes):
            for kind in ("min", "stride"):
                name = f"tile_sample_{kind}_{axis}"
                setattr(model, name, getattr(model, name) // 2)


def decoder_devices(model) -> tuple[int, ...]:
    devices = placement.devices_of(model)
    if devices:
        return tuple(devices)
    return tuple(range(envs.SGLANG_RBLN_NUM_DEVICES or 1))


def on_device(compile_kwargs: dict, device: int) -> dict:
    """The compile on one device of the set.

    With the compile cache, every rank stays in the set's weight-sharing context: each
    tile graph compiles once and its weight pool is brought up on each rank's device.
    Without it rebel brings a pool up only where the graph compiled, so each device
    keeps a context of its own.
    """
    options = compile_kwargs["options"] | {"num_devices": 1, "device": device}
    if "cache_dir" not in options:
        options["compile_context"] = device_set_context((device,))
    return compile_kwargs | {"options": options}


def compile_diffusion_decoder(model) -> None:
    """Compile the decoder on its component's devices.

    Stages 1-3 compile on the set's first NPU, one region per stage
    ModuleList. Split over the set, their attention aborts on the device
    (rebel_requests/dfr_stage3_rsd_abort.md). The stage-4/5 tiles follow
    upstream's decode-parallel tiling: each other NPU of the set is a rank
    (ltx_2_tile_ranks) whose tile graphs run on that NPU alone, with the
    decoder weights resident there. A set of two keeps both NPUs as ranks,
    since one rank would serialise the tiles. The set's one context compiles
    each tile graph once for all ranks.
    """
    compile_kwargs = static_compile_kwargs(model)
    devices = decoder_devices(model)
    stages_device, tile_devices = split_decoder_devices(devices)
    stage_kwargs = on_device(compile_kwargs, stages_device)
    decoder = model.decoder
    fit_diffusion_decoder_tiles(model)
    decoder.conv_in.compile(**stage_kwargs)
    for blocks, upsample in zip(
        decoder.det_stages[:-1], decoder.upsamples[:-1], strict=True
    ):
        regions.group_blocks(list(blocks), None, stage_kwargs)
        upsample.compile(**stage_kwargs)
    for name in DIFFUSION_DECODER_GRAPHS:
        phase = getattr(decoder, name)
        setattr(
            decoder,
            name,
            ltx_2_tile_ranks.RankDispatch(
                [
                    torch.compile(phase, **on_device(compile_kwargs, device))
                    for device in tile_devices
                ]
            ),
        )
    setattr(model, ltx_2_tile_ranks.TILE_DEVICES, tile_devices)
    logger.info(
        "Compiling the LTX-2.5 diffusion decoder: stages 1-3 on %s, tiles on %s",
        stages_device,
        tile_devices,
    )


def split_decoder_devices(devices: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    """The NPU for stages 1-3 and the NPUs that act as tile ranks."""
    stages_device = devices[0]
    tile_devices = devices[1:] if len(devices) > 2 else devices
    return stages_device, tile_devices


def compile_video_decoder(forward, stage, batch, server_args, *args, **kwargs):
    module = stage.diffusion_decoder if batch.use_diffusion_decoder else stage.vae
    opted_in = (
        envs.SGLANG_RBLN_COMPILE_DIFFUSION_DECODER
        if batch.use_diffusion_decoder
        else True
    )
    if (
        server_args.enable_torch_compile
        and opted_in
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


def decode_in_decoder_dtype(decode, stage, decoder, latents, batch, server_args):
    """Hand a compiled diffusion decoder its latents in its own dtype.

    Upstream casts them only with autocast off and otherwise leaves the cast to
    autocast, which does not reach the ops of a compiled graph.
    """
    if getattr(decoder, COMPILED, False):
        latents = latents.to(next(decoder.parameters()).dtype)
    return decode(stage, decoder, latents, batch, server_args)


def register() -> tuple[str, ...]:
    HookRegistry.register(HOOK_TARGET, compile_decoders, HookType.AROUND)
    HookRegistry.register(
        DIFFUSION_DECODE_TARGET, decode_in_decoder_dtype, HookType.AROUND
    )
    return HOOK_TARGET, DIFFUSION_DECODE_TARGET

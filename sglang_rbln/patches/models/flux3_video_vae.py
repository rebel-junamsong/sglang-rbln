# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from sglang_rbln import envs
from sglang_rbln.layers.attention.neighborhood import neighborhood_attention

logger = logging.getLogger(__name__)

VAE = "sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae"
ATTENTION_TARGET = f"{VAE}.neighborhood_attention"
LOAD_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines.flux3_action.Flux3ActionPipeline._load_vae"
)


def regions(module: nn.Module) -> Iterator[nn.Module]:
    for child in module.children():
        if isinstance(child, nn.Sequential):
            yield from regions(child)
        else:
            yield child


def decoder_region_inputs(decoder, latents):
    with torch.no_grad(), FakeTensorMode(allow_non_fake_inputs=True) as mode:
        x = mode.from_tensor(latents).permute(0, 2, 3, 4, 1).contiguous()
        inputs = []
        for region in [decoder.proj_in, *regions(decoder.features), decoder.proj_out]:
            inputs.append((region, x.shape, x.dtype))
            x = region.forward(x)
    return inputs


def compile_largest_first(decoder, args):
    (latents,) = args
    if latents.shape in decoder.compiled_shapes:
        return
    with torch.no_grad():
        for region, shape, dtype in reversed(decoder_region_inputs(decoder, latents)):
            region(torch.zeros(shape, dtype=dtype))
    decoder.compiled_shapes.add(latents.shape)


def compile_regions(vae, server_args, config):
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    if not server_args.enable_torch_compile:
        return vae
    if vae.model.decoder is not None:
        vae.model.decoder.to(torch.bfloat16)
    compile_kwargs = build_torch_compile_kwargs(mode=None, module=vae)
    compile_kwargs["dynamic"] = False
    num_devices = compile_kwargs["options"].pop("num_devices", None) or 1
    placements = [
        (region, 0) for region in regions(vae.model.encoder or nn.Module())
    ] + [
        (region, index % num_devices)
        for index, region in enumerate(regions(vae.model.decoder or nn.Module()))
    ]
    for region, device in placements:
        options = compile_kwargs["options"]
        if num_devices > 1:
            options = options | {"num_devices": 1, "device": device}
        region.compile(**(compile_kwargs | {"options": options}))
    if vae.model.decoder is not None:
        vae.model.decoder.compiled_shapes = set()
        vae.model.decoder.register_forward_pre_hook(compile_largest_first)
    logger.info("Compiling %d FLUX 3 video VAE regions", len(placements))
    return vae


def register() -> tuple[str, ...]:
    if not envs.SGLANG_RBLN_COMPILE_VAE:
        return ()
    HookRegistry.register(ATTENTION_TARGET, neighborhood_attention, HookType.REPLACE)
    HookRegistry.register(LOAD_TARGET, compile_regions, HookType.AFTER)
    return ATTENTION_TARGET, LOAD_TARGET

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Iterator

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

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


def float32_rotary_tables(vae: nn.Module) -> None:
    from sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae import (
        RotaryPositionEmbedding3D,
    )

    for module in vae.modules():
        if isinstance(module, RotaryPositionEmbedding3D):
            module.inv_freq = module.inv_freq.float()


def compile_regions(vae, server_args, config):
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    if not server_args.enable_torch_compile:
        return vae
    coders = [c for c in (vae.model.encoder, vae.model.decoder) if c is not None]
    if vae.model.decoder is not None:
        vae.model.decoder.to(torch.bfloat16)
    float32_rotary_tables(vae)
    compile_kwargs = build_torch_compile_kwargs(mode=None, module=vae)
    compile_kwargs["dynamic"] = False
    if num_devices := compile_kwargs["options"].get("num_devices"):
        compile_kwargs["options"] |= {"num_devices": 1, "device": num_devices - 1}
    compiled = [region for coder in coders for region in regions(coder)]
    for region in compiled:
        region.compile(**compile_kwargs)
    logger.info("Compiling %d FLUX 3 video VAE regions", len(compiled))
    return vae


def register() -> tuple[str, ...]:
    if not envs.SGLANG_RBLN_COMPILE_VAE:
        return ()
    HookRegistry.register(ATTENTION_TARGET, neighborhood_attention, HookType.REPLACE)
    HookRegistry.register(LOAD_TARGET, compile_regions, HookType.AFTER)
    return ATTENTION_TARGET, LOAD_TARGET

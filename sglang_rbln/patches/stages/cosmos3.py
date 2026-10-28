# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

from sglang_rbln import envs
from sglang_rbln.patches.layers import layernorm
from sglang_rbln.patches.models import cosmos3video

logger = logging.getLogger(__name__)

STAGES = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
HOOK_TARGET = f"{STAGES}.cosmos3.Cosmos3DenoisingStage._maybe_enable_torch_compile"
DECODE_HOOK_TARGET = f"{STAGES}.cosmos3.Cosmos3DecodingStage._decode_latents"


HOST_MODULES = ("language_model", "norm_moe_gen")


def compile_layers(stage, transformer: nn.Module, server_args) -> None:
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    if not isinstance(transformer, nn.Module):
        return
    for name in HOST_MODULES:
        cosmos3video.run_on_host(transformer.get_submodule(name))
    if not server_args.enable_torch_compile:
        return
    layernorm.prebuild_qk_norm_rope(transformer.head_dim)
    compile_kwargs = build_torch_compile_kwargs(mode=None, module=transformer)
    compile_kwargs["dynamic"] = False
    layers = transformer.gen_layers
    logger.info("Compiling %d Cosmos3 gen_layers: %s", len(layers), compile_kwargs)
    for layer in layers:
        layer.compile(**compile_kwargs)
    transformer._gen_layers_torch_compiled = True


def vae_regions(module: nn.Module) -> list[nn.Module]:
    from sglang.multimodal_gen.runtime.models.vaes import wanvae

    region_types = (
        wanvae.WanResidualBlock,
        wanvae.WanResample,
        wanvae.WanAttentionBlock,
        wanvae.WanCausalConv3d,
        wanvae.WanRMS_norm,
        wanvae.DupUp3D,
    )
    regions = []
    for child in module.children():
        if isinstance(child, region_types):
            regions.append(child)
        else:
            regions.extend(vae_regions(child))
    return regions


def compile_vae_regions(vae: nn.Module) -> None:
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    if getattr(vae, "_rbln_regions_compiled", False):
        return
    compile_kwargs = build_torch_compile_kwargs(mode=None, module=vae)
    compile_kwargs["dynamic"] = False
    regions = vae_regions(vae)
    logger.info("Compiling %d VAE regions: %s", len(regions), compile_kwargs)
    for region in regions:
        region.compile(**compile_kwargs)
    vae._rbln_regions_compiled = True


def place_vae_decode(decode_latents, stage, latents: torch.Tensor, *args):
    if envs.SGLANG_RBLN_COMPILE_VAE:
        compile_vae_regions(stage.vae)
        return decode_latents(stage, latents, *args)
    if latents.device.type != "rbln":
        return decode_latents(stage, latents, *args)
    if next(stage.vae.parameters()).device.type == "rbln":
        stage.vae.to("cpu")
        stage._latents_mean = stage._latents_std = None
    return decode_latents(stage, latents.cpu(), *args).to(latents.device)


def register() -> tuple[str, str]:
    HookRegistry.register(HOOK_TARGET, compile_layers, HookType.REPLACE)
    HookRegistry.register(DECODE_HOOK_TARGET, place_vae_decode, HookType.AROUND)
    return HOOK_TARGET, DECODE_HOOK_TARGET

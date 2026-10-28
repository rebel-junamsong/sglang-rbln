# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

logger = logging.getLogger(__name__)

STAGES = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
HOOK_TARGET = f"{STAGES}.cosmos3.Cosmos3DenoisingStage._maybe_enable_torch_compile"
DECODE_HOOK_TARGET = f"{STAGES}.cosmos3.Cosmos3DecodingStage._decode_latents"
GEN_LAYER_CACHE_SIZE_LIMIT = 128


def compile_gen_layers(stage, transformer: nn.Module, server_args) -> None:
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    if not server_args.enable_torch_compile or not isinstance(transformer, nn.Module):
        return
    gen_layers = getattr(transformer, "gen_layers", None)
    if not isinstance(gen_layers, nn.ModuleList):
        logger.warning("Cosmos3 gen_layers not found, skipping torch.compile")
        return
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, GEN_LAYER_CACHE_SIZE_LIMIT
    )
    compile_kwargs = build_torch_compile_kwargs(mode=None, module=transformer)
    compile_kwargs["dynamic"] = False
    logger.info(
        "Compiling %d Cosmos3 gen_layers with %s", len(gen_layers), compile_kwargs
    )
    transformer._gen_layers_torch_compiled = True
    for layer in gen_layers:
        layer.compile(**compile_kwargs)


def decode_latents_on_host(decode_latents, stage, latents: torch.Tensor, *args):
    if latents.device.type != "rbln":
        return decode_latents(stage, latents, *args)
    if next(stage.vae.parameters()).device.type == "rbln":
        stage.vae.to("cpu")
        stage._latents_mean = stage._latents_std = None
    return decode_latents(stage, latents.cpu(), *args).to(latents.device)


def register() -> tuple[str, str]:
    HookRegistry.register(HOOK_TARGET, compile_gen_layers, HookType.REPLACE)
    HookRegistry.register(DECODE_HOOK_TARGET, decode_latents_on_host, HookType.AROUND)
    return HOOK_TARGET, DECODE_HOOK_TARGET

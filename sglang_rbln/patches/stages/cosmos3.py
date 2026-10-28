# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

logger = logging.getLogger(__name__)

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".cosmos3.Cosmos3DenoisingStage._maybe_enable_torch_compile"
)
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


def register() -> str:
    HookRegistry.register(HOOK_TARGET, compile_gen_layers, HookType.REPLACE)
    return HOOK_TARGET

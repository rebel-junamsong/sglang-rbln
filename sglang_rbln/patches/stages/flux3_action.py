# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import envs

logger = logging.getLogger(__name__)

STAGES = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".flux3_action"
)
HOOK_TARGET = f"{STAGES}.Flux3ActionDenoisingStage.forward"
OBSERVATION_HOOK_TARGET = f"{STAGES}.Flux3ActionObservationEncodingStage.forward"
TEXT_HOOK_TARGET = f"{STAGES}.Flux3ActionTextEncodingStage.forward"
COMPILED = "_rbln_blocks_compiled"
GRAPH_METHODS = {
    "Flux3ActionTextEncodingStage": ("transformer", "encode_context"),
    "Flux3ActionObservationEncodingStage": (None, "_encode_streams"),
    "Flux3ActionDenoisingStage": (None, "_denoise_step"),
}


def compile_kwargs_for(transformer):
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    compile_kwargs = build_torch_compile_kwargs(mode=None, module=transformer)
    compile_kwargs["dynamic"] = False
    return compile_kwargs


def compile_graph(stage):
    owner_attr, method = GRAPH_METHODS[type(stage).__name__]
    owner = getattr(stage, owner_attr) if owner_attr else stage
    if method in vars(owner):
        return
    compile_kwargs = compile_kwargs_for(stage.transformer) | {"fullgraph": True}
    setattr(owner, method, torch.compile(getattr(owner, method), **compile_kwargs))
    logger.info("Compiling FLUX 3 %s.%s as one graph", type(stage).__name__, method)


def compile_blocks(stage):
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        compile_matching_submodules,
    )

    transformer = stage.transformer
    if getattr(transformer, COMPILED, False):
        return
    compile_kwargs = compile_kwargs_for(transformer)
    count = compile_matching_submodules(transformer, compile_kwargs=compile_kwargs)
    setattr(transformer, COMPILED, True)
    logger.info("Compiling %d FLUX 3 blocks: %s", count, compile_kwargs)


def compile_dit(forward, stage, batch, server_args):
    if server_args.enable_torch_compile:
        if envs.SGLANG_RBLN_FLUX3_ONE_GRAPH:
            compile_graph(stage)
        else:
            compile_blocks(stage)
    return forward(stage, batch, server_args)


def register() -> tuple[str, ...]:
    targets = (HOOK_TARGET, OBSERVATION_HOOK_TARGET, TEXT_HOOK_TARGET)
    for target in targets:
        HookRegistry.register(target, compile_dit, HookType.AROUND)
    return targets

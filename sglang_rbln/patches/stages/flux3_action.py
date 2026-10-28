# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

logger = logging.getLogger(__name__)

STAGES = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".flux3_action"
)
HOOK_TARGET = f"{STAGES}.Flux3ActionDenoisingStage.forward"
OBSERVATION_HOOK_TARGET = f"{STAGES}.Flux3ActionObservationEncodingStage.forward"
COMPILED = "_rbln_blocks_compiled"


def compile_blocks(forward, stage, batch, server_args):
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
        compile_matching_submodules,
    )

    transformer = stage.transformer
    if server_args.enable_torch_compile and not getattr(transformer, COMPILED, False):
        compile_kwargs = build_torch_compile_kwargs(mode=None, module=transformer)
        compile_kwargs["dynamic"] = False
        count = compile_matching_submodules(transformer, compile_kwargs=compile_kwargs)
        setattr(transformer, COMPILED, True)
        logger.info("Compiling %d FLUX 3 blocks: %s", count, compile_kwargs)
    return forward(stage, batch, server_args)


def register() -> tuple[str, str]:
    for target in (HOOK_TARGET, OBSERVATION_HOOK_TARGET):
        HookRegistry.register(target, compile_blocks, HookType.AROUND)
    return HOOK_TARGET, OBSERVATION_HOOK_TARGET

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import envs
from sglang_rbln.patches.stages.ltx_2.vocoder import COMPILED, compile_on_one_device

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".ltx_2.duration.LTX2DurationStage.forward"
)


def compile_duration_head(forward, stage, batch, server_args):
    head = stage.duration_head
    if (
        envs.SGLANG_RBLN_DURATION_HEAD_ON_NPU
        and server_args.enable_torch_compile
        and batch.auto_duration
        and head is not None
        and not getattr(head, COMPILED, False)
    ):
        compile_on_one_device(head, "duration head")
    return forward(stage, batch, server_args)


def register() -> str:
    HookRegistry.register(HOOK_TARGET, compile_duration_head, HookType.AROUND)
    return HOOK_TARGET

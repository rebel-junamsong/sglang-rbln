# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.compilation import placement
from sglang_rbln.compilation.backend import COMPILED, compile_whole
from sglang_rbln.compilation.regions import run_on_cpu_if_it_does_not_fit

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".ltx_2.duration.LTX2DurationStage.forward"
)


def compile_duration_head(forward, stage, batch, server_args):
    head = stage.duration_head
    if not (
        server_args.enable_torch_compile and batch.auto_duration and head is not None
    ):
        return forward(stage, batch, server_args)
    if not getattr(head, COMPILED, False):
        # The head reads the text encoder's output, so unless it was placed itself
        # it goes on the text encoder's set; with neither, on the DiT's NPUs.
        if placement.devices_of(head) is None and (
            devices := placement.text_encoder_devices()
        ):
            placement.place(head, devices)
        compile_whole(head, "duration head")
    return run_on_cpu_if_it_does_not_fit(
        lambda: forward(stage, batch, server_args), head, "duration head"
    )


def register() -> str:
    HookRegistry.register(HOOK_TARGET, compile_duration_head, HookType.AROUND)
    return HOOK_TARGET

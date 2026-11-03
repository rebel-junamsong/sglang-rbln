# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import envs
from sglang_rbln.compilation.backend import COMPILED, compile_on_one_device

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".ltx_2.upsampling.LTX2UpsampleStage.forward"
)


def compile_upsampler(stage, batch, server_args, *args, **kwargs):
    """Compile the upsampler the first time the stage runs, and change nothing else.

    Compiling is a setup step, not something that wraps the call, so it registers
    BEFORE rather than AROUND: only one hook can wrap this stage's call, and that one
    belongs to the detailing LoRA, which needs the stage-1 latent from around it
    (see ltx_2/detailing.py). Returning None leaves the arguments as they are.
    """
    upsampler = stage.spatial_upsampler
    if (
        envs.SGLANG_RBLN_UPSAMPLER_ON_NPU
        and server_args.enable_torch_compile
        and upsampler is not None
        and not getattr(upsampler, COMPILED, False)
    ):
        compile_on_one_device(upsampler, "latent upsampler")
    return None


def register() -> str:
    HookRegistry.register(HOOK_TARGET, compile_upsampler, HookType.BEFORE)
    return HOOK_TARGET

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
    """BEFORE, not AROUND: the stage's one AROUND hook belongs to detailing."""
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

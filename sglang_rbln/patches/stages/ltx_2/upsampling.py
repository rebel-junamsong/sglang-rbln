# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import envs
from sglang_rbln.compilation import placement
from sglang_rbln.compilation.backend import (
    COMPILED,
    compile_once,
    static_compile_kwargs,
)

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".ltx_2.upsampling.LTX2UpsampleStage.forward"
)


def compile_upsampler(stage, batch, server_args, *args, **kwargs):
    """Compile the upsampler on the text encoder's NPUs, unless it was placed itself.

    The DiT's set has no room for it beside both DiT shapes and the VAE decoder; in
    device mode its latents cross between the two sets on the device.
    BEFORE, not AROUND: the stage's one AROUND hook belongs to detailing.
    """
    upsampler = stage.spatial_upsampler
    if (
        envs.SGLANG_RBLN_UPSAMPLER_ON_NPU
        and server_args.enable_torch_compile
        and upsampler is not None
        and not getattr(upsampler, COMPILED, False)
    ):
        if placement.devices_of(upsampler) is None and (
            devices := placement.text_encoder_devices()
        ):
            placement.place(upsampler, devices)
        compile_once(upsampler, "latent upsampler", static_compile_kwargs(upsampler))
    return None


def register() -> str:
    HookRegistry.register(HOOK_TARGET, compile_upsampler, HookType.BEFORE)
    return HOOK_TARGET

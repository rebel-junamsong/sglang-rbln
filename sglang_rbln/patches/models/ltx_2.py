# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

MODELS = "sglang.multimodal_gen.runtime.models"
HOOK_TARGETS = (
    f"{MODELS}.dits.ltx_2.LTX2AudioVideoRotaryPosEmbed.forward",
    f"{MODELS}.adapter.ltx_2_connector.LTX2RotaryPosEmbed1d.forward",
)
FUSED_ADA_VALUES = f"{MODELS}.dits.ltx_2._ltx2_try_fused_ada_values9"


def contiguous_rotary_tables(result, *args, **kwargs):
    cos, sin = result
    return cos.contiguous(), sin.contiguous()


def without_triton_ada_values(*args, **kwargs) -> None:
    return None


def register() -> tuple[str, ...]:
    for target in HOOK_TARGETS:
        HookRegistry.register(target, contiguous_rotary_tables, HookType.AFTER)
    HookRegistry.register(FUSED_ADA_VALUES, without_triton_ada_values, HookType.REPLACE)
    return (*HOOK_TARGETS, FUSED_ADA_VALUES)

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

MODELS = "sglang.multimodal_gen.runtime.models"
HOOK_TARGETS = (
    f"{MODELS}.dits.ltx_2.LTX2AudioVideoRotaryPosEmbed.forward",
    f"{MODELS}.adapter.ltx_2_connector.LTX2RotaryPosEmbed1d.forward",
)


def contiguous_rotary_tables(result, *args, **kwargs):
    cos, sin = result
    return cos.contiguous(), sin.contiguous()


def register() -> tuple[str, ...]:
    for target in HOOK_TARGETS:
        HookRegistry.register(target, contiguous_rotary_tables, HookType.AFTER)
    return HOOK_TARGETS

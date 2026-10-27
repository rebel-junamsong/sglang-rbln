# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.models.dits.ltx_2"
    ".LTX2AudioVideoRotaryPosEmbed.forward"
)


def contiguous_rotary_tables(result, *args, **kwargs):
    cos, sin = result
    return cos.contiguous(), sin.contiguous()


def register() -> str:
    HookRegistry.register(HOOK_TARGET, contiguous_rotary_tables, HookType.AFTER)
    return HOOK_TARGET

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

HOOK_TARGET = "sglang.multimodal_gen.configs.models.base.ArchConfig.__getattr__"


def extra_attr(self, name: str):
    extras = self.__dict__.get("extra_attrs")
    if extras is not None and name in extras:
        return extras[name]
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def register() -> str:
    HookRegistry.register(HOOK_TARGET, extra_attr, HookType.REPLACE)
    return HOOK_TARGET

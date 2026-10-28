# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
from types import ModuleType

CONTEXT_VARS = ("is_first_frame", "feat_cache", "feat_idx", "first_chunk")


class TraceableContextVar:
    def __init__(self, name: str, value=None) -> None:
        self.name = name
        self.value = value

    def get(self, *default):
        return self.value

    def set(self, value):
        token, self.value = self.value, value
        return token

    def reset(self, token) -> None:
        self.value = token


def register(module: ModuleType | None = None) -> list[str]:
    if module is None:
        from sglang.multimodal_gen.runtime.models.vaes import wanvae as module

    replaced = []
    for name in CONTEXT_VARS:
        if isinstance(var := getattr(module, name), contextvars.ContextVar):
            setattr(module, name, TraceableContextVar(name, var.get()))
            replaced.append(name)
    return replaced

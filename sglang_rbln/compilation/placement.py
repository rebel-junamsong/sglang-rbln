# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref

from torch import nn

from sglang_rbln import envs

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader."
    "PipelineComponentLoader.load_component"
)

_placements: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def place(module: nn.Module, devices: tuple[int, ...]) -> None:
    for submodule in module.modules():
        _placements[submodule] = devices


def devices_of(module: nn.Module) -> tuple[int, ...] | None:
    return _placements.get(module)


def place_component(load_component, component_name: str, *args, **kwargs):
    module, memory_usage = load_component(component_name, *args, **kwargs)
    devices = envs.SGLANG_RBLN_COMPONENT_DEVICES.get(component_name)
    if devices and isinstance(module, nn.Module):
        place(module, devices)
    return module, memory_usage


def register() -> str:
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    HookRegistry.register(HOOK_TARGET, place_component, HookType.AROUND)
    return HOOK_TARGET

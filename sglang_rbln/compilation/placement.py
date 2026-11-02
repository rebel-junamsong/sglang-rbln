# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import weakref

from torch import nn

from sglang_rbln import device, envs

logger = logging.getLogger(__name__)

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.loader.component_loaders.component_loader."
    "PipelineComponentLoader.load_component"
)

_placements: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# A text encoder runs once per request, before the DiT. With no explicit placement it
# runs tensor-parallel at the DiT's width on the next visible NPUs, if they hold it.
SPARE_NPU_COMPONENTS = ("text_encoder",)
RUNTIME_RESERVE = 1536 << 20


def place(module: nn.Module, devices: tuple[int, ...]) -> None:
    for submodule in module.modules():
        _placements[submodule] = devices


def devices_of(module: nn.Module) -> tuple[int, ...] | None:
    return _placements.get(module)


def visible_npus() -> int:
    if visible := os.environ.get("RBLN_VISIBLE_DEVICES", "").strip():
        return len([index for index in visible.split(",") if index.strip()])
    return len(device.device_nodes())


def weight_bytes(module: nn.Module) -> int:
    tensors = {id(t): t for t in (*module.parameters(), *module.buffers())}
    return sum(t.numel() * t.element_size() for t in tensors.values())


def spare_set() -> tuple[int, ...] | None:
    """The NPUs after the DiT's, as many as it has, when that many are visible."""
    width = envs.SGLANG_RBLN_NUM_DEVICES or 1
    if visible_npus() < 2 * width:
        return None
    return tuple(range(width, 2 * width))


def spare_devices(module: nn.Module) -> tuple[int, ...] | None:
    """As many NPUs after the DiT's as it has, if they can hold `module` together."""
    spare = spare_set()
    if spare is None:
        return None
    share = weight_bytes(module) // len(spare) + RUNTIME_RESERVE
    free = min(device.npu_memory_info(index).free for index in spare)
    if share > free:
        logger.info(
            "%s needs %.1f GiB on each of NPUs %s, which have %.1f GiB free; "
            "it stays with the DiT",
            type(module).__name__,
            share / (1 << 30),
            spare,
            free / (1 << 30),
        )
        return None
    return spare


def place_component(load_component, component_name: str, *args, **kwargs):
    module, memory_usage = load_component(component_name, *args, **kwargs)
    if not isinstance(module, nn.Module):
        return module, memory_usage
    devices = envs.SGLANG_RBLN_COMPONENT_DEVICES.get(component_name)
    if devices is None and component_name in SPARE_NPU_COMPONENTS:
        devices = spare_devices(module)
        if devices:
            logger.info("%s runs on the spare NPUs %s", component_name, devices)
    if devices:
        place(module, devices)
    return module, memory_usage


def register() -> str:
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    HookRegistry.register(HOOK_TARGET, place_component, HookType.AROUND)
    return HOOK_TARGET

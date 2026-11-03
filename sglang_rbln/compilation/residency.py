# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import functools
import os
from collections.abc import Callable
from typing import Any

import torch
from torch.overrides import TorchFunctionMode, _pop_mode_temporarily
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
from torch.utils.weak import WeakIdKeyDictionary

from sglang_rbln import envs
from sglang_rbln.compilation import placement
from sglang_rbln.compilation.backend import COMPILE_BACKEND_NAME, static_compile_kwargs

NPUS_PER_DEVICE_ENV = "RBLN_NPUS_PER_DEVICE"
DEVICE_MAP_ENV = "RBLN_DEVICE_MAP"
_TENSOR_QUERIES = frozenset(
    {
        torch.Tensor.size,
        torch.Tensor.dim,
        torch.Tensor.numel,
        torch.Tensor.stride,
        torch.Tensor.is_contiguous,
        torch.Tensor.element_size,
        torch.Tensor.is_floating_point,
        torch.Tensor.is_complex,
        torch.Tensor.to,
        torch.Tensor.cpu,
        torch.Tensor.__format__,
    }
)


_DESCRIPTORS = frozenset({"__get__", "__set__"})
# What rebel leaves on a compiled graph's device output: the view it wrote and its
# device config (rebel.core.io_placement).
PRODUCER_VIEW = "_rbln_producer_view"
PRODUCER_CONFIG = "_rbln_producer_config"


def device_groups() -> list[tuple[int, ...]]:
    """The NPU sets the pipeline runs on, each one logical device, the DiT's first.

    Then every component placed on NPUs of its own, the spare set a text encoder
    takes, and each NPU left over on its own.
    """
    width = envs.SGLANG_RBLN_NUM_DEVICES or 1
    visible = placement.visible_npus()
    groups = [tuple(range(min(width, visible)))]
    used = set(groups[0])
    for group in sorted(set(envs.SGLANG_RBLN_COMPONENT_DEVICES.values())):
        if group in groups:
            continue
        if shared := used & set(group):
            raise ValueError(
                f"SGLANG_RBLN_COMPONENT_DEVICES: NPUs {group} share {sorted(shared)} "
                "with another set; components share NPUs only as identical sets"
            )
        groups.append(group)
        used |= set(group)
    spare = placement.spare_set()
    if spare is not None and not used & set(spare):
        groups.append(spare)
        used |= set(spare)
    groups += [(npu,) for npu in range(visible) if npu not in used]
    return groups


def configure_device_group() -> None:
    if not (os.environ.get(DEVICE_MAP_ENV) or os.environ.get(NPUS_PER_DEVICE_ENV)):
        os.environ[DEVICE_MAP_ENV] = ",".join(
            "[" + ",".join(map(str, group)) + "]" for group in device_groups()
        )
    import torch_rbln  # noqa: F401


def logical_device_of(module: torch.nn.Module) -> int:
    """The logical device whose NPUs are the ones `module` is placed on."""
    devices = placement.devices_of(module)
    if devices is None:
        return 0
    from torch_rbln._internal.rsd_utils import get_physical_device_ids

    for index in range(torch.rbln.device_count()):
        if tuple(get_physical_device_ids(index) or ()) == tuple(devices):
            return index
    raise RuntimeError(
        f"no logical device spans NPUs {devices}; "
        f"{DEVICE_MAP_ENV}={os.environ.get(DEVICE_MAP_ENV)!r}"
    )


def npus_of(device: torch.device) -> tuple[int, ...]:
    from torch_rbln._internal.rsd_utils import get_physical_device_ids

    return tuple(get_physical_device_ids(device.index or 0) or ())


def layout_to_keep(tensor: torch.Tensor, device: torch.device, read_as: str | None):
    """The device config to write `tensor` in on `device`, or None for a dense copy.

    The layout a graph wrote can be kept on another NPU set of the same size: a config
    names its buffers by node within the set. It is kept only while the graph that reads
    it there reads that layout, the one it first ran with (`read_as`), which it was
    compiled against. Another layout comes over dense, and the runtime lays it out.
    """
    config = getattr(tensor, PRODUCER_CONFIG, None)
    if config is None or not tensor.is_contiguous():
        return None
    if len(npus_of(tensor.device)) != len(npus_of(device)):
        return None
    if read_as is not None and read_as != config.json:
        return None
    return config


def copy_across_devices(
    tensor: torch.Tensor, device: torch.device, config=None
) -> torch.Tensor:
    """`tensor` on `device`, copied on the device between the two NPU sets: in the
    layout `config` describes, which it keeps for the next graph, or else dense."""
    from rebel._C import vmem

    source = tensor if config is not None else tensor.contiguous()
    copied = torch.empty(source.shape, dtype=source.dtype, device=device)
    if config is not None:
        vmem.set_device_alloc(copied.data_ptr(), config)
        setattr(copied, PRODUCER_VIEW, getattr(source, PRODUCER_VIEW))
        setattr(copied, PRODUCER_CONFIG, config)
    if source.numel():
        vmem.copy(
            source.data_ptr(), copied.data_ptr(), source.numel() * source.element_size()
        )
    return copied


def _is_resident(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type == "rbln"


def resident_module(
    cache: dict, key: tuple, build: Callable[[], torch.nn.Module], tensors, devices=None
) -> torch.nn.Module:
    """`build()`, compiled for `devices` and cached under `key` when any of `tensors`
    lives on the NPUs, so it runs where they are; on the host, `build()` to run
    eagerly."""
    if not any(map(_is_resident, tensors)):
        return build()
    if key not in cache:
        module = build()
        if devices:
            placement.place(module, devices)
        module.compile(**static_compile_kwargs(module))
        cache[key] = module
    return cache[key]


class _Transfers:
    def __init__(self) -> None:
        self.moved: dict[torch.device, WeakIdKeyDictionary] = {}
        self.downloads: WeakIdKeyDictionary = WeakIdKeyDictionary()

    def to(self, value: Any, device: torch.device, read_as: str | None = None) -> Any:
        if not isinstance(value, torch.Tensor) or value.device == device:
            return value
        if value.device.type not in ("cpu", "rbln"):
            return value
        if value.device.type == "rbln":
            config = layout_to_keep(value, device, read_as)
            move = functools.partial(copy_across_devices, value, device, config)
        else:
            move = functools.partial(value.to, device)
        # An inference tensor has no version counter to tell an in-place update by.
        if value.is_inference():
            return move()
        moved = self.moved.setdefault(device, WeakIdKeyDictionary())
        cached = moved.get(value)
        if cached is not None and cached[0] == value._version:
            return cached[1]
        resident = move()
        moved[value] = (value._version, resident)
        return resident

    def download(self, value: Any) -> Any:
        if not _is_resident(value):
            return value
        cached = self.downloads.get(value)
        if cached is None:
            cached = self.downloads[value] = value.cpu()
        return cached


_transfers = _Transfers()


class HostBoundary(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _TENSOR_QUERIES or getattr(func, "__name__", None) in _DESCRIPTORS:
            return func(*args, **kwargs)
        flat = (*args, *kwargs.values())
        if any(
            _is_resident(leaf)
            for value in flat
            for leaf in torch.utils._pytree.tree_leaves(value)
        ):
            args, kwargs = tree_map(_transfers.download, (args, kwargs))
        return func(*args, **kwargs)


def _outside_boundary() -> contextlib.AbstractContextManager:
    stack = torch.overrides._get_current_function_mode_stack()
    if stack and isinstance(stack[-1], HostBoundary):
        return _pop_mode_temporarily()
    return contextlib.nullcontext()


def resident_call(module: torch.nn.Module, compiled: Callable) -> Callable:
    device: torch.device | None = None
    # The layout each input from another NPU set first came in, which the graph was
    # compiled against.
    read_as: dict[int, str] = {}

    @functools.wraps(compiled)
    def run(*args, **kwargs):
        nonlocal device
        with _outside_boundary():
            if device is None:
                device = torch.device("rbln", logical_device_of(module))
                module.to(device)
            leaves, spec = tree_flatten((args, kwargs))
            for index, leaf in enumerate(leaves):
                config = getattr(leaf, PRODUCER_CONFIG, None)
                if config is not None and leaf.device != device:
                    read_as.setdefault(index, config.json)
                leaves[index] = _transfers.to(leaf, device, read_as.get(index))
            args, kwargs = tree_unflatten(leaves, spec)
            return compiled(*args, **kwargs)

    return run


def install(module: torch.nn.Module) -> None:
    compiled = module._compiled_call_impl
    if compiled is not None and not getattr(compiled, "_rbln_resident", False):
        module._compiled_call_impl = resident_call(module, compiled)
        module._compiled_call_impl._rbln_resident = True


STRATEGIES = (
    "sglang.multimodal_gen.runtime.managers.memory_managers."
    "component_residency_strategies"
)
MOVE_TARGET = f"{STRATEGIES}._module_to_local_device"
READY_TARGET = f"{STRATEGIES}._module_ready_on_local_device"


def _on_device(module: torch.nn.Module) -> bool:
    return any(map(_is_resident, (*module.parameters(), *module.buffers())))


def keep_resident_placement(move, module: torch.nn.Module, *args, **kwargs) -> None:
    """Leave a component whose regions hold it on their NPUs where they put it.

    The component manager moves each component to the platform's local device before
    a use, and that device is the host here: it would take the weights off the NPUs
    every request, and the next call would run against host weights.
    """
    if not _on_device(module):
        move(module, *args, **kwargs)


def resident_is_ready(ready, module: torch.nn.Module, *args, **kwargs) -> bool:
    return _on_device(module) or ready(module, *args, **kwargs)


COMPILE_TARGET = "torch.nn.modules.module.Module.compile"
STAGE_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.PipelineStage.__call__"
)


def install_after_compile(result, module: torch.nn.Module, *args, **kwargs):
    if kwargs.get("backend") == COMPILE_BACKEND_NAME:
        install(module)
    return result


def stage_on_host_boundary(call, stage, *args, **kwargs):
    with HostBoundary():
        return call(stage, *args, **kwargs)


def register() -> tuple[str, ...]:
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    HookRegistry.register(COMPILE_TARGET, install_after_compile, HookType.AFTER)
    HookRegistry.register(STAGE_TARGET, stage_on_host_boundary, HookType.AROUND)
    HookRegistry.register(MOVE_TARGET, keep_resident_placement, HookType.AROUND)
    HookRegistry.register(READY_TARGET, resident_is_ready, HookType.AROUND)
    return COMPILE_TARGET, STAGE_TARGET, MOVE_TARGET, READY_TARGET

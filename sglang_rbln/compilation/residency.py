# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import functools
import os
from collections.abc import Callable
from typing import Any

import torch
from torch.overrides import TorchFunctionMode, _pop_mode_temporarily
from torch.utils._pytree import tree_map
from torch.utils.weak import WeakIdKeyDictionary

from sglang_rbln import envs
from sglang_rbln.compilation import placement
from sglang_rbln.compilation.backend import COMPILE_BACKEND_NAME

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


def copy_across_devices(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """`tensor` on `device`, copied on the device between the two NPU sets."""
    from rebel._C import vmem

    source = tensor.contiguous()
    copied = torch.empty(source.shape, dtype=source.dtype, device=device)
    if source.numel():
        vmem.copy(
            source.data_ptr(), copied.data_ptr(), source.numel() * source.element_size()
        )
    return copied


def _is_resident(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type == "rbln"


class _Transfers:
    def __init__(self) -> None:
        self.moved: dict[torch.device, WeakIdKeyDictionary] = {}
        self.downloads: WeakIdKeyDictionary = WeakIdKeyDictionary()

    def to(self, value: Any, device: torch.device) -> Any:
        if not isinstance(value, torch.Tensor) or value.device == device:
            return value
        if value.device.type not in ("cpu", "rbln"):
            return value
        moved = self.moved.setdefault(device, WeakIdKeyDictionary())
        cached = moved.get(value)
        if cached is not None and cached[0] == value._version:
            return cached[1]
        if value.device.type == "rbln":
            resident = copy_across_devices(value, device)
        else:
            resident = value.to(device)
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

    @functools.wraps(compiled)
    def run(*args, **kwargs):
        nonlocal device
        with _outside_boundary():
            if device is None:
                device = torch.device("rbln", logical_device_of(module))
                module.to(device)
            args, kwargs = tree_map(
                lambda value: _transfers.to(value, device), (args, kwargs)
            )
            return compiled(*args, **kwargs)

    return run


def install(module: torch.nn.Module) -> None:
    compiled = module._compiled_call_impl
    if compiled is not None and not getattr(compiled, "_rbln_resident", False):
        module._compiled_call_impl = resident_call(module, compiled)
        module._compiled_call_impl._rbln_resident = True


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


def register() -> tuple[str, str]:
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    HookRegistry.register(COMPILE_TARGET, install_after_compile, HookType.AFTER)
    HookRegistry.register(STAGE_TARGET, stage_on_host_boundary, HookType.AROUND)
    return COMPILE_TARGET, STAGE_TARGET

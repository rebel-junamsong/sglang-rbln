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
from sglang_rbln.compilation.backend import COMPILE_BACKEND_NAME

DEVICE = "rbln:0"
NPUS_PER_DEVICE_ENV = "RBLN_NPUS_PER_DEVICE"
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


def configure_device_group() -> None:
    os.environ.setdefault(NPUS_PER_DEVICE_ENV, str(envs.SGLANG_RBLN_NUM_DEVICES))
    import torch_rbln  # noqa: F401


def _is_resident(value: Any) -> bool:
    return isinstance(value, torch.Tensor) and value.device.type == "rbln"


class _Transfers:
    def __init__(self) -> None:
        self.uploads: WeakIdKeyDictionary = WeakIdKeyDictionary()
        self.downloads: WeakIdKeyDictionary = WeakIdKeyDictionary()

    def upload(self, value: Any) -> Any:
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            return value
        cached = self.uploads.get(value)
        if cached is not None and cached[0] == value._version:
            return cached[1]
        resident = value.to(DEVICE)
        self.uploads[value] = (value._version, resident)
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
    placed = False

    @functools.wraps(compiled)
    def run(*args, **kwargs):
        nonlocal placed
        with _outside_boundary():
            if not placed:
                module.to(DEVICE)
                placed = True
            args, kwargs = tree_map(_transfers.upload, (args, kwargs))
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

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import gc
import logging
import os
import sys
import traceback
import types
import weakref
from typing import Any

import torch
from torch._dynamo.eval_frame import _debug_get_cache_entry_list

from sglang_rbln import device, envs
from sglang_rbln.compilation import device_views, placement, shapes, timeline

logger = logging.getLogger(__name__)

COMPILE_BACKEND_NAME = "rbln"

_device_set_contexts: dict[tuple[int, ...] | int | None, Any] = {}
_module_contexts: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _new_context():
    from rebel import CompileContext

    return CompileContext(use_weight_sharing=True)


def device_set_context(device_set: tuple[int, ...] | int | None):
    context = _device_set_contexts.get(device_set)
    if context is None:
        context = _device_set_contexts[device_set] = _new_context()
    return context


def module_context(module: torch.nn.Module):
    context = _module_contexts.get(module)
    if context is None:
        context = _module_contexts[module] = _new_context()
    return context


def on_one_device(options: dict[str, Any], device: int, module: torch.nn.Module):
    return options | {
        "num_devices": 1,
        "device": device,
        "compile_context": module_context(module),
    }


def compile_options(module: torch.nn.Module) -> dict[str, Any]:
    devices = placement.devices_of(module)
    num_devices = envs.SGLANG_RBLN_NUM_DEVICES
    options: dict[str, Any] = {
        "compile_context": device_set_context(devices or num_devices)
    }
    modes = [
        mode
        for mode, enabled in (
            ("compile_only", envs.SGLANG_RBLN_COMPILE_ONLY),
            ("strict", envs.SGLANG_RBLN_COMPILE_STRICT),
        )
        if enabled
    ]
    if modes:
        options["mode"] = modes
    if not envs.SGLANG_RBLN_DISABLE_COMPILE_CACHE:
        options["cache_dir"] = str(envs.SGLANG_RBLN_CACHE_DIR)
    if envs.SGLANG_RBLN_STATIC_OUTPUT:
        options["use_static_output"] = True
    if devices:
        options["device"] = list(devices)
        options["num_devices"] = len(devices)
    elif num_devices:
        options["num_devices"] = num_devices
    return options


def static_compile_kwargs(module: torch.nn.Module) -> dict[str, Any]:
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    compile_kwargs = build_torch_compile_kwargs(mode=None, module=module)
    compile_kwargs["dynamic"] = False
    return compile_kwargs


COMPILED = "_rbln_compiled"


def forget_frames(error: BaseException) -> None:
    """Drop the locals of the frames an error's traceback holds, and its causes'.

    A graph that did not fit fails inside its own call, so the frames of that call
    hold its runtime: while the caught error lives, releasing the graph frees
    nothing, and the given-up size's buffers stay on the device beside the next.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        error = error.__cause__ or error.__context__


def release_graphs(module: torch.nn.Module) -> None:
    """Drop the graphs compiled for `module`'s forward, so the device memory their
    runtimes hold is freed; its next call compiles again, from the artifact cache.

    The graphs are cached on the forward of `module`'s class, so every instance of
    that class loses its graphs too.
    """
    forward = type(module).forward
    _release_code(forward.__code__, forward.__globals__, set())
    gc.collect()


def _release_code(code: types.CodeType, scope: dict, released: set) -> None:
    """Drop the graphs cached on `code` and on the functions its graph breaks resume in.

    Dynamo keeps each compiled graph, and the resume function a graph break leaves the
    rest of the forward to, in the module's globals under a name only the cached code
    refers to. Resetting the forward's cache alone kept them, and their runtimes held
    their device buffers for good.
    """
    if code in released:
        return
    released.add(code)
    names = {
        name
        for entry in _debug_get_cache_entry_list(code)
        for name in entry.code.co_names
        if name.startswith(("__compiled_fn", "__resume_at"))
    }
    torch._dynamo.eval_frame.remove_from_cache(code)
    for name in names:
        value = scope.pop(name, None)
        if name.startswith("__resume_at") and value is not None:
            _release_code(value.__code__, scope, released)


def compile_once(module: torch.nn.Module, what: str, compile_kwargs: dict) -> None:
    module.compile(**compile_kwargs)
    setattr(module, COMPILED, True)
    logger.info("Compiling the %s: %s", what, compile_kwargs)


def compile_on_one_device(module: torch.nn.Module, what: str) -> None:
    """Compile `module` for one NPU: its own if placed, else the last TP one.

    With resident regions each NPU set is one logical device, and a graph reads a
    device input only on the device that holds it, so `module` then compiles for
    the whole set its inputs live on.
    """
    compile_kwargs = static_compile_kwargs(module)
    if device.resident_regions():
        compile_once(module, what, compile_kwargs)
        return
    options = compile_kwargs["options"]
    if devices := placement.devices_of(module):
        if len(devices) > 1:
            compile_kwargs["options"] = on_one_device(options, devices[-1], module)
    elif num_devices := options.get("num_devices"):
        compile_kwargs["options"] = on_one_device(options, num_devices - 1, module)
    compile_once(module, what, compile_kwargs)


def describe_graph(graph: torch.fx.GraphModule, example_inputs) -> str:
    inputs = [
        f"{'x'.join(map(str, tensor.shape))}:{str(tensor.dtype).removeprefix('torch.')}"
        if isinstance(tensor, torch.Tensor)
        else type(tensor).__name__
        for tensor in example_inputs
    ]
    node_count = sum(node.op.startswith("call_") for node in graph.graph.nodes)
    return f"{node_count} ops, inputs [{', '.join(inputs)}]"


def contiguous_device_inputs(args):
    # A dummy device runs no graph, the eager copy behind contiguous() included; a
    # compile-only run reads only the shape, so an empty tensor of it stands in.
    contiguous = (
        torch.empty_like if envs.SGLANG_RBLN_COMPILE_ONLY else torch.Tensor.contiguous
    )
    return [
        contiguous(value, memory_format=torch.contiguous_format)
        if isinstance(value, torch.Tensor)
        and value.device.type == "rbln"
        and not value.is_contiguous()
        else value
        for value in args
    ]


def on_device(value: torch.Tensor) -> bool:
    return value.device.type == "rbln"


def _contiguous_callable(compiled, rebased: device_views.RebasedViews | None = None):
    @functools.wraps(compiled)
    def run(*args, **kwargs):
        if rebased is not None:
            args = rebased.bases(args)
        return compiled(*contiguous_device_inputs(args), **kwargs)

    return run


def _instrumented_backend(backend):
    @functools.wraps(backend)
    def wrapped(graph, example_inputs, *args, **kwargs):
        if shapes.repeats_compile(graph, example_inputs):
            logger.warning(
                "rbln recompile at unchanged shape key %s, guarded on more than "
                "shapes: %s",
                shapes.current_shape_key.get(),
                describe_graph(graph, example_inputs),
            )
        # A sliced device view reaches the graph as its base, sliced inside it,
        # instead of as an eager contiguous copy (see device_views).
        rebased = device_views.rebase_strided_views(graph, example_inputs, on_device)
        if rebased is not None:
            logger.info(
                "rbln graph takes %d sliced device views as their bases: %s",
                len(rebased.slices),
                sorted(rebased.slices),
            )
            example_inputs = rebased.bases(example_inputs)
        compiled = backend(
            graph, contiguous_device_inputs(example_inputs), *args, **kwargs
        )
        if any(
            isinstance(value, torch.Tensor) and on_device(value)
            for value in example_inputs
        ):
            compiled = _contiguous_callable(compiled, rebased)
        if envs.SGLANG_RBLN_LOG_TIMELINE:
            compiled = timeline.timed_callable(
                compiled, timeline.graph_label(example_inputs)
            )
        return compiled

    wrapped.__wrapped_backend__ = backend
    return wrapped


def ensure_compile_backend() -> None:
    import rebel.core.torch_compile  # noqa: F401
    from torch._dynamo.backends import registry

    torch._dynamo.config.recompile_limit = sys.maxsize
    torch._dynamo.config.accumulated_recompile_limit = sys.maxsize
    torch._dynamo.config.automatic_dynamic_shapes = False
    backend = registry._COMPILER_FNS.get(COMPILE_BACKEND_NAME)
    if backend is None:
        raise RuntimeError(
            f"rebel-compiler did not register the {COMPILE_BACKEND_NAME!r} "
            "torch.compile backend"
        )
    for flag in ("RBLN_POOL_IO_BUFFERS", "RBLN_RSD_GROUPING"):
        os.environ.setdefault(flag, "1")
    if not hasattr(backend, "__wrapped_backend__"):
        registry._COMPILER_FNS[COMPILE_BACKEND_NAME] = _instrumented_backend(backend)

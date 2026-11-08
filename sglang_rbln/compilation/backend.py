# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import gc
import itertools
import logging
import os
import pickle
import sys
import traceback
import types
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from sglang_rbln import device, envs
from sglang_rbln.compilation import device_views, placement, shapes, timeline

logger = logging.getLogger(__name__)

COMPILE_BACKEND_NAME = "rbln"
RUNTIME_REPORT_EVERY = 100

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
    torch._dynamo.eval_frame.remove_from_cache(type(module).forward)
    gc.collect()


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


def single_device_compile_kwargs(module: torch.nn.Module) -> dict[str, Any]:
    compile_kwargs = static_compile_kwargs(module)
    options = compile_kwargs["options"]
    if "device" not in options and (num_devices := options.get("num_devices", 1)) > 1:
        compile_kwargs["options"] = on_one_device(options, num_devices - 1, module)
    return compile_kwargs


def describe_graph(graph: torch.fx.GraphModule, example_inputs) -> str:
    inputs = [
        f"{'x'.join(map(str, tensor.shape))}:{str(tensor.dtype).removeprefix('torch.')}"
        if isinstance(tensor, torch.Tensor)
        else type(tensor).__name__
        for tensor in example_inputs
    ]
    node_count = sum(node.op.startswith("call_") for node in graph.graph.nodes)
    return f"{node_count} ops, inputs [{', '.join(inputs)}]"


_TIMER_TOTALS = frozenset({"seq", "total", "total_host", "total_device", "total_ccl"})


@dataclass
class RuntimeReportTotals:
    calls: int = 0
    device_us: int = 0
    host_us: int = 0
    ccl_us: int = 0
    prepare_us: int = 0
    node_us: dict[str, int] = field(default_factory=dict)

    def add(self, reports: list[dict]) -> None:
        for report in reports:
            if report.get("type") == "timer":
                self.calls += 1
                self.device_us += report.get("total_device", 0)
                self.host_us += report.get("total_host", 0)
                self.ccl_us += report.get("total_ccl", 0)
                for key, value in report.items():
                    if key not in _TIMER_TOTALS and isinstance(value, int):
                        self.node_us[key] = self.node_us.get(key, 0) + value
            elif report.get("type") == "prep":
                self.prepare_us += report.get("prepare_input_us", 0)
                self.prepare_us += report.get("prepare_output_us", 0)

    def top_nodes(self, count: int = 5) -> list[tuple[str, int]]:
        return sorted(self.node_us.items(), key=lambda item: -item[1])[:count]

    def summary(self) -> str:
        total = self.device_us + self.host_us + self.ccl_us + self.prepare_us
        share = (self.device_us / total * 100) if total else 0.0
        return (
            f"{self.calls} runs: device {self.device_us / 1e3:.1f} ms, "
            f"host ops {self.host_us / 1e3:.1f} ms, ccl {self.ccl_us / 1e3:.1f} ms, "
            f"buffer prep {self.prepare_us / 1e3:.1f} ms, device share {share:.1f}%, "
            f"top nodes {[(name, round(us / 1e3, 1)) for name, us in self.top_nodes()]}"
        )


runtime_report_totals = RuntimeReportTotals()


def _reporting_callable(compiled):
    import rebel

    @functools.wraps(compiled)
    def run(*args, **kwargs):
        with rebel.capture_reports() as reports:
            outputs = compiled(*args, **kwargs)
        runtime_report_totals.add(reports)
        if runtime_report_totals.calls % RUNTIME_REPORT_EVERY == 0:
            logger.info("rbln runtime: %s", runtime_report_totals.summary())
        return outputs

    return run


_dump_counter = itertools.count()


def _dropped_on_dump() -> tuple[type, ...]:
    from torch.distributed import ProcessGroup

    return (ProcessGroup,)


class _DumpPickler(pickle.Pickler):
    def reducer_override(self, obj):
        if isinstance(obj, _dropped_on_dump()):
            return type(None), ()
        return NotImplemented


_dump_pickle = types.ModuleType("sglang_rbln_dump_pickle")
_dump_pickle.__dict__.update(vars(pickle))
_dump_pickle.Pickler = _DumpPickler


def dump_graph(graph: torch.fx.GraphModule, example_inputs, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"graph_{os.getpid()}_{next(_dump_counter)}"
    stem.with_suffix(".txt").write_text(graph.print_readable(print_output=False))
    tensors = [
        value if isinstance(value, torch.Tensor) else repr(value)
        for value in example_inputs
    ]
    torch.save(tensors, stem.with_suffix(".inputs.pt"))
    torch.save(graph.state_dict(), stem.with_suffix(".state.pt"))
    try:
        torch.save(graph, stem.with_suffix(".module.pt"), pickle_module=_dump_pickle)
    except (TypeError, pickle.PicklingError) as error:
        stem.with_suffix(".module.pt").unlink(missing_ok=True)
        logger.warning("rbln graph %s kept without its module: %s", stem, error)
    return stem


def output_deviation(actual, expected) -> tuple[float, float]:
    pairs = [
        (a.detach().cpu().float(), e.detach().cpu().float())
        for a, e in zip(
            torch.utils._pytree.tree_leaves(actual),
            torch.utils._pytree.tree_leaves(expected),
            strict=True,
        )
        if isinstance(a, torch.Tensor) and a.is_floating_point()
    ]
    if not pairs:
        return 0.0, 0.0
    max_abs = max((a - e).abs().max().item() for a, e in pairs)
    scale = max(e.abs().max().item() for _, e in pairs) or 1.0
    return max_abs, max_abs / scale


def _checking_callable(compiled, graph: torch.fx.GraphModule, name: str):
    checked = False

    @functools.wraps(compiled)
    def run(*args, **kwargs):
        nonlocal checked
        outputs = compiled(*args, **kwargs)
        if not checked:
            checked = True
            with torch.no_grad():
                expected = torch.func.functional_call(
                    graph,
                    {
                        name: tensor.cpu()
                        for name, tensor in itertools.chain(
                            graph.named_parameters(), graph.named_buffers()
                        )
                    },
                    tuple(
                        value.cpu() if isinstance(value, torch.Tensor) else value
                        for value in args
                    ),
                )
            max_abs, relative = output_deviation(outputs, expected)
            logger.info(
                "rbln graph check %s: max abs %.4g, relative to output range %.4g",
                name,
                max_abs,
                relative,
            )
        return outputs

    return run


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
        if envs.SGLANG_RBLN_LOG_GRAPHS:
            logger.info(
                "rbln graph for %s: %s",
                shapes.current_shape_key.get(),
                describe_graph(graph, example_inputs),
            )
        if shapes.repeats_compile(graph, example_inputs):
            logger.warning(
                "rbln recompile at unchanged shape key %s, guarded on more than "
                "shapes: %s",
                shapes.current_shape_key.get(),
                describe_graph(graph, example_inputs),
            )
        if dump_dir := envs.SGLANG_RBLN_DUMP_GRAPHS:
            logger.info(
                "rbln graph dumped to %s", dump_graph(graph, example_inputs, dump_dir)
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
        if envs.SGLANG_RBLN_CHECK_GRAPHS:
            compiled = _checking_callable(
                compiled, graph, describe_graph(graph, example_inputs)
            )
        if envs.SGLANG_RBLN_RUNTIME_REPORTS:
            compiled = _reporting_callable(compiled)
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
    if envs.SGLANG_RBLN_RUNTIME_REPORTS:
        os.environ.setdefault("RBLN_RUNTIME_TIMER", "1")
    if not hasattr(backend, "__wrapped_backend__"):
        registry._COMPILER_FNS[COMPILE_BACKEND_NAME] = _instrumented_backend(backend)

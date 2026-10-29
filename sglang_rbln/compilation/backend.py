# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import itertools
import logging
import os
import pickle
import sys
import types
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from sglang_rbln import envs

logger = logging.getLogger(__name__)

COMPILE_BACKEND_NAME = "rbln"
RUNTIME_REPORT_EVERY = 100

_compile_contexts: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def compile_options(module: torch.nn.Module) -> dict[str, Any]:
    from rebel import CompileContext

    context = _compile_contexts.get(module)
    if context is None:
        context = _compile_contexts[module] = CompileContext(use_weight_sharing=True)
    options: dict[str, Any] = {"compile_context": context}
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
    if num_devices := envs.SGLANG_RBLN_NUM_DEVICES:
        options["num_devices"] = num_devices
    return options


def static_compile_kwargs(module: torch.nn.Module) -> dict[str, Any]:
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    compile_kwargs = build_torch_compile_kwargs(mode=None, module=module)
    compile_kwargs["dynamic"] = False
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
    return [
        value.contiguous()
        if isinstance(value, torch.Tensor)
        and value.device.type == "rbln"
        and not value.is_contiguous()
        else value
        for value in args
    ]


def _contiguous_callable(compiled):
    @functools.wraps(compiled)
    def run(*args, **kwargs):
        return compiled(*contiguous_device_inputs(args), **kwargs)

    return run


def _instrumented_backend(backend):
    @functools.wraps(backend)
    def wrapped(graph, example_inputs, *args, **kwargs):
        if envs.SGLANG_RBLN_LOG_GRAPHS:
            logger.info("rbln graph: %s", describe_graph(graph, example_inputs))
        if dump_dir := envs.SGLANG_RBLN_DUMP_GRAPHS:
            logger.info(
                "rbln graph dumped to %s", dump_graph(graph, example_inputs, dump_dir)
            )
        compiled = backend(
            graph, contiguous_device_inputs(example_inputs), *args, **kwargs
        )
        if any(
            isinstance(value, torch.Tensor) and value.device.type == "rbln"
            for value in example_inputs
        ):
            compiled = _contiguous_callable(compiled)
        if envs.SGLANG_RBLN_CHECK_GRAPHS:
            compiled = _checking_callable(
                compiled, graph, describe_graph(graph, example_inputs)
            )
        if envs.SGLANG_RBLN_RUNTIME_REPORTS:
            compiled = _reporting_callable(compiled)
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

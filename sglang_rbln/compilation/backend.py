# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import itertools
import logging
import os
import weakref
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

COMPILE_BACKEND_NAME = "rbln"
COMPILE_ONLY_ENV = "SGLANG_RBLN_COMPILE_ONLY"
STRICT_ENV = "SGLANG_RBLN_COMPILE_STRICT"
LOG_GRAPHS_ENV = "SGLANG_RBLN_LOG_GRAPHS"
RUNTIME_REPORTS_ENV = "SGLANG_RBLN_RUNTIME_REPORTS"
DUMP_GRAPHS_ENV = "SGLANG_RBLN_DUMP_GRAPHS"
RUNTIME_REPORT_EVERY = 100
CACHE_DIR_ENV = "SGLANG_RBLN_CACHE_DIR"
DISABLE_CACHE_ENV = "SGLANG_RBLN_DISABLE_COMPILE_CACHE"
DEFAULT_CACHE_DIR = "~/.cache/sglang_rbln"
DYNAMO_CACHE_SIZE_LIMIT = 64

_compile_contexts: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def cache_dir() -> str | None:
    if _env_flag(DISABLE_CACHE_ENV):
        return None
    return str(Path(os.environ.get(CACHE_DIR_ENV) or DEFAULT_CACHE_DIR).expanduser())


def compile_options(module: torch.nn.Module | None = None) -> dict[str, Any]:
    from rebel import CompileContext

    options: dict[str, Any] = {}
    if module is not None:
        context = _compile_contexts.get(module)
        if context is None:
            context = _compile_contexts[module] = CompileContext()
        options["compile_context"] = context
    modes = [
        mode
        for mode, env in (("compile_only", COMPILE_ONLY_ENV), ("strict", STRICT_ENV))
        if _env_flag(env)
    ]
    if modes:
        options["mode"] = modes
    compile_cache_dir = cache_dir()
    if compile_cache_dir:
        options["cache_dir"] = compile_cache_dir
        options["mega_cache_only"] = True
    return options


def describe_graph(graph: torch.fx.GraphModule, example_inputs) -> str:
    inputs = [
        f"{'x'.join(map(str, tensor.shape))}:{str(tensor.dtype).removeprefix('torch.')}"
        if isinstance(tensor, torch.Tensor)
        else type(tensor).__name__
        for tensor in example_inputs
    ]
    node_count = sum(node.op.startswith("call_") for node in graph.graph.nodes)
    return f"{node_count} ops, inputs [{', '.join(inputs)}]"


class RuntimeReportTotals:
    def __init__(self) -> None:
        self.calls = 0
        self.device_us = 0
        self.host_us = 0
        self.ccl_us = 0
        self.prepare_us = 0
        self.host_ops: set[str] = set()

    def add(self, reports: list[dict]) -> None:
        for report in reports:
            if report.get("type") == "timer":
                self.calls += 1
                self.device_us += report.get("total_device", 0)
                self.host_us += report.get("total_host", 0)
                self.ccl_us += report.get("total_ccl", 0)
                self.host_ops.update(key for key in report if key.startswith("host_"))
            elif report.get("type") == "prep":
                self.prepare_us += report.get("prepare_input_us", 0)
                self.prepare_us += report.get("prepare_output_us", 0)

    def summary(self) -> str:
        total = self.device_us + self.host_us + self.ccl_us + self.prepare_us
        share = (self.device_us / total * 100) if total else 0.0
        return (
            f"{self.calls} runs: device {self.device_us / 1e3:.1f} ms, "
            f"host ops {self.host_us / 1e3:.1f} ms, ccl {self.ccl_us / 1e3:.1f} ms, "
            f"buffer prep {self.prepare_us / 1e3:.1f} ms, device share {share:.1f}%, "
            f"host ops {sorted(self.host_ops) or 'none'}"
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


def dump_graph(graph: torch.fx.GraphModule, example_inputs, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / f"graph_{os.getpid()}_{next(_dump_counter)}"
    stem.with_suffix(".txt").write_text(graph.print_readable(print_output=False))
    torch.save(list(example_inputs), stem.with_suffix(".inputs.pt"))
    try:
        exported = torch.export.export(graph, tuple(example_inputs), strict=False)
        torch.export.save(exported, stem.with_suffix(".pt2"))
    except Exception as error:
        stem.with_suffix(".export_error.txt").write_text(repr(error))
    return stem


def _logging_backend(backend):
    @functools.wraps(backend)
    def wrapped(graph, example_inputs, *args, **kwargs):
        if _env_flag(LOG_GRAPHS_ENV):
            logger.info("rbln graph: %s", describe_graph(graph, example_inputs))
        if dump_dir := os.environ.get(DUMP_GRAPHS_ENV):
            logger.info(
                "rbln graph dumped to %s",
                dump_graph(graph, example_inputs, Path(dump_dir)),
            )
        compiled = backend(graph, example_inputs, *args, **kwargs)
        if _env_flag(RUNTIME_REPORTS_ENV):
            compiled = _reporting_callable(compiled)
        return compiled

    wrapped.__wrapped_backend__ = backend
    return wrapped


def ensure_compile_backend() -> str:
    import rebel.core.torch_compile  # noqa: F401
    from torch._dynamo.backends import registry

    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, DYNAMO_CACHE_SIZE_LIMIT
    )
    torch._dynamo.config.automatic_dynamic_shapes = False
    backend = registry._COMPILER_FNS.get(COMPILE_BACKEND_NAME)
    if backend is None:
        raise RuntimeError(
            f"rebel-compiler did not register the {COMPILE_BACKEND_NAME!r} "
            "torch.compile backend"
        )
    if _env_flag(RUNTIME_REPORTS_ENV):
        os.environ.setdefault("RBLN_RUNTIME_TIMER", "1")
    diagnostics = (
        _env_flag(LOG_GRAPHS_ENV)
        or _env_flag(RUNTIME_REPORTS_ENV)
        or bool(os.environ.get(DUMP_GRAPHS_ENV))
    )
    if diagnostics and not hasattr(backend, "__wrapped_backend__"):
        registry._COMPILER_FNS[COMPILE_BACKEND_NAME] = _logging_backend(backend)
    return COMPILE_BACKEND_NAME

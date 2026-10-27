# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
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


def _logging_backend(backend):
    @functools.wraps(backend)
    def wrapped(graph, example_inputs, *args, **kwargs):
        logger.info("rbln graph: %s", describe_graph(graph, example_inputs))
        return backend(graph, example_inputs, *args, **kwargs)

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
    if _env_flag(LOG_GRAPHS_ENV) and not hasattr(backend, "__wrapped_backend__"):
        registry._COMPILER_FNS[COMPILE_BACKEND_NAME] = _logging_backend(backend)
    return COMPILE_BACKEND_NAME

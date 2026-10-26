# SPDX-License-Identifier: Apache-2.0
"""``torch.compile`` backend that compiles FX graphs with rebel-compiler."""

from __future__ import annotations

import logging
import os
import weakref
from collections.abc import Sequence
from typing import Any

import torch
import torch.fx as fx

logger = logging.getLogger(__name__)

COMPILE_BACKEND_NAME = "rbln"
COMPILE_ONLY_ENV = "SGLANG_RBLN_COMPILE_ONLY"
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
    return os.path.expanduser(os.environ.get(CACHE_DIR_ENV) or DEFAULT_CACHE_DIR)


def compile_options(module: torch.nn.Module | None = None) -> dict[str, Any]:
    from rebel import CompileContext

    options: dict[str, Any] = {}
    if module is not None:
        context = _compile_contexts.get(module)
        if context is None:
            context = _compile_contexts[module] = CompileContext()
        options["compile_context"] = context
    if _env_flag(COMPILE_ONLY_ENV):
        options["mode"] = ["compile_only"]
    compile_cache_dir = cache_dir()
    if compile_cache_dir:
        options["cache_dir"] = compile_cache_dir
        options["mega_cache_only"] = True
    return options


def _describe_inputs(graph: fx.GraphModule, example_inputs: Sequence[Any]) -> str:
    placeholder_names = [n.name for n in graph.graph.nodes if n.op == "placeholder"]
    described = []
    for index, example in enumerate(example_inputs):
        name = (
            placeholder_names[index]
            if index < len(placeholder_names)
            else f"arg{index}"
        )
        if isinstance(example, torch.Tensor):
            shape = "x".join(str(d) for d in example.shape)
            described.append(
                f"{name}:{str(example.dtype).removeprefix('torch.')}[{shape}]"
            )
        else:
            described.append(f"{name}:{type(example).__name__}")
    return ", ".join(described)


def describe_graph_inputs(graph: fx.GraphModule, example_inputs: Sequence[Any]) -> str:
    return _describe_inputs(graph, example_inputs)


def ensure_compile_backend() -> str:
    """Import rebel's dynamo backend, which registers itself as ``rbln``."""
    import rebel.core.torch_compile  # noqa: F401
    from torch._dynamo import list_backends

    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, DYNAMO_CACHE_SIZE_LIMIT
    )
    if COMPILE_BACKEND_NAME not in list_backends():
        raise RuntimeError(
            f"rebel-compiler did not register the {COMPILE_BACKEND_NAME!r} "
            "torch.compile backend"
        )
    return COMPILE_BACKEND_NAME

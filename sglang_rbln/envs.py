# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    RBLN_FORCE_NPU_NAME: str | None = None
    SGLANG_RBLN_EXECUTION_MODE: str = "auto"
    SGLANG_RBLN_CACHE_DIR: Path = Path("~/.cache/sglang_rbln").expanduser()
    SGLANG_RBLN_DISABLE_COMPILE_CACHE: bool = False
    SGLANG_RBLN_COMPILE_ONLY: bool = False
    SGLANG_RBLN_COMPILE_STRICT: bool = True
    SGLANG_RBLN_STATIC_OUTPUT: bool = False
    SGLANG_RBLN_NUM_DEVICES: int | None = None
    SGLANG_RBLN_COMPILE_VAE: bool = False
    SGLANG_RBLN_CHECK_GRAPHS: bool = False
    SGLANG_RBLN_LOG_GRAPHS: bool = False
    SGLANG_RBLN_DUMP_GRAPHS: Path | None = None
    SGLANG_RBLN_RUNTIME_REPORTS: bool = False
    SGLANG_RBLN_PROFILE_DIR: Path | None = None
    SGLANG_RBLN_PROFILE_REQUEST: int = 3
    SGLANG_RBLN_VOCODER_ON_NPU: bool = False
    SGLANG_RBLN_BLOCKS_PER_REGION: int | None = None
    SGLANG_RBLN_FLUX3_ONE_GRAPH: bool = False


def _flag(name: str, default: bool = False) -> Callable[[], bool]:
    def parse() -> bool:
        value = os.environ.get(name, "").strip().lower()
        return value in ("1", "true", "yes") if value else default

    return parse


def _optional(name: str, parse: Callable[[str], Any] = str) -> Callable[[], Any]:
    return lambda: parse(value) if (value := os.environ.get(name)) else None


def _int(name: str, default: int) -> Callable[[], int]:
    return lambda: int(os.environ.get(name) or default)


environment_variables: dict[str, Callable[[], Any]] = {
    "RBLN_FORCE_NPU_NAME": _optional("RBLN_FORCE_NPU_NAME"),
    "SGLANG_RBLN_EXECUTION_MODE": lambda: (
        os.environ.get("SGLANG_RBLN_EXECUTION_MODE", "").strip().lower() or "auto"
    ),
    "SGLANG_RBLN_CACHE_DIR": lambda: Path(
        os.environ.get("SGLANG_RBLN_CACHE_DIR") or "~/.cache/sglang_rbln"
    ).expanduser(),
    "SGLANG_RBLN_DISABLE_COMPILE_CACHE": _flag("SGLANG_RBLN_DISABLE_COMPILE_CACHE"),
    "SGLANG_RBLN_COMPILE_ONLY": _flag("SGLANG_RBLN_COMPILE_ONLY"),
    "SGLANG_RBLN_COMPILE_STRICT": _flag("SGLANG_RBLN_COMPILE_STRICT", default=True),
    "SGLANG_RBLN_STATIC_OUTPUT": _flag("SGLANG_RBLN_STATIC_OUTPUT"),
    "SGLANG_RBLN_NUM_DEVICES": _optional("SGLANG_RBLN_NUM_DEVICES", int),
    "SGLANG_RBLN_COMPILE_VAE": _flag("SGLANG_RBLN_COMPILE_VAE"),
    "SGLANG_RBLN_CHECK_GRAPHS": _flag("SGLANG_RBLN_CHECK_GRAPHS"),
    "SGLANG_RBLN_LOG_GRAPHS": _flag("SGLANG_RBLN_LOG_GRAPHS"),
    "SGLANG_RBLN_DUMP_GRAPHS": _optional("SGLANG_RBLN_DUMP_GRAPHS", Path),
    "SGLANG_RBLN_RUNTIME_REPORTS": _flag("SGLANG_RBLN_RUNTIME_REPORTS"),
    "SGLANG_RBLN_PROFILE_DIR": _optional("SGLANG_RBLN_PROFILE_DIR", Path),
    "SGLANG_RBLN_PROFILE_REQUEST": _int("SGLANG_RBLN_PROFILE_REQUEST", 3),
    "SGLANG_RBLN_VOCODER_ON_NPU": _flag("SGLANG_RBLN_VOCODER_ON_NPU"),
    "SGLANG_RBLN_BLOCKS_PER_REGION": _optional("SGLANG_RBLN_BLOCKS_PER_REGION", int),
    "SGLANG_RBLN_FLUX3_ONE_GRAPH": _flag("SGLANG_RBLN_FLUX3_ONE_GRAPH"),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables)

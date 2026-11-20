# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

DEFAULT_CACHE_DIR = "~/.cache/sglang_rbln"
DEFAULT_PROFILE_REQUEST = 3

if TYPE_CHECKING:
    RBLN_FORCE_NPU_NAME: str | None = None
    SGLANG_RBLN_CACHE_DIR: Path = Path(DEFAULT_CACHE_DIR).expanduser()
    SGLANG_RBLN_DISABLE_COMPILE_CACHE: bool = False
    SGLANG_RBLN_COMPILE_ONLY: bool = False
    SGLANG_RBLN_NUM_DEVICES: int | None = None
    SGLANG_RBLN_COMPONENT_DEVICES: dict[str, tuple[int, ...]] = {}
    SGLANG_RBLN_LOG_TIMELINE: bool = False
    SGLANG_RBLN_PROFILE_DIR: Path | None = None
    SGLANG_RBLN_PROFILE_REQUEST: int = DEFAULT_PROFILE_REQUEST
    SGLANG_RBLN_BLOCKS_PER_REGION: int | None = None
    SGLANG_RBLN_PROMPT_ENHANCER: bool = False
    SGLANG_RBLN_PROMPT_ENHANCER_SYSTEM_PROMPT: Path | None = None
    SGLANG_RBLN_LTX2_DETAILING_LORA: Path | None = None


def _flag(name: str) -> Callable[[], bool]:
    return lambda: os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _optional(name: str, parse: Callable[[str], Any] = str) -> Callable[[], Any]:
    return lambda: parse(value) if (value := os.environ.get(name)) else None


def _component_devices(value: str) -> dict[str, tuple[int, ...]]:
    """`text_encoder=4,5;vae=8` -> `{"text_encoder": (4, 5), "vae": (8,)}`."""
    return {
        name.strip(): tuple(int(index) for index in devices.split(","))
        for entry in value.split(";")
        if entry.strip()
        for name, devices in [entry.split("=")]
    }


environment_variables: dict[str, Callable[[], Any]] = {
    "RBLN_FORCE_NPU_NAME": _optional("RBLN_FORCE_NPU_NAME"),
    "SGLANG_RBLN_CACHE_DIR": lambda: Path(
        os.environ.get("SGLANG_RBLN_CACHE_DIR") or DEFAULT_CACHE_DIR
    ).expanduser(),
    "SGLANG_RBLN_DISABLE_COMPILE_CACHE": _flag("SGLANG_RBLN_DISABLE_COMPILE_CACHE"),
    "SGLANG_RBLN_COMPILE_ONLY": _flag("SGLANG_RBLN_COMPILE_ONLY"),
    "SGLANG_RBLN_NUM_DEVICES": _optional("SGLANG_RBLN_NUM_DEVICES", int),
    "SGLANG_RBLN_COMPONENT_DEVICES": lambda: _component_devices(
        os.environ.get("SGLANG_RBLN_COMPONENT_DEVICES", "")
    ),
    "SGLANG_RBLN_LOG_TIMELINE": _flag("SGLANG_RBLN_LOG_TIMELINE"),
    "SGLANG_RBLN_PROFILE_DIR": _optional("SGLANG_RBLN_PROFILE_DIR", Path),
    "SGLANG_RBLN_PROFILE_REQUEST": lambda: int(
        os.environ.get("SGLANG_RBLN_PROFILE_REQUEST") or DEFAULT_PROFILE_REQUEST
    ),
    "SGLANG_RBLN_BLOCKS_PER_REGION": _optional("SGLANG_RBLN_BLOCKS_PER_REGION", int),
    "SGLANG_RBLN_PROMPT_ENHANCER": _flag("SGLANG_RBLN_PROMPT_ENHANCER"),
    "SGLANG_RBLN_PROMPT_ENHANCER_SYSTEM_PROMPT": _optional(
        "SGLANG_RBLN_PROMPT_ENHANCER_SYSTEM_PROMPT", Path
    ),
    "SGLANG_RBLN_LTX2_DETAILING_LORA": _optional(
        "SGLANG_RBLN_LTX2_DETAILING_LORA", Path
    ),
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(environment_variables)

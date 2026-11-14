# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable
from importlib import import_module
from typing import Any

PATCHED_MODULES = ("imageio_ffmpeg._utils", "imageio_ffmpeg._io")
PATCHED_NAME = "_popen_kwargs"


def spawn_in_own_group(popen_kwargs: Callable[..., dict[str, Any]]):
    @functools.wraps(popen_kwargs)
    def wrapper(*args, **kwargs) -> dict[str, Any]:
        popen = popen_kwargs(*args, **kwargs)
        if popen.get("preexec_fn") is os.setpgrp:
            popen["preexec_fn"] = None
            popen["process_group"] = 0
        return popen

    wrapper.spawns_in_own_group = True
    return wrapper


def register() -> tuple[str, ...]:
    if sys.version_info < (3, 11) or sys.platform.startswith("win"):
        return ()
    try:
        modules = [import_module(name) for name in PATCHED_MODULES]
    except ImportError:
        return ()
    for module in modules:
        original = getattr(module, PATCHED_NAME)
        if not getattr(original, "spawns_in_own_group", False):
            setattr(module, PATCHED_NAME, spawn_in_own_group(original))
    return PATCHED_MODULES

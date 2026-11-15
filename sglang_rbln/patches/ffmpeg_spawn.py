# SPDX-License-Identifier: Apache-2.0
"""Spawn imageio-ffmpeg's ffmpeg in its own process group without ``preexec_fn``.

imageio-ffmpeg passes ``preexec_fn=os.setpgrp``, which makes CPython fork() the
whole worker; a worker holding many NPU runtimes spent seconds copying its page
tables per saved video. ``process_group=0`` puts ffmpeg in its own group the same
way and keeps the spawn on the vfork path.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from importlib import import_module
from typing import Any

# _io imports its own reference to _utils._popen_kwargs.
PATCHED_MODULES = ("imageio_ffmpeg._utils", "imageio_ffmpeg._io")
PATCHED_FUNCTION = "_popen_kwargs"


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
    for module in map(import_module, PATCHED_MODULES):
        original = getattr(module, PATCHED_FUNCTION)
        if not getattr(original, "spawns_in_own_group", False):
            setattr(module, PATCHED_FUNCTION, spawn_in_own_group(original))
    return PATCHED_MODULES

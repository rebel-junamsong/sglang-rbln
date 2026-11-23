# SPDX-License-Identifier: Apache-2.0
"""Upstream's autocast, on rbln tensors as well as the platform's device type.

Upstream opens autocast on the platform's device type, which is the host's. The
compiled regions and the eager ops between them take rbln tensors, which only an rbln
autocast casts, so the same autocast opens on both device types.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.patches.torch_rbln import autocast as rbln_autocast

HOOK_TARGET = "sglang.multimodal_gen.runtime.utils.precision.autocast_context"


@contextmanager
def _host_and_rbln_autocast(host, dtype: torch.dtype):
    with (
        host,
        torch.autocast(rbln_autocast.DEVICE_TYPE, dtype=dtype, enabled=host._enabled),
    ):
        yield


def resident_autocast(autocast_context, dtype, disable_autocast, *, enabled=None):
    host = autocast_context(dtype, disable_autocast, enabled=enabled)
    if not isinstance(host, torch.autocast):
        return host
    return _host_and_rbln_autocast(host, dtype)


def register() -> str:
    rbln_autocast.register()
    HookRegistry.register(HOOK_TARGET, resident_autocast, HookType.AROUND)
    return HOOK_TARGET

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import ModuleType

TRITON_NORM_EXPORTS = ("norm_infer", "rms_norm_fn")


def _triton_norm_unavailable(*args, **kwargs):
    raise NotImplementedError("Triton norm kernels are not available on rbln.")


def register(exports_module: ModuleType | None = None) -> list[str]:
    if exports_module is None:
        import sglang.kernels.ops.diffusion as exports_module

    bound_names = []
    for export_name in TRITON_NORM_EXPORTS:
        if export_name in vars(exports_module):
            continue
        setattr(exports_module, export_name, _triton_norm_unavailable)
        bound_names.append(export_name)
    return bound_names

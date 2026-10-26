# SPDX-License-Identifier: Apache-2.0
"""Keep the Triton norm kernels out of the import path on rbln.

Why: ``runtime/layers/layernorm.py`` imports ``norm_infer`` and ``rms_norm_fn``
on every non-CPU platform, and ``norm/norm_triton.py`` autotunes against the
current accelerator at import time, which fails off CUDA.
How: bind both names on ``sglang.kernels.ops.diffusion`` before the layers
import, so its PEP 562 lazy export never loads ``norm_triton``. The CUDA
forwards that call them are never selected on an out-of-tree platform.
Upstream: gate the import in ``layernorm.py`` on ``has_triton()``.
Remove when: sglang no longer imports the Triton norms off CUDA.
"""

from __future__ import annotations

from types import ModuleType

TRITON_NORM_EXPORTS = ("norm_infer", "rms_norm_fn")


def _triton_norm_unavailable(*args, **kwargs):
    raise NotImplementedError("Triton norm kernels are not available on rbln.")


def bind_missing_exports(exports_module: ModuleType | None = None) -> list[str]:
    if exports_module is None:
        import sglang.kernels.ops.diffusion as exports_module

    bound_names = []
    for export_name in TRITON_NORM_EXPORTS:
        if export_name in vars(exports_module):
            continue
        setattr(exports_module, export_name, _triton_norm_unavailable)
        bound_names.append(export_name)
    return bound_names

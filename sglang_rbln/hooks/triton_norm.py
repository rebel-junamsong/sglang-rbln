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

NAMES = ("norm_infer", "rms_norm_fn")


def _unavailable(*args, **kwargs):
    raise NotImplementedError("Triton norm kernels are not available on rbln.")


def seed(module: ModuleType | None = None) -> list[str]:
    if module is None:
        import sglang.kernels.ops.diffusion as module

    bound = []
    for name in NAMES:
        if name in vars(module):
            continue
        setattr(module, name, _unavailable)
        bound.append(name)
    return bound

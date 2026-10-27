# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import pytest

STAGES = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"


@dataclass(frozen=True, slots=True)
class HookDebt:
    hook: str
    module: str
    upstream_pattern: str
    upstream_fix: str


HOOK_DEBTS = [
    HookDebt(
        hook="triton_norm",
        module="sglang.multimodal_gen.runtime.layers.layernorm",
        upstream_pattern=(
            "if not _is_cpu:\n"
            "    from sglang.kernels.ops.diffusion import norm_infer, rms_norm_fn"
        ),
        upstream_fix="gate the Triton norm import on has_triton()",
    ),
    HookDebt(
        hook="vsa_import",
        module="sglang.multimodal_gen.runtime.layers.attention.backends"
        ".video_sparse_attn",
        upstream_pattern=(
            "try:\n    from vsa import video_sparse_attn\nexcept ImportError:"
        ),
        upstream_fix="also catch the CUDA RuntimeError vsa raises at import",
    ),
    HookDebt(
        hook="ltx2_rope",
        module="sglang.multimodal_gen.runtime.models.dits.ltx_2",
        upstream_pattern=(
            "cos_freqs = torch.swapaxes(cos_freq, 1, 2)\n"
            "            sin_freqs = torch.swapaxes(sin_freq, 1, 2)"
        ),
        upstream_fix="return the rotary cos/sin tables contiguous",
    ),
    HookDebt(
        hook="cosmos3_compile",
        module=f"{STAGES}.cosmos3",
        upstream_pattern='compile_kwargs = {\n                "mode": "default",',
        upstream_fix="build the kwargs through resolve_torch_compile_kwargs",
    ),
]


def _upstream_source(module: str) -> str:
    spec = find_spec(module)
    assert spec is not None and spec.origin, f"{module} is not installed"
    return Path(spec.origin).read_text()


@pytest.mark.parametrize("debt", HOOK_DEBTS, ids=lambda debt: debt.hook)
def test_upstream_still_needs_the_hook(debt: HookDebt):
    assert debt.upstream_pattern in _upstream_source(debt.module), (
        f"sglang no longer has the code hooks/{debt.hook} works around "
        f"({debt.upstream_fix}); delete the hook"
    )

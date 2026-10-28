# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import pytest

from sglang_rbln import device

STAGES = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"


@dataclass(frozen=True, slots=True)
class PatchDebt:
    patch: str
    module: str
    upstream_pattern: str
    upstream_fix: str


PATCH_DEBTS = [
    PatchDebt(
        patch="layers/layernorm",
        module="sglang.multimodal_gen.runtime.layers.layernorm",
        upstream_pattern=(
            "if not _is_cpu:\n"
            "    from sglang.kernels.ops.diffusion import norm_infer, rms_norm_fn"
        ),
        upstream_fix="gate the Triton norm import on has_triton()",
    ),
    PatchDebt(
        patch="layers/attention/video_sparse_attn",
        module="sglang.multimodal_gen.runtime.layers.attention.backends"
        ".video_sparse_attn",
        upstream_pattern=(
            "try:\n    from vsa import video_sparse_attn\nexcept ImportError:"
        ),
        upstream_fix="also catch the CUDA RuntimeError vsa raises at import",
    ),
    PatchDebt(
        patch="srt/layers/quantization/fp4_utils",
        module="sglang.srt.layers.quantization.fp4_utils",
        upstream_pattern=(
            "    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize"
        ),
        upstream_fix="catch the AssertionError flashinfer raises without CUDA",
    ),
    PatchDebt(
        patch="models/ltx_2",
        module="sglang.multimodal_gen.runtime.models.dits.ltx_2",
        upstream_pattern=(
            "cos_freqs = torch.swapaxes(cos_freq, 1, 2)\n"
            "            sin_freqs = torch.swapaxes(sin_freq, 1, 2)"
        ),
        upstream_fix="return the rotary cos/sin tables contiguous",
    ),
    PatchDebt(
        patch="stages/cosmos3",
        module=f"{STAGES}.cosmos3",
        upstream_pattern='compile_kwargs = {\n                "mode": "default",',
        upstream_fix="build the kwargs through resolve_torch_compile_kwargs",
    ),
    PatchDebt(
        patch="stages/cosmos3 (decode)",
        module=f"{STAGES}.cosmos3",
        upstream_pattern="        video = self.vae.decode(latents)",
        upstream_fix="let the platform choose the VAE decode device",
    ),
    PatchDebt(
        patch="models/cosmos3video",
        module="sglang.multimodal_gen.runtime.models.dits.cosmos3video",
        upstream_pattern="        t_scaled = t * self.timestep_scale",
        upstream_fix="let the platform run float32 timestep embeddings on the host",
    ),
    PatchDebt(
        patch="stages/flux3_action",
        module=f"{STAGES}.flux3_action",
        upstream_pattern="class Flux3ActionDenoisingStage(PipelineStage):",
        upstream_fix="compile the DiT blocks like DenoisingStage does",
    ),
    pytest.param(
        PatchDebt(
            patch="torch_rbln/ops_utils",
            module="torch_rbln._internal.ops_utils",
            upstream_pattern=(
                "promoted_dtype = torch.promote_types(promoted_dtype, tensor.dtype)"
            ),
            upstream_fix="promote like torch.result_type (zero-dim tensors do not"
            " raise the result dtype within a category)",
        ),
        id="torch_rbln/ops_utils",
        marks=pytest.mark.xfail(
            not device.torch_rbln_available(), reason="torch-rbln is not installed"
        ),
    ),
]


def _upstream_source(module: str) -> str:
    spec = find_spec(module)
    assert spec is not None and spec.origin, f"{module} is not installed"
    return Path(spec.origin).read_text()


@pytest.mark.parametrize(
    "debt",
    PATCH_DEBTS,
    ids=lambda debt: debt.patch if isinstance(debt, PatchDebt) else None,
)
def test_upstream_still_needs_the_patch(debt: PatchDebt):
    assert debt.upstream_pattern in _upstream_source(debt.module), (
        f"sglang no longer has the code patches/{debt.patch} works around "
        f"({debt.upstream_fix}); delete the patch"
    )

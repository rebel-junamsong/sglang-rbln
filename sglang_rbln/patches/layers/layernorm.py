# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import ModuleType

from torch import nn

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


def prebuild_qk_norm_rope(head_dim: int, *, is_neox: bool = True) -> nn.Module:
    from sglang.multimodal_gen.runtime.layers import layernorm
    from sglang.multimodal_gen.runtime.layers.rotary_embedding import RotaryEmbedding

    key = (head_dim, is_neox)
    if (rotary_emb := layernorm._QK_NORM_ROPE_DICT.get(key)) is None:
        rotary_emb = layernorm._QK_NORM_ROPE_DICT[key] = RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            use_precomputed_cache=False,
            is_neox_style=is_neox,
        )
    return rotary_emb

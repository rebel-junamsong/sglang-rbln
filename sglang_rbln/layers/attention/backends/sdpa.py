# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    SDPABackend,
    SDPAImpl,
)


def expand_kv_heads(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    query_heads, kv_heads = query.shape[2], key.shape[2]
    if query_heads == kv_heads:
        return key, value
    if query_heads % kv_heads:
        raise ValueError(
            f"query heads ({query_heads}) must be a multiple of kv heads ({kv_heads})"
        )
    repeats = query_heads // kv_heads
    return key.repeat_interleave(repeats, dim=2), value.repeat_interleave(
        repeats, dim=2
    )


class RBLNSdpaImpl(SDPAImpl):
    def forward(self, query, key, value, attn_metadata):
        key, value = expand_kv_heads(query, key, value)
        return super().forward(query, key, value, attn_metadata)


class RBLNSdpaBackend(SDPABackend):
    @staticmethod
    def get_impl_cls():
        return RBLNSdpaImpl

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

MODULE = "sglang.multimodal_gen.runtime.models.dits.cosmos3video"
HOOK_TARGET = f"{MODULE}.Cosmos3TimestepEmbedder.forward"
QK_NORM_ROPE_HOOK_TARGET = f"{MODULE}._apply_qwen3_qk_norm_rope"
TEXT_KV_HOOK_TARGET = f"{MODULE}.Cosmos3UndDecoderLayer.forward"
HOST_STATE = "_rbln_host_state"


def embed_timesteps_on_host(forward, embedder: nn.Module, t: torch.Tensor):
    device = next(embedder.parameters()).device
    if device.type != "rbln":
        return forward(embedder, t)
    host_state = getattr(embedder, HOST_STATE, None)
    if host_state is None:
        host_state = {
            name: tensor.cpu()
            for name, tensor in (
                *embedder.named_parameters(),
                *embedder.named_buffers(),
            )
        }
        object.__setattr__(embedder, HOST_STATE, host_state)
    with torch.no_grad():
        embedding = torch.func.functional_call(embedder, host_state, (t.cpu(),))
    return embedding.to(device)


def split_qk_norm_rope(
    q, k, q_norm, k_norm, head_dim, cos_sin_cache, positions, **kwargs
):
    from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
        _apply_qwen3_qk_norm_rope_split,
    )

    return _apply_qwen3_qk_norm_rope_split(
        q, k, q_norm, k_norm, head_dim, cos_sin_cache
    )


def contiguous_text_kv(forward, layer: nn.Module, *args, **kwargs):
    hidden_states, key, value = forward(layer, *args, **kwargs)
    return hidden_states, key.contiguous(), value.contiguous()


def register() -> tuple[str, str, str]:
    HookRegistry.register(HOOK_TARGET, embed_timesteps_on_host, HookType.AROUND)
    HookRegistry.register(
        QK_NORM_ROPE_HOOK_TARGET, split_qk_norm_rope, HookType.REPLACE
    )
    HookRegistry.register(TEXT_KV_HOOK_TARGET, contiguous_text_kv, HookType.AROUND)
    return HOOK_TARGET, QK_NORM_ROPE_HOOK_TARGET, TEXT_KV_HOOK_TARGET


def run_on_host(module: nn.Module) -> bool:
    if next(module.parameters()).device.type != "rbln":
        return False
    host_state = {
        name: tensor.cpu()
        for name, tensor in (*module.named_parameters(), *module.named_buffers())
    }
    device_forward = module.forward
    reentered = False

    def forward(*args):
        nonlocal reentered
        if reentered:
            return device_forward(*args)
        reentered = True
        try:
            with torch.no_grad():
                out = torch.func.functional_call(
                    module,
                    host_state,
                    tuple(
                        arg.cpu() if isinstance(arg, torch.Tensor) else arg
                        for arg in args
                    ),
                )
        finally:
            reentered = False
        return torch.utils._pytree.tree_map_only(
            torch.Tensor, lambda tensor: tensor.to(args[0].device), out
        )

    module.forward = forward
    return True

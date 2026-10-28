# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.models.dits.cosmos3video"
    ".Cosmos3TimestepEmbedder.forward"
)
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


def register() -> str:
    HookRegistry.register(HOOK_TARGET, embed_timesteps_on_host, HookType.AROUND)
    return HOOK_TARGET


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
        return out.to(args[0].device)

    module.forward = forward
    return True

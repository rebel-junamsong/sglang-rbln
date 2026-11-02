# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from collections import OrderedDict

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

MODELS = "sglang.multimodal_gen.runtime.models"
HOOK_TARGETS = (
    f"{MODELS}.dits.ltx_2.LTX2AudioVideoRotaryPosEmbed.forward",
    f"{MODELS}.adapter.ltx_2_connector.LTX2RotaryPosEmbed1d.forward",
)
FUSED_ADA_VALUES = f"{MODELS}.dits.ltx_2._ltx2_try_fused_ada_values9"
VAE = f"{MODELS}.vaes.ltx_2_vae.AutoencoderKLLTX2Video"


ROTARY_TABLES = "_rbln_rotary_tables"
ROTARY_TABLES_KEPT = 4


def contiguous_rotary_tables(result, *args, **kwargs):
    cos, sin = result
    return cos.contiguous(), sin.contiguous()


def _argument_key(value):
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu().contiguous().reshape(-1)
        digest = hashlib.sha256(data.view(torch.uint8).numpy() if data.numel() else b"")
        return (
            "tensor",
            tuple(value.shape),
            value.dtype,
            str(value.device),
            digest.digest(),
        )
    return value


def cached_rotary_tables(forward, rope: torch.nn.Module, *args, **kwargs):
    """The rotary tables of these positions, computed once per position set.

    A denoise step asks for the same positions as the step before it, and building
    the tables for every video token on the host cost about a tenth of a second per
    step. A table depends only on the positions and the output dtype and device.
    """
    key = (
        tuple(map(_argument_key, args)),
        tuple(sorted((name, _argument_key(value)) for name, value in kwargs.items())),
    )
    tables = rope.__dict__.setdefault(ROTARY_TABLES, OrderedDict())
    if key in tables:
        tables.move_to_end(key)
        return tables[key]
    tables[key] = contiguous_rotary_tables(forward(rope, *args, **kwargs))
    while len(tables) > ROTARY_TABLES_KEPT:
        tables.popitem(last=False)
    return tables[key]


def without_triton_ada_values(*args, **kwargs) -> None:
    return None


def _blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
    """Upstream's per-slice blend as whole-slab ops, rounding at the same steps."""
    extent = min(a.shape[dim], b.shape[dim], extent)
    if extent <= 0:
        return b
    ramp = torch.arange(extent, dtype=torch.float64) / extent
    shape = [1] * b.dim()
    shape[dim] = extent
    head = b.narrow(dim, 0, extent)
    tail = torch.mul(
        a.narrow(dim, a.shape[dim] - extent, extent),
        (1 - ramp).to(torch.float32).view(shape),
        out=torch.empty_like(head),
    )
    head.mul_(ramp.to(torch.float32).view(shape)).add_(tail)
    return b


def blend_v(self, a, b, blend_extent):
    return _blend(a, b, blend_extent, 3)


def blend_h(self, a, b, blend_extent):
    return _blend(a, b, blend_extent, 4)


def blend_t(self, a, b, blend_extent):
    return _blend(a, b, blend_extent, -3)


BLENDS = {
    f"{VAE}.blend_v": blend_v,
    f"{VAE}.blend_h": blend_h,
    f"{VAE}.blend_t": blend_t,
}


def register() -> tuple[str, ...]:
    for target in HOOK_TARGETS:
        HookRegistry.register(target, cached_rotary_tables, HookType.AROUND)
    HookRegistry.register(FUSED_ADA_VALUES, without_triton_ada_values, HookType.REPLACE)
    for target, blend in BLENDS.items():
        HookRegistry.register(target, blend, HookType.REPLACE)
    return (*HOOK_TARGETS, FUSED_ADA_VALUES, *BLENDS)

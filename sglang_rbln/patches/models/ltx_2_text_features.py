# SPDX-License-Identifier: Apache-2.0
"""Keep the LTX-2 text features on the NPUs between Gemma and the connectors.

Upstream packs Gemma's 49 hidden states into one 188160-channel feature per token
(`_gemma_postprocess_func`), rescales it for each modality (`_rescale_v2_features`), and
replaces the padded tokens with learned registers in each connector transformer. All
three run eagerly, so with resident regions each brought its device tensors to the
host, and the ~385 MB feature was uploaded dense to the projections and laid out again
on every call. Here each runs, unchanged, as a module compiled for the connectors' NPUs
whenever its inputs live on the NPUs; on the host, upstream runs as it is.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

from sglang_rbln import envs
from sglang_rbln.compilation import placement, residency

CONFIGS = "sglang.multimodal_gen.configs.pipeline_configs.ltx_2"
CONNECTOR = "sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector"
POSTPROCESS = f"{CONFIGS}._gemma_postprocess_func"
RESCALE = f"{CONNECTOR}.LTX2TextConnectors._rescale_v2_features"
CONNECTOR_FORWARD = f"{CONNECTOR}.LTX2ConnectorTransformer1d.forward"
COMPILED: dict[tuple, nn.Module] = {}


def _connector_devices() -> tuple[int, ...] | None:
    return envs.SGLANG_RBLN_COMPONENT_DEVICES.get("connectors")


def _signature(*tensors: torch.Tensor) -> tuple:
    return tuple((tuple(t.shape), t.dtype) for t in tensors)


class PackTextFeatures(nn.Module):
    """Upstream's packing of the text encoder's hidden states, as one graph."""

    def __init__(self, postprocess, pipeline_config) -> None:
        super().__init__()
        self.postprocess = postprocess
        self.pipeline_config = pipeline_config

    def forward(self, attention_mask: torch.Tensor, *hidden_states: torch.Tensor):
        return self.postprocess(
            SimpleNamespace(hidden_states=hidden_states),
            {"attention_mask": attention_mask},
            self.pipeline_config,
        )


class Rescale(nn.Module):
    """Upstream's per-modality rescale of the packed features."""

    def __init__(self, rescale, target_dim: int, source_dim: int) -> None:
        super().__init__()
        self.rescale = rescale
        self.dims = (target_dim, source_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.rescale(features, *self.dims)


class RegisterStep(nn.Module):
    """Upstream's register replacement over a fixed length.

    `src` moves the valid tokens to the front, in their order. Upstream zero-pads behind
    them and blends with the flipped mask; `keep` is that mask where a valid token sits,
    so the rows past them add nothing, as upstream's zeros do.
    """

    def __init__(self, learnable_registers: nn.Parameter, repeats: int) -> None:
        super().__init__()
        # The connector's own parameter, a weight here, so its tiling folds at compile.
        self.learnable_registers = learnable_registers
        self.repeats = repeats

    def forward(self, hidden_states, src, keep, flipped):
        registers = torch.tile(self.learnable_registers, (self.repeats, 1))
        index = src.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        padded = torch.gather(hidden_states, 1, index)
        return keep * padded + (1 - flipped) * registers


def gemma_postprocess(postprocess, outputs, text_inputs, pipeline_config=None):
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        return postprocess(outputs, text_inputs, pipeline_config)
    mask = text_inputs["attention_mask"]
    key = ("pack", id(pipeline_config), _signature(mask, *hidden_states))
    module = residency.resident_module(
        COMPILED,
        key,
        lambda: PackTextFeatures(postprocess, pipeline_config),
        hidden_states,
        _connector_devices(),
    )
    return module(mask, *hidden_states)


def rescale_v2_features(rescale, x, target_dim, source_dim):
    key = ("rescale", target_dim, source_dim, _signature(x))
    module = residency.resident_module(
        COMPILED,
        key,
        lambda: Rescale(rescale, target_dim, source_dim),
        (x,),
        _connector_devices(),
    )
    return module(x)


def connector_forward(
    forward,
    self,
    hidden_states,
    attention_mask=None,
    attn_mask_binarize_threshold=-9000.0,
):
    """Upstream's forward, with its register replacement over a fixed length when the
    hidden states live on the NPUs."""
    if self.learnable_registers is None or not residency._is_resident(hidden_states):
        return forward(
            self, hidden_states, attention_mask, attn_mask_binarize_threshold
        )
    batch_size, seq_len, _ = hidden_states.shape
    if seq_len % self.num_learnable_registers != 0:
        raise ValueError(
            f"The `hidden_states` sequence length {hidden_states.shape[1]} should be "
            "divisible by the number of learnable registers "
            f"{self.num_learnable_registers}"
        )

    # The index is host math on the host mask: the valid tokens first, in their order.
    binary_attn_mask = (attention_mask.cpu() >= attn_mask_binarize_threshold).int()
    if binary_attn_mask.ndim == 4:
        binary_attn_mask = binary_attn_mask.squeeze(1).squeeze(1)
    src = torch.argsort(1 - binary_attn_mask, dim=1, stable=True).to(torch.int32)
    flipped = torch.flip(binary_attn_mask, dims=[1]).unsqueeze(-1)
    valid = torch.arange(seq_len) < binary_attn_mask.sum(dim=1, keepdim=True)
    keep = (flipped * valid.unsqueeze(-1)).to(hidden_states.dtype)
    flipped = flipped.to(hidden_states.dtype)
    repeats = seq_len // self.num_learnable_registers
    key = ("registers", id(self), repeats, _signature(hidden_states))
    step = residency.resident_module(
        COMPILED,
        key,
        lambda: RegisterStep(self.learnable_registers, repeats),
        (hidden_states,),
        placement.devices_of(self),
    )
    hidden_states = step(hidden_states, src, keep, flipped)
    attention_mask = torch.zeros_like(attention_mask)

    # On the host, where upstream's eager register step left the hidden states.
    rotary_emb = self.rope(
        batch_size, seq_len, device=torch.device("cpu"), dtype=hidden_states.dtype
    )
    for block in self.transformer_blocks:
        hidden_states = block(
            hidden_states, attention_mask=attention_mask, rotary_emb=rotary_emb
        )
    hidden_states = self.norm_out(hidden_states)
    return hidden_states, attention_mask


TARGETS = {
    POSTPROCESS: gemma_postprocess,
    RESCALE: rescale_v2_features,
    CONNECTOR_FORWARD: connector_forward,
}


def register() -> tuple[str, ...]:
    for target, hook in TARGETS.items():
        HookRegistry.register(target, hook, HookType.AROUND)
    return tuple(TARGETS)

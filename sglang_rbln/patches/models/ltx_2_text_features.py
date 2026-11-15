# SPDX-License-Identifier: Apache-2.0
"""Keep the LTX-2 text features on the NPUs between Gemma and the connectors.

Upstream packs Gemma's hidden states (`_gemma_postprocess_func`) and rescales them
(`_rescale_v2_features`), and its connector transformer replaces the padded tokens
with learned registers, eagerly, on the device the hidden states live on. Here the
packed features are returned on the connectors' NPUs rather than the platform
device, the host, and the packing itself runs as one compiled graph of upstream's
code, so the 368 MiB stack of Gemma's 49 layers is not taken to the host and back for
each of its eager ops.
"""

from __future__ import annotations

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

from sglang_rbln.compilation import placement, residency

ENCODE_TEXT = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding."
    "TextEncodingStage.encode_text"
)
PACK = "sglang.multimodal_gen.configs.pipeline_configs.ltx_2.pack_text_embeds_v2"
_pack_graphs: dict[tuple, nn.Module] = {}


def _connector_devices() -> tuple[int, ...] | None:
    return placement.component_devices("connectors")


def encode_text(forward, self, text, server_args, *args, **kwargs):
    """Upstream moves the packed features to the platform device, the host; with the
    connectors placed they stay on the connectors' NPUs, where they were packed."""
    devices = _connector_devices()
    if devices and len(args) < 4 and kwargs.get("device") is None:
        kwargs["device"] = torch.device("rbln", residency.logical_device_on(devices))
    return forward(self, text, server_args, *args, **kwargs)


class PackTextFeatures(nn.Module):
    """Upstream's `pack_text_embeds_v2`, unchanged, as one graph."""

    def __init__(self, pack) -> None:
        super().__init__()
        self.pack = pack

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        return self.pack(hidden_states, attention_mask)


def pack_text_features(pack, hidden_states, attention_mask, *args, **kwargs):
    if args or kwargs or not residency._is_resident(hidden_states):
        return pack(hidden_states, attention_mask, *args, **kwargs)
    module = residency.resident_module(
        _pack_graphs,
        (tuple(hidden_states.shape), hidden_states.dtype, attention_mask.dtype),
        lambda: PackTextFeatures(pack),
        (hidden_states, attention_mask),
        residency.npus_of(hidden_states.device),
    )
    return module(hidden_states, attention_mask)


TARGETS = {
    ENCODE_TEXT: encode_text,
    PACK: pack_text_features,
}


def register() -> tuple[str, ...]:
    for target, hook in TARGETS.items():
        HookRegistry.register(target, hook, HookType.AROUND)
    return tuple(TARGETS)

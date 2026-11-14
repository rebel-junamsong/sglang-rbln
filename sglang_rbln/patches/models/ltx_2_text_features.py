# SPDX-License-Identifier: Apache-2.0
"""Keep the LTX-2 text features on the NPUs between Gemma and the connectors.

Upstream packs Gemma's hidden states (`_gemma_postprocess_func`) and rescales them
(`_rescale_v2_features`), and its connector transformer replaces the padded tokens
with learned registers, eagerly, on the device the hidden states live on. Here the
packed features are returned on the connectors' NPUs rather than the platform
device, the host.
"""

from __future__ import annotations

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.compilation import placement, residency

ENCODE_TEXT = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding."
    "TextEncodingStage.encode_text"
)


def _connector_devices() -> tuple[int, ...] | None:
    return placement.component_devices("connectors")


def encode_text(forward, self, text, server_args, *args, **kwargs):
    """Upstream moves the packed features to the platform device, the host; with the
    connectors placed they stay on the connectors' NPUs, where they were packed."""
    devices = _connector_devices()
    if devices and len(args) < 4 and kwargs.get("device") is None:
        kwargs["device"] = torch.device("rbln", residency.logical_device_on(devices))
    return forward(self, text, server_args, *args, **kwargs)


TARGETS = {
    ENCODE_TEXT: encode_text,
}


def register() -> tuple[str, ...]:
    for target, hook in TARGETS.items():
        HookRegistry.register(target, hook, HookType.AROUND)
    return tuple(TARGETS)

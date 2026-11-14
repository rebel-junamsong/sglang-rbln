# SPDX-License-Identifier: Apache-2.0
"""Keep the LTX-2 text features on the NPUs between Gemma and the connectors.

Upstream packs Gemma's hidden states (`_gemma_postprocess_func`) and rescales them
(`_rescale_v2_features`), and its connector transformer replaces the padded tokens
with learned registers, eagerly, on the device the hidden states live on. Here the
packed features are returned on the connectors' NPUs rather than the platform
device, the host, and their NaN check runs there too.
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
VALIDATE = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.validators."
    "StageValidators.list_of_tensors_with_min_dims"
)
COMPILED: dict[tuple, nn.Module] = {}


def _connector_devices() -> tuple[int, ...] | None:
    return placement.component_devices("connectors")


def _signature(*tensors: torch.Tensor) -> tuple:
    return tuple((tuple(t.shape), t.dtype) for t in tensors)


class HasNan(nn.Module):
    """Whether a tensor holds a NaN, the check upstream's stage validators make."""

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.isnan(tensor).any()


def encode_text(forward, self, text, server_args, *args, **kwargs):
    """Upstream moves the packed features to the platform device, the host; with the
    connectors placed they stay on the connectors' NPUs, where they were packed."""
    devices = _connector_devices()
    if devices and len(args) < 4 and kwargs.get("device") is None:
        kwargs["device"] = torch.device("rbln", residency.logical_device_on(devices))
    return forward(self, text, server_args, *args, **kwargs)


def tensors_with_min_dims(validate, value, min_dims):
    """Upstream's check brings each tensor to the host whole to look for a NaN; with
    the text features on the NPUs, the NPUs look and only the answer comes back."""
    resident = isinstance(value, list) and all(map(residency._is_resident, value))
    if not (value and resident):
        return validate(value, min_dims)
    for tensor in value:
        if tensor.dim() < min_dims:
            return False
        module = residency.resident_module(
            COMPILED,
            ("nan", tensor.device.index, _signature(tensor)),
            HasNan,
            (tensor,),
            residency.npus_of(tensor.device),
        )
        if module(tensor).item():
            return False
    return True


TARGETS = {
    ENCODE_TEXT: encode_text,
    VALIDATE: tensors_with_min_dims,
}


def register() -> tuple[str, ...]:
    for target, hook in TARGETS.items():
        HookRegistry.register(target, hook, HookType.AROUND)
    return tuple(TARGETS)

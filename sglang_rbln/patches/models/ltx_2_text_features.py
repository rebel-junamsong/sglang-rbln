# SPDX-License-Identifier: Apache-2.0
"""Keep the LTX-2 text features on the NPUs between Gemma and the connectors.

Upstream packs Gemma's hidden states (`_gemma_postprocess_func`) and rescales them
(`_rescale_v2_features`), and its connector transformer replaces the padded tokens
with learned registers, eagerly, on the device the hidden states live on. Here the
packed features are returned on the connectors' NPUs rather than the platform
device, the host, and upstream's postprocessing itself (stacking the 49 hidden
states, then packing them) runs as one compiled graph of upstream's code on the NPUs
that hold Gemma's outputs: eager, each op would gather those outputs to one NPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

from sglang_rbln.compilation import placement, residency

ENCODE_TEXT = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding."
    "TextEncodingStage.encode_text"
)
POSTPROCESS = (
    "sglang.multimodal_gen.configs.pipeline_configs.ltx_2._gemma_postprocess_func"
)
# encode_text(text, server_args, encoder_index, return_attention_mask, return_type,
# device, ...): the index of `device` among the arguments after server_args.
DEVICE_POSITION = 3
_postprocess_graphs: dict[tuple, nn.Module] = {}


def encode_text(forward, self, text, server_args, *args, **kwargs):
    """Upstream moves the packed features to the platform device, the host; with the
    connectors placed they stay on the connectors' NPUs, where they were packed."""
    devices = placement.component_devices("connectors")
    if devices and len(args) <= DEVICE_POSITION and kwargs.get("device") is None:
        kwargs["device"] = torch.device("rbln", residency.logical_device_on(devices))
    return forward(self, text, server_args, *args, **kwargs)


class PostprocessTextFeatures(nn.Module):
    """Upstream's `_gemma_postprocess_func`, unchanged, as one graph over the encoder's
    hidden states."""

    def __init__(self, postprocess, pipeline_config) -> None:
        super().__init__()
        self.postprocess = postprocess
        self.pipeline_config = pipeline_config

    def forward(self, attention_mask: torch.Tensor, *hidden_states: torch.Tensor):
        outputs = SimpleNamespace(hidden_states=hidden_states)
        text_inputs = {"attention_mask": attention_mask}
        return self.postprocess(outputs, text_inputs, self.pipeline_config)


def postprocess_text_features(
    postprocess, outputs, text_inputs, pipeline_config=None, *args, **kwargs
):
    hidden_states = getattr(outputs, "hidden_states", None)
    resident = [state for state in hidden_states or () if residency._is_resident(state)]
    if args or kwargs or not resident:
        return postprocess(outputs, text_inputs, pipeline_config, *args, **kwargs)
    attention_mask = text_inputs["attention_mask"]
    module = residency.resident_module(
        _postprocess_graphs,
        (
            tuple((tuple(state.shape), state.dtype) for state in hidden_states),
            tuple(attention_mask.shape),
            attention_mask.dtype,
        ),
        lambda: PostprocessTextFeatures(postprocess, pipeline_config),
        resident,
        residency.npus_of(resident[0].device),
    )
    return module(attention_mask, *hidden_states)


HOOKS = {
    ENCODE_TEXT: encode_text,
    POSTPROCESS: postprocess_text_features,
}


def register() -> tuple[str, ...]:
    for target, hook in HOOKS.items():
        HookRegistry.register(target, hook, HookType.AROUND)
    return tuple(HOOKS)

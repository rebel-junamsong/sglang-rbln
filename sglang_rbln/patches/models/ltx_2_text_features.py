# SPDX-License-Identifier: Apache-2.0
"""Keep the LTX-2 text features on the NPUs between Gemma and the connectors.

Upstream packs Gemma's hidden states (`_gemma_postprocess_func`) and rescales them
(`_rescale_v2_features`), and its connector transformer replaces the padded tokens
with learned registers, eagerly, on the device the hidden states live on. Here the
packed features are returned on the connectors' NPUs rather than the platform
device, the host, and upstream's postprocessing itself (stacking the 49 hidden
states, then packing them) runs as one compiled graph of upstream's code on the NPUs
that hold Gemma's outputs: eager, each op would gather those outputs to one NPU.
Each connector transformer runs as one graph on the connectors' NPUs, its register
replacement included.
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
CONNECTOR_TRANSFORMER_FORWARD = (
    "sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector."
    "LTX2ConnectorTransformer1d.forward"
)
CONNECTOR_GRAPHS = "_rbln_connector_graphs"
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


class ConnectorTransformer(nn.Module):
    """Upstream's `LTX2ConnectorTransformer1d.forward` from its learned registers to
    its output norm, as one graph.

    Upstream takes the valid tokens out with a boolean mask, a length that depends on
    the data, which a static-shape graph cannot hold. Here `src` moves them to the
    front in their order over the fixed length; `keep` is upstream's flipped mask
    where a valid token sits, so the rows behind them add nothing, as upstream's zero
    padding does.
    """

    def __init__(self, connector: nn.Module, repeats: int) -> None:
        super().__init__()
        self.connector = connector
        self.repeats = repeats

    def forward(self, hidden_states, src, keep, flipped, attention_mask, *rotary_emb):
        connector = self.connector
        registers = torch.tile(connector.learnable_registers, (self.repeats, 1))
        index = src.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        padded = torch.gather(hidden_states, 1, index)
        hidden_states = keep * padded + (1 - flipped) * registers
        for block in connector.transformer_blocks:
            hidden_states = block(
                hidden_states, attention_mask=attention_mask, rotary_emb=rotary_emb
            )
        return connector.norm_out(hidden_states)


def connector_transformer_forward(
    forward,
    self,
    hidden_states,
    attention_mask=None,
    attn_mask_binarize_threshold=-9000.0,
):
    """Upstream's forward as one graph where the hidden states live on the NPUs."""
    batch_size, seq_len, _ = hidden_states.shape
    if (
        self.learnable_registers is None
        or not residency._is_resident(hidden_states)
        or seq_len % self.num_learnable_registers  # upstream raises
    ):
        return forward(
            self, hidden_states, attention_mask, attn_mask_binarize_threshold
        )

    # The gather index is host math on the host mask: the valid tokens first, in order.
    binary_attn_mask = (attention_mask.cpu() >= attn_mask_binarize_threshold).int()
    if binary_attn_mask.ndim == 4:
        binary_attn_mask = binary_attn_mask.squeeze(1).squeeze(1)
    src = torch.argsort(1 - binary_attn_mask, dim=1, stable=True).to(torch.int32)
    flipped = torch.flip(binary_attn_mask, dims=[1]).unsqueeze(-1)
    valid = torch.arange(seq_len) < binary_attn_mask.sum(dim=1, keepdim=True)
    keep = (flipped * valid.unsqueeze(-1)).to(hidden_states.dtype)
    flipped = flipped.to(hidden_states.dtype)
    attention_mask = torch.zeros(attention_mask.shape, dtype=attention_mask.dtype)
    rotary_emb = self.rope(
        batch_size, seq_len, device=torch.device("cpu"), dtype=hidden_states.dtype
    )
    repeats = seq_len // self.num_learnable_registers
    module = residency.resident_module(
        self.__dict__.setdefault(CONNECTOR_GRAPHS, {}),
        (tuple(hidden_states.shape), hidden_states.dtype, attention_mask.dtype),
        lambda: ConnectorTransformer(self, repeats),
        (hidden_states,),
        placement.devices_of(self),
    )
    return module(hidden_states, src, keep, flipped, attention_mask, *rotary_emb), (
        attention_mask
    )


HOOKS = {
    ENCODE_TEXT: encode_text,
    POSTPROCESS: postprocess_text_features,
    CONNECTOR_TRANSFORMER_FORWARD: connector_transformer_forward,
}


def register() -> tuple[str, ...]:
    for target, hook in HOOKS.items():
        HookRegistry.register(target, hook, HookType.AROUND)
    return tuple(HOOKS)

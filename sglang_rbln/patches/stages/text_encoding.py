# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import functools
import logging
from collections import OrderedDict
from collections.abc import Iterable

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

from sglang_rbln import envs
from sglang_rbln.compilation import regions
from sglang_rbln.compilation.backend import static_compile_kwargs

logger = logging.getLogger(__name__)

STAGES = "sglang.multimodal_gen.runtime.pipelines_core.stages"
LTX2_STAGES = f"{STAGES}.model_specific_stages.ltx_2"
HOOK_TARGET = f"{STAGES}.text_encoding.TextEncodingStage.forward"
CONNECTOR_HOOK_TARGET = f"{LTX2_STAGES}.text_connector.LTX2TextConnectorStage.forward"
FORWARD_ENCODER_TARGET = (
    f"{STAGES}.text_encoding.TextEncodingStage._forward_text_encoder"
)
# The encoder inputs that feed only its eager prelude: the token embedding lookup,
# the attention masks and the positions.
HOST_INPUTS = ("input_ids", "attention_mask")
COMPILED = "_rbln_regions_compiled"
# transformers' qualname, its typo included.
CAPTURE_HOOK = "install_output_capuring_hook.<locals>.output_capturing_hook"
CONNECTOR_PROJECTIONS = (
    "text_proj_in",
    "video_aggregate_embed",
    "audio_aggregate_embed",
)


# Dynamo skips entry frames inside torch.nn, so a compiled bare nn.Linear traces
# nothing; a subclass whose forward is defined here gives dynamo a frame to trace.
@functools.cache
def _traceable_class(cls: type[nn.Module]) -> type[nn.Module]:
    def forward(self, *args, **kwargs):
        return cls.forward(self, *args, **kwargs)

    return type(cls.__name__, (cls,), {"forward": forward, "__module__": __name__})


def traceable(module: nn.Module) -> nn.Module:
    module.__class__ = _traceable_class(type(module))
    return module


def _is_capture_hook(hook) -> bool:
    return getattr(hook, "__qualname__", "") == CAPTURE_HOOK


def _without_capture_hooks(hooks: dict) -> OrderedDict:
    return OrderedDict(
        (name, hook) for name, hook in hooks.items() if not _is_capture_hook(hook)
    )


def install_capture_hooks(module: nn.Module) -> None:
    """Install transformers' output-capture hooks before the layers compile, so
    `capture_outside` finds them; transformers installs them on the first forward
    that requests an output."""
    from transformers import PreTrainedModel
    from transformers.utils.output_capturing import maybe_install_capturing_hooks

    if isinstance(module, PreTrainedModel):
        maybe_install_capturing_hooks(module)


def has_capture_hook(layer: nn.Module) -> bool:
    return any(map(_is_capture_hook, layer._forward_hooks.values()))


def capture_outside(layer: nn.Module) -> nn.Module:
    """The module to compile for `layer`, without transformers' output-capture hooks.

    The hooks read a ContextVar, which dynamo cannot trace. The submodules' hooks are
    dropped, as nothing here requests their outputs. When `layer` has its own, it
    keeps them and calls a hook-free shallow copy, so they run eagerly around the
    graph.
    """
    for submodule in layer.modules():
        if submodule is not layer:
            submodule._forward_hooks = _without_capture_hooks(submodule._forward_hooks)
    if not has_capture_hook(layer):
        return layer
    region = copy.copy(layer)
    region._forward_hooks = _without_capture_hooks(layer._forward_hooks)
    layer.forward = region.__call__
    return region


def layer_stack(module: nn.Module) -> list[nn.Module]:
    if names := getattr(module, "layer_names", ()):
        return [layer for name in names for layer in module.get_submodule(name)]
    # An encoder without layer_names (Gemma 4 Unified on sglang 0.5.21) names its
    # decoder layer class in _no_split_modules.
    blocks = set(getattr(module, "_no_split_modules", None) or ())
    return [m for m in module.modules() if type(m).__name__ in blocks]


def layer_runs(module: nn.Module) -> list[list[nn.Module]]:
    """The module's layers, in runs of consecutive indices under one parent."""
    stack = {id(layer) for layer in layer_stack(module)}
    named = [(name, m) for name, m in module.named_modules() if id(m) in stack]
    return regions.consecutive_runs(named)


def compile_regions(module: nn.Module, projections: Iterable[str] = ()) -> int:
    """Compile `module`'s layer runs as regions and its named `projections` whole,
    once per module; returns how many graphs that makes."""
    if getattr(module, COMPILED, False):
        return 0
    install_capture_hooks(module)
    runs = layer_runs(module)
    linears = [
        traceable(linear)
        for name in projections
        if (linear := getattr(module, name, None)) is not None
    ]
    graphs = 0
    if runs or linears:
        compile_kwargs = static_compile_kwargs(module)
        blocks_per_region = envs.SGLANG_RBLN_BLOCKS_PER_REGION
        for layers in runs:
            keep_outputs = any(map(has_capture_hook, layers))
            layers = [capture_outside(layer) for layer in layers]
            regions.group_blocks(
                layers, blocks_per_region, compile_kwargs, keep_outputs
            )
            graphs += regions.graph_count(layers, blocks_per_region)
        for linear in linears:
            linear.compile(**compile_kwargs)
        graphs += len(linears)
        logger.info(
            "Compiling %d layers and %d projections of %s as %d regions",
            sum(map(len, runs)),
            len(linears),
            type(module).__name__,
            graphs,
        )
    setattr(module, COMPILED, True)
    return graphs


def compile_text_encoders(forward, stage, batch, server_args):
    if server_args.enable_torch_compile and not server_args.text_encoder_cpu_offload:
        for encoder in stage.text_encoders:
            if isinstance(encoder, nn.Module):
                compile_regions(encoder)
    return forward(stage, batch, server_args)


def compile_connectors(forward, stage, batch, server_args):
    if server_args.enable_torch_compile:
        compile_regions(stage.connectors, CONNECTOR_PROJECTIONS)
    return forward(stage, batch, server_args)


def encode_from_host_inputs(forward_encoder, stage, text_encoder, forward_kwargs):
    """Give the encoder its token ids and mask on the host, as on a CPU platform.

    Upstream takes them to the device the stage returns its features on, the
    connectors' NPUs here. On the device the encoder's eager prelude (the embedding
    lookup in the host table, the mask and position construction) would run there
    op by op; on the host it runs as upstream's CPU path does, and its regions still
    take their inputs to the NPUs.
    """
    forward_kwargs = {
        name: value.cpu() if name in HOST_INPUTS and value is not None else value
        for name, value in forward_kwargs.items()
    }
    return forward_encoder(stage, text_encoder, forward_kwargs)


def register() -> tuple[str, str, str]:
    HookRegistry.register(HOOK_TARGET, compile_text_encoders, HookType.AROUND)
    HookRegistry.register(CONNECTOR_HOOK_TARGET, compile_connectors, HookType.AROUND)
    HookRegistry.register(
        FORWARD_ENCODER_TARGET, encode_from_host_inputs, HookType.AROUND
    )
    return HOOK_TARGET, CONNECTOR_HOOK_TARGET, FORWARD_ENCODER_TARGET

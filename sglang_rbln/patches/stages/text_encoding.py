# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Iterable

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

logger = logging.getLogger(__name__)

STAGES = "sglang.multimodal_gen.runtime.pipelines_core.stages"
LTX2_STAGES = f"{STAGES}.model_specific_stages.ltx_2"
HOOK_TARGET = f"{STAGES}.text_encoding.TextEncodingStage.forward"
CONNECTOR_HOOK_TARGET = f"{LTX2_STAGES}.text_connector.LTX2TextConnectorStage.forward"
COMPILED = "_rbln_layers_compiled"


def compile_layer_stacks(module: nn.Module) -> int:
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    if getattr(module, COMPILED, False):
        return 0
    layers = [
        layer
        for name in getattr(module, "layer_names", ())
        for layer in module.get_submodule(name)
    ]
    if layers:
        compile_kwargs = build_torch_compile_kwargs(mode=None, module=module)
        compile_kwargs["dynamic"] = False
        for layer in layers:
            layer.compile(**compile_kwargs)
        logger.info("Compiling %d %s layers", len(layers), type(module).__name__)
    setattr(module, COMPILED, True)
    return len(layers)


def _compile_all(modules: Iterable[object]) -> None:
    for module in modules:
        if isinstance(module, nn.Module):
            compile_layer_stacks(module)


def compile_text_encoders(forward, stage, batch, server_args):
    if server_args.enable_torch_compile and not server_args.text_encoder_cpu_offload:
        _compile_all(stage.text_encoders)
    return forward(stage, batch, server_args)


def compile_connectors(forward, stage, batch, server_args):
    if server_args.enable_torch_compile:
        _compile_all([stage.connectors])
    return forward(stage, batch, server_args)


def register() -> tuple[str, str]:
    HookRegistry.register(HOOK_TARGET, compile_text_encoders, HookType.AROUND)
    HookRegistry.register(CONNECTOR_HOOK_TARGET, compile_connectors, HookType.AROUND)
    return HOOK_TARGET, CONNECTOR_HOOK_TARGET

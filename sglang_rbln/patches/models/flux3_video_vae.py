# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import envs
from sglang_rbln.compilation.backend import static_compile_kwargs
from sglang_rbln.layers.attention.neighborhood import neighborhood_attention

logger = logging.getLogger(__name__)

VAE = "sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae"
ATTENTION_TARGET = f"{VAE}.neighborhood_attention"
NATTEN_TARGET = f"{VAE}.natten_available"
LOAD_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines.flux3_action.Flux3ActionPipeline._load_vae"
)


def natten_available() -> bool:
    return False


def compile_coders(vae, server_args, config):
    if not server_args.enable_torch_compile:
        return vae
    model = vae.model
    if model.decoder is not None:
        model.decoder.to(torch.bfloat16)
    compile_kwargs = static_compile_kwargs(vae)
    options = compile_kwargs["options"]
    if "device" not in options and (num_devices := options.get("num_devices", 1)) > 1:
        options |= {"num_devices": 1, "device": num_devices - 1}
    for coder in (model.encoder, model.decoder):
        if coder is not None:
            coder.compile(**compile_kwargs)
    logger.info("Compiling the FLUX 3 video VAE: %s", compile_kwargs)
    return vae


def register() -> tuple[str, ...]:
    if not envs.SGLANG_RBLN_COMPILE_VAE:
        return ()
    HookRegistry.register(ATTENTION_TARGET, neighborhood_attention, HookType.REPLACE)
    HookRegistry.register(NATTEN_TARGET, natten_available, HookType.REPLACE)
    HookRegistry.register(LOAD_TARGET, compile_coders, HookType.AFTER)
    return ATTENTION_TARGET, NATTEN_TARGET, LOAD_TARGET

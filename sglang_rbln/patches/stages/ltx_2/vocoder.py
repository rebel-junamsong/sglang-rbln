# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from torch import nn

from sglang_rbln import envs
from sglang_rbln.compilation.backend import on_one_device

logger = logging.getLogger(__name__)

COMPILED = "_rbln_vocoder_compiled"


def _compile_on_one_device(module: nn.Module, what: str) -> None:
    """Compile `module` for one NPU, the last tensor-parallel device."""
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    compile_kwargs = build_torch_compile_kwargs(mode=None, module=module)
    compile_kwargs["dynamic"] = False
    if num_devices := compile_kwargs["options"].get("num_devices"):
        compile_kwargs["options"] = on_one_device(
            compile_kwargs["options"], num_devices - 1, module
        )
    module.compile(**compile_kwargs)
    setattr(module, COMPILED, True)
    logger.info("Compiling the LTX-2 %s: %s", what, compile_kwargs)


def compile_vocoder(forward, stage, batch, server_args):
    vocoder = stage.vocoder
    if (
        envs.SGLANG_RBLN_VOCODER_ON_NPU
        and server_args.enable_torch_compile
        and not getattr(vocoder, COMPILED, False)
    ):
        _compile_on_one_device(vocoder, "vocoder")
    return forward(stage, batch, server_args)


def compile_audio_vae(forward, stage, batch, server_args):
    # `decode` is `self.decoder(z)` (per sample with slicing), so the decoder is
    # the whole latent -> mel graph.
    if envs.SGLANG_RBLN_COMPILE_AUDIO_VAE and server_args.enable_torch_compile:
        audio_vae = getattr(stage, "audio_vae", None)
        decoder = getattr(audio_vae, "decoder", None)
        if decoder is not None and not getattr(decoder, COMPILED, False):
            _compile_on_one_device(decoder, "audio VAE decoder")
    return forward(stage, batch, server_args)

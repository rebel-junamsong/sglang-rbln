# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from torch import nn

from sglang_rbln import envs
from sglang_rbln.compilation import placement
from sglang_rbln.compilation.backend import on_one_device

logger = logging.getLogger(__name__)

COMPILED = "_rbln_vocoder_compiled"


def _compile_on_one_device(module: nn.Module, what: str) -> None:
    """Compile `module` for one NPU: its own if placed, else the last TP one."""
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    compile_kwargs = build_torch_compile_kwargs(mode=None, module=module)
    compile_kwargs["dynamic"] = False
    options = compile_kwargs["options"]
    if devices := placement.devices_of(module):
        if len(devices) > 1:
            compile_kwargs["options"] = on_one_device(options, devices[-1], module)
    elif num_devices := options.get("num_devices"):
        compile_kwargs["options"] = on_one_device(options, num_devices - 1, module)
    module.compile(**compile_kwargs)
    setattr(module, COMPILED, True)
    logger.info("Compiling the LTX-2 %s: %s", what, compile_kwargs)


def compile_vocoder(forward, stage, batch, server_args):
    vocoder = stage.vocoder
    if (
        envs.SGLANG_RBLN_VOCODER_ON_NPU
        and not envs.SGLANG_RBLN_COMPILE_AUDIO
        and server_args.enable_torch_compile
        and not getattr(vocoder, COMPILED, False)
    ):
        _compile_on_one_device(vocoder, "vocoder")
    return forward(stage, batch, server_args)


class AudioChain(nn.Module):
    """Audio latent -> mel (audio VAE decoder) -> waveform (vocoder), one graph,
    so the mel never leaves the device between the two."""

    def __init__(self, decoder: nn.Module, vocoder: nn.Module):
        super().__init__()
        self.decoder = decoder
        self.vocoder = vocoder
        self.latent_dtype = next(decoder.parameters()).dtype

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        spectrogram = self.decoder(latents.to(self.latent_dtype))
        # The class forward: the instance's own is the compiled chain below.
        return type(self.vocoder).forward(self.vocoder, spectrogram)


def compile_audio(forward, stage, batch, server_args):
    """Run the audio VAE decoder and the vocoder as one compiled graph.

    The stage calls ``audio_vae.decode(latents, return_dict=False)`` and then
    ``vocoder(spectrogram)``. Here ``decode`` hands the latents through and the
    vocoder call runs the chain, so the stage's own code is unchanged.
    """
    audio_vae = getattr(stage, "audio_vae", None)
    vocoder = getattr(stage, "vocoder", None)
    if (
        envs.SGLANG_RBLN_COMPILE_AUDIO
        and server_args.enable_torch_compile
        and audio_vae is not None
        and vocoder is not None
        and not getattr(vocoder, COMPILED, False)
    ):
        chain = AudioChain(audio_vae.decoder, vocoder)
        if devices := placement.devices_of(vocoder):
            placement.place(chain, devices)
        _compile_on_one_device(chain, "audio VAE decoder + vocoder")
        audio_vae.decode = lambda latents, return_dict=True: (latents,)
        vocoder.forward = lambda latents, *args, **kwargs: chain(latents)
        setattr(vocoder, COMPILED, True)
    return forward(stage, batch, server_args)

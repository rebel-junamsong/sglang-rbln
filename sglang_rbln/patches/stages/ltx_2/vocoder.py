# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from sglang.multimodal_gen.runtime.utils.precision import align_tensor_to_module_dtype
from torch import nn

from sglang_rbln import envs
from sglang_rbln.compilation import placement
from sglang_rbln.compilation.backend import COMPILED, compile_whole


def audio_devices(vocoder: nn.Module) -> tuple[int, ...] | None:
    """The vocoder's own placement, else the text encoder's set.

    The DiT's set has no room for the audio graphs beside both DiT shapes and the VAE.
    """
    if devices := placement.devices_of(vocoder):
        return devices
    return placement.text_encoder_devices()


class AudioChain(nn.Module):
    """The audio VAE decoder and the vocoder as one graph, so the mel stays on the
    device between them. It takes the stage's place for both: the stage hands it
    the latents it would decode and gets the waveform back."""

    def __init__(self, audio_vae: nn.Module, vocoder: nn.Module):
        super().__init__()
        self.decoder = audio_vae.decoder
        self.vocoder = vocoder

    @property
    def sample_rate(self):
        return getattr(self.vocoder, "sample_rate", None)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        spectrogram = align_tensor_to_module_dtype(self.decoder(latents), self.vocoder)
        # The class forward: an instance-level forward on the vocoder makes the
        # traced program read the wrong weights on the device.
        return type(self.vocoder).forward(self.vocoder, spectrogram)


def compile_audio(forward, stage, batch, server_args):
    """Run the audio VAE decoder and the vocoder as one compiled graph.

    The stage denormalizes the latents, calls
    ``audio_vae.decode(latents, return_dict=False)`` and then ``vocoder(spectrogram)``.
    Here ``decode`` hands the latents through and the stage's vocoder becomes the
    compiled chain, so the stage's own code is unchanged.
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
        chain = AudioChain(audio_vae, vocoder)
        if devices := audio_devices(vocoder):
            placement.place(chain, devices)
        compile_whole(chain, "audio VAE decoder + vocoder")
        audio_vae.decode = lambda latents, return_dict=True: (latents,)
        stage.vocoder = chain
        setattr(chain, COMPILED, True)
    return forward(stage, batch, server_args)

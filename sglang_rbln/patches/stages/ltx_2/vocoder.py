# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from sglang_rbln import device, envs
from sglang_rbln.compilation import placement
from sglang_rbln.compilation.backend import COMPILED, compile_on_one_device


def audio_devices(vocoder: nn.Module) -> tuple[int, ...] | None:
    """The vocoder's own placement, else, with resident regions, the text encoder's set.

    The DiT's set has no room for the audio graphs beside both DiT shapes and the VAE.
    """
    if devices := placement.devices_of(vocoder):
        return devices
    return placement.text_encoder_devices() if device.resident_regions() else None


TILE = 64  # elements per device tile of the last axis


class AudioChain(nn.Module):
    """Audio latent -> mel (audio VAE decoder) -> waveform (vocoder), one graph.

    Every edge stays on the device: the latent comes in flat (channels x mel
    bins is a whole number of tiles at any length) and is denormalized and
    reshaped inside the graph, and the waveform goes out padded to whole tiles.
    Taking or returning the unaligned shapes instead makes the compiler pad,
    transpose or crop them on the host at the graph's edge.
    """

    def __init__(self, audio_vae: nn.Module, vocoder: nn.Module):
        super().__init__()
        self.decoder = audio_vae.decoder
        self.vocoder = vocoder
        self.latent_dtype = next(self.decoder.parameters()).dtype
        self.register_buffer("latents_mean", audio_vae.latents_mean.detach().clone())
        self.register_buffer("latents_std", audio_vae.latents_std.detach().clone())
        self.latent_shape: tuple[int, ...] = ()

    def waveform(self, flat: torch.Tensor) -> torch.Tensor:
        channels, frames, bins = self.latent_shape
        latents = flat.view(1, channels, frames, bins).to(self.latent_dtype)
        stats = (1, channels, 1, bins)
        latents = latents * self.latents_std.view(stats).to(
            self.latent_dtype
        ) + self.latents_mean.view(stats).to(self.latent_dtype)
        spectrogram = self.decoder(latents)
        # The class forward: the instance's own is the compiled chain.
        return type(self.vocoder).forward(self.vocoder, spectrogram.float())

    def forward(self, flat: torch.Tensor) -> torch.Tensor:
        waveform = self.waveform(flat)
        return F.pad(waveform, (0, -waveform.shape[-1] % TILE))


class CompiledAudio(nn.Module):
    """Takes the vocoder's place in the stage: runs the compiled chain on the
    latents the stage hands it and returns the waveform at its exact length.

    The vocoder module itself stays untouched: it is traced inside the chain,
    and an instance-level ``forward`` on it makes the traced program read the
    wrong weights on the device (the task aborts).
    """

    def __init__(self, chain: AudioChain):
        super().__init__()
        self.chain = chain
        self.lengths: dict[tuple[int, ...], int] = {}

    @property
    def sample_rate(self):
        return getattr(self.chain.vocoder, "sample_rate", None)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        shape = tuple(latents.shape[1:])
        flat = latents.reshape(1, -1).to(self.chain.latent_dtype)
        self.chain.latent_shape = shape
        if shape not in self.lengths:
            # The exact waveform length, from shapes only: the compiled output
            # is padded to whole tiles.
            with FakeTensorMode(allow_non_fake_inputs=True) as fake:
                probe = self.chain.waveform(fake.from_tensor(flat))
            self.lengths[shape] = probe.shape[-1]
        return self.chain(flat)[..., : self.lengths[shape]]


def compile_audio(forward, stage, batch, server_args):
    """Run the audio VAE decoder and the vocoder as one compiled graph.

    The stage denormalizes the latents with the audio VAE's statistics, calls
    ``audio_vae.decode(latents, return_dict=False)`` and then
    ``vocoder(spectrogram)``. Here the chain takes the statistics (the stage's
    host-side denormalization is switched off), ``decode`` hands the latents
    through and the stage's vocoder becomes the compiled chain, so the stage's
    own code is unchanged.
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
        compile_on_one_device(chain, "audio VAE decoder + vocoder")
        audio_vae.latents_mean = None
        audio_vae.latents_std = None
        audio_vae.decode = lambda latents, return_dict=True: (latents,)
        stage.vocoder = CompiledAudio(chain)
        setattr(stage.vocoder, COMPILED, True)
    return forward(stage, batch, server_args)

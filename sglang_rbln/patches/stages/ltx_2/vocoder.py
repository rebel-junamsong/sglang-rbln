# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import contextlib
from collections.abc import Iterator

import torch
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.utils.precision import (
    align_tensor_to_module_dtype,
    resolve_precision,
)
from torch import nn

from sglang_rbln import envs
from sglang_rbln.compilation import placement
from sglang_rbln.compilation.backend import COMPILED, compile_whole

# The audio graph of a request runs here while the video decodes on other NPUs.
_audio_worker = concurrent.futures.ThreadPoolExecutor(1, "rbln-audio")


def audio_devices(vocoder: nn.Module) -> tuple[int, ...] | None:
    """The vocoder's own placement, else the text encoder's set.

    The DiT's set has no room for the audio graphs beside both DiT shapes and the VAE.
    """
    return placement.devices_of(vocoder) or placement.text_encoder_devices()


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
        # The class forward, bypassing any instance-level forward.
        return type(self.vocoder).forward(self.vocoder, spectrogram)

    def __call__(self, *args, **kwargs):
        """The waveform started ahead for these latents, else a run now."""
        pending = self.__dict__.pop("_rbln_pending", None)
        if pending is not None:
            latents, future = pending
            # the worker's error surfaces here, where the stage asks for the audio
            waveform = future.result()
            if len(args) == 1 and not kwargs and torch.equal(args[0], latents):
                return waveform
        waveform = super().__call__(*args, **kwargs)
        self.__dict__["_rbln_ran"] = True
        return waveform


def stage_latents(stage, batch, server_args) -> torch.Tensor:
    """What LTX2AVDecodingStage hands its vocoder: the audio latents denormalized
    and cast as the stage does, then aligned to the vocoder's dtype."""
    audio_vae = stage.audio_vae
    dtype = resolve_precision(
        server_args, "audio_vae", precision_attr="audio_vae_precision"
    )
    device = get_local_torch_device()
    latents = batch.audio_latents.to(device, dtype=dtype)
    mean = getattr(audio_vae, "latents_mean", None)
    std = getattr(audio_vae, "latents_std", None)
    if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
        mean = mean.to(device=device, dtype=dtype)
        std = std.to(device=device, dtype=dtype)
        if latents.ndim == 4:
            mean = mean.view(1, latents.shape[1], 1, latents.shape[3])
            std = std.view(1, latents.shape[1], 1, latents.shape[3])
        latents = latents * std + mean
    latents = audio_vae.decode(latents, return_dict=False)[0]
    return align_tensor_to_module_dtype(latents, stage.vocoder)


def overlaps_the_video(chain: AudioChain, stage) -> bool:
    """Whether the audio graph shares an NPU with the video decoder's graphs."""
    audio = placement.devices_of(chain)
    video = placement.devices_of(stage.vae.decoder) or tuple(
        range(envs.SGLANG_RBLN_NUM_DEVICES or 1)
    )
    return audio is None or not set(audio).isdisjoint(video)


@contextlib.contextmanager
def audio_beside_video(stage, batch, server_args) -> Iterator[None]:
    """Run the request's audio graph while the stage decodes the video.

    The stage decodes the video, then the audio, each a few graph runs on its own
    NPUs. On NPUs of its own, the audio graph's 0.3 s at 1280x704x121 can run
    meanwhile: started here on the latents the stage will hand it, it is waited for
    where the stage asks for it. A first request (the chain still compiling), a
    chain on the video's NPUs, or a stage handing other latents runs it in place.
    """
    chain = getattr(stage, "vocoder", None)
    if (
        not isinstance(chain, AudioChain)
        or not chain.__dict__.get("_rbln_ran", False)
        or getattr(batch, "audio_latents", None) is None
        or overlaps_the_video(chain, stage)
    ):
        yield
        return
    latents = stage_latents(stage, batch, server_args)
    run = torch.no_grad()(super(AudioChain, chain).__call__)
    future = _audio_worker.submit(run, latents)
    chain.__dict__["_rbln_pending"] = (latents, future)
    try:
        yield
    except BaseException:
        # The stage failed before it took the audio: the graph still finishes
        # before the stage's error leaves, which stays the error raised.
        if chain.__dict__.pop("_rbln_pending", None) is not None:
            concurrent.futures.wait([future])
        raise
    # Not taken by a stage that succeeded: its error is not lost either.
    if chain.__dict__.pop("_rbln_pending", None) is not None:
        future.result()


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
        server_args.enable_torch_compile
        and audio_vae is not None
        and vocoder is not None
        and not getattr(vocoder, COMPILED, False)
    ):
        chain = AudioChain(audio_vae, vocoder)
        if devices := audio_devices(vocoder):
            placement.place(chain, devices)
        compile_whole(chain, "audio VAE decoder + vocoder")
        # The chain compiles layout-static: latents that arrive as a view would be
        # traced again beside the contiguous ones (11 s at the second request).
        audio_vae.decode = lambda latents, return_dict=True: (latents.contiguous(),)
        stage.vocoder = chain
        setattr(chain, COMPILED, True)
    return forward(stage, batch, server_args)

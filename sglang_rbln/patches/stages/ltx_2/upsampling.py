# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

from sglang_rbln import device, envs
from sglang_rbln.compilation import placement, residency
from sglang_rbln.compilation.backend import (
    COMPILED,
    compile_once,
    static_compile_kwargs,
)

STAGE = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".ltx_2.upsampling.LTX2UpsampleStage"
)
HOOK_TARGET = f"{STAGE}.forward"
UPSAMPLE_TARGET = f"{STAGE}._upsample_video_latents"
PACK_TARGET = f"{STAGE}._pack_video_latents"
GRAPHS = "_rbln_upsample_graphs"


def _devices(upsampler: nn.Module) -> tuple[int, ...] | None:
    return placement.devices_of(upsampler) or placement.text_encoder_devices()


def compile_upsampler(stage, batch, server_args, *args, **kwargs):
    """Compile the upsampler on the text encoder's NPUs, unless it was placed itself.

    The DiT's set has no room for it beside both DiT shapes and the VAE decoder.
    With resident regions it is not compiled alone: `upsample_video_latents`
    compiles it inside the graph that also scales the latents around it.
    BEFORE, not AROUND: the stage's one AROUND hook belongs to detailing.
    """
    upsampler = stage.spatial_upsampler
    if (
        envs.SGLANG_RBLN_UPSAMPLER_ON_NPU
        and server_args.enable_torch_compile
        and upsampler is not None
        and not getattr(upsampler, COMPILED, False)
    ):
        if placement.devices_of(upsampler) is None and (
            devices := placement.text_encoder_devices()
        ):
            placement.place(upsampler, devices)
        if not device.resident_regions():
            compile_once(
                upsampler, "latent upsampler", static_compile_kwargs(upsampler)
            )
    return None


class UpsampleLatents(nn.Module):
    """Upstream's `_upsample_video_latents` as one graph: undo the VAE's latent
    normalisation, upsample, normalise again."""

    def __init__(self, upsampler: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.upsampler = upsampler
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents * self.std + self.mean
        latents = self.upsampler(latents)
        return (latents - self.mean) / self.std


class PackLatents(nn.Module):
    """The pipeline config's latent packing, as one graph."""

    def __init__(self, pipeline_config) -> None:
        super().__init__()
        self.pipeline_config = pipeline_config

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.pipeline_config.maybe_pack_latents(latents, latents.shape[0], None)


def upsample_video_latents(original, self, latents, server_args, local_device):
    if not residency._is_resident(latents):
        return original(self, latents, server_args, local_device)

    def build() -> nn.Module:
        shape = (1, -1, 1, 1, 1)
        return UpsampleLatents(
            self.spatial_upsampler,
            self.vae.latents_mean.view(shape).to(latents.dtype),
            self.vae.latents_std.view(shape).to(latents.dtype),
        )

    module = residency.resident_module(
        self.__dict__.setdefault(GRAPHS, {}),
        ("upsample", tuple(latents.shape), latents.dtype),
        build,
        (latents,),
        _devices(self.spatial_upsampler),
    )
    return module(latents)


def pack_video_latents(original, batch, latents, server_args):
    if not residency._is_resident(latents) or latents.dim() == 3:
        return original(batch, latents, server_args)
    config = server_args.pipeline_config
    module = residency.resident_module(
        config.__dict__.setdefault(GRAPHS, {}),
        ("pack", tuple(latents.shape), latents.dtype),
        lambda: PackLatents(config),
        (latents,),
        residency.npus_of(latents.device),
    )
    packed = module(latents)
    batch.latents = packed
    batch.raw_latent_shape = packed.shape
    return None


def register() -> str:
    HookRegistry.register(HOOK_TARGET, compile_upsampler, HookType.BEFORE)
    return HOOK_TARGET


def register_resident() -> tuple[str, str]:
    HookRegistry.register(UPSAMPLE_TARGET, upsample_video_latents, HookType.AROUND)
    HookRegistry.register(PACK_TARGET, pack_video_latents, HookType.AROUND)
    return UPSAMPLE_TARGET, PACK_TARGET

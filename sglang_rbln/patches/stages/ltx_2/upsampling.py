# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

from sglang_rbln.compilation import placement, residency

STAGE = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".ltx_2.upsampling.LTX2UpsampleStage"
)
UPSAMPLE_TARGET = f"{STAGE}._upsample_video_latents"
PACK_TARGET = f"{STAGE}._pack_video_latents"
GRAPHS = "_rbln_upsample_graphs"


class UpsampleLatents(nn.Module):
    """Upstream's `_upsample_video_latents` as one graph: undo the VAE's latent
    normalization, upsample, normalize again."""

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

    # As upstream: the upsampler runs in the latents' dtype, which the component
    # manager gives it on entering the use.
    with self.use_declared_component(
        component_name="spatial_upsampler",
        module=self.spatial_upsampler,
        target_dtype=latents.dtype,
    ) as spatial_upsampler:

        def build() -> nn.Module:
            shape = (1, -1, 1, 1, 1)
            return UpsampleLatents(
                spatial_upsampler,
                self.vae.latents_mean.view(shape).to(latents.dtype),
                self.vae.latents_std.view(shape).to(latents.dtype),
            )

        module = residency.resident_module(
            self.__dict__.setdefault(GRAPHS, {}),
            ("upsample", tuple(latents.shape), latents.dtype),
            build,
            (latents,),
            placement.devices_of(spatial_upsampler) or placement.text_encoder_devices(),
        )
        return module(latents)


def pack_video_latents(original, batch, latents, server_args):
    # Three dims: already packed into (batch, tokens, channels).
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


def register() -> tuple[str, str]:
    HookRegistry.register(UPSAMPLE_TARGET, upsample_video_latents, HookType.AROUND)
    HookRegistry.register(PACK_TARGET, pack_video_latents, HookType.AROUND)
    return UPSAMPLE_TARGET, PACK_TARGET

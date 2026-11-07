# SPDX-License-Identifier: Apache-2.0
"""Stage 2 of the LTX-2 two-stage pipeline as the official DFR runs it: the
detailing IC-LoRA, conditioned in context on the half-resolution stage-1 latent.

The reference tokens follow ltx-core's ``VideoConditionByReferenceLatent``: they
are appended clean (timestep 0) after the noisy tokens, at the stage-1 grid's
positions with height and width scaled by the LoRA's ``reference_downscale_factor``,
and attend to the whole sequence. The LoRA is sglang's own dynamic LoRA. Stage 1
keeps it at strength 0 rather than deactivated, so both stages compile against
the same weights and share one device weight pool.
"""

from __future__ import annotations

import functools
import logging

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import envs

logger = logging.getLogger(__name__)

ADAPTER = "ltx2_detailing"
# ltx-pipelines dfr_helpers/ops.py DETAILING_LORA_STRENGTH
STAGE_STRENGTHS = {"stage1": 0.0, "stage2": 0.5}
REFERENCE = "rbln_ltx2_reference"
TARGET_TOKENS = "rbln_ltx2_target_tokens"

PIPELINE = "sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline.LTX2TwoStagePipeline"
STAGES = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2"
)
SWITCH_PHASE = f"{PIPELINE}.switch_lora_phase"
UPSAMPLE = f"{STAGES}.upsampling.LTX2UpsampleStage.forward"
PREPARE_LOOP = f"{STAGES}.denoising.LTX2DenoisingStage._prepare_denoising_loop"
MODEL_INPUTS = f"{STAGES}.denoising.LTX2DenoisingStage._prepare_ltx2_model_inputs"
POST_LOOP = f"{STAGES}.denoising_av.LTX2AVDenoisingStage._post_denoising_loop"


@functools.cache
def reference_downscale_factor(path: str) -> int:
    from safetensors import safe_open

    with safe_open(path, framework="pt") as file:
        return int((file.metadata() or {}).get("reference_downscale_factor", 1))


def switch_lora_phase(switch, pipeline, phase, *args, **kwargs):
    switch(pipeline, phase, *args, **kwargs)
    if (path := envs.SGLANG_RBLN_LTX2_DETAILING_LORA) and phase in STAGE_STRENGTHS:
        pipeline.set_lora(
            ADAPTER,
            str(path),
            target="transformer",
            strength=STAGE_STRENGTHS[phase],
            merge_weights=False,
        )


def keep_reference(forward, stage, batch, server_args, *args, **kwargs):
    """Keep the stage-1 latent, as stage-2 tokens with their RoPE positions."""
    if not (path := envs.SGLANG_RBLN_LTX2_DETAILING_LORA):
        return forward(stage, batch, server_args, *args, **kwargs)
    latents = batch.latents
    _, _, frames, height, width = latents.shape
    tokens = server_args.pipeline_config.maybe_pack_latents(
        latents, latents.shape[0], batch
    )
    fps = batch.fps
    batch = forward(stage, batch, server_args, *args, **kwargs)
    batch.extra[REFERENCE] = (
        tokens.contiguous(),
        (frames, height, width, fps, reference_downscale_factor(str(path))),
    )
    return batch


def reference_coords(model, reference, batch_size: int, device) -> torch.Tensor:
    _, (frames, height, width, fps, downscale) = reference
    coords = model.rope.prepare_video_coords(
        batch_size=batch_size,
        num_frames=frames,
        height=height,
        width=width,
        device=device,
        fps=fps,
    )
    coords[:, 1:3] *= downscale
    return coords


def append_reference(prepare, stage, batch, server_args, *args, **kwargs):
    ctx = prepare(stage, batch, server_args, *args, **kwargs)
    reference = batch.extra.get(REFERENCE)
    if reference is None or ctx.stage != "stage2":
        return ctx
    if batch.did_sp_shard_latents:
        raise NotImplementedError("LTX-2 reference tokens under sequence parallelism")
    latents = ctx.latents
    tokens = reference[0].to(device=latents.device, dtype=latents.dtype)
    tokens = tokens.expand(latents.shape[0], -1, -1)
    mask = ctx.denoise_mask
    if mask is None:
        mask = torch.ones(
            (*latents.shape[:2], 1), device=latents.device, dtype=torch.float32
        )
    clean = ctx.clean_latent if ctx.clean_latent is not None else latents
    batch.extra[TARGET_TOKENS] = latents.shape[1]
    ctx.latents = torch.cat([latents, tokens], dim=1)
    ctx.clean_latent = torch.cat([clean, tokens.to(clean.dtype)], dim=1)
    ctx.denoise_mask = torch.cat([mask, mask.new_zeros(*tokens.shape[:2], 1)], dim=1)
    return ctx


def reference_positions(build, stage, ctx, step, batch, *args, **kwargs):
    inputs = build(stage, ctx, step, batch, *args, **kwargs)
    reference = batch.extra.get(REFERENCE)
    if reference is None or TARGET_TOKENS not in batch.extra:
        return inputs
    model = step.current_model
    latents = inputs.latent_model_input
    coords = inputs.video_coords
    if coords is None:
        coords = model.rope.prepare_video_coords(
            batch_size=latents.shape[0],
            num_frames=ctx.latent_num_frames_for_model,
            height=ctx.latent_height,
            width=ctx.latent_width,
            device=latents.device,
            fps=batch.fps,
        )
    inputs.video_coords = torch.cat(
        [coords, reference_coords(model, reference, latents.shape[0], coords.device)],
        dim=2,
    )
    return inputs


def drop_reference(finish, stage, batch, latents, *args, **kwargs):
    if (count := batch.extra.pop(TARGET_TOKENS, None)) is not None:
        batch.extra.pop(REFERENCE, None)
        latents = latents[:, :count]
    return finish(stage, batch, latents, *args, **kwargs)


HOOKS = {
    SWITCH_PHASE: switch_lora_phase,
    UPSAMPLE: keep_reference,
    PREPARE_LOOP: append_reference,
    MODEL_INPUTS: reference_positions,
    POST_LOOP: drop_reference,
}


def register() -> tuple[str, ...]:
    for target, hook in HOOKS.items():
        HookRegistry.register(target, hook, HookType.AROUND)
    return tuple(HOOKS)

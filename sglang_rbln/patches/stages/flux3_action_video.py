# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import copy
import dataclasses
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

PROTOCOL = "sglang.multimodal_gen.runtime.entrypoints.action.protocol"
API = "sglang.multimodal_gen.runtime.entrypoints.action.api"
STAGES = "sglang.multimodal_gen.runtime.pipelines_core.stages"
FLUX3_STAGES = f"{STAGES}.model_specific_stages.flux3_action"
PIPELINE = "sglang.multimodal_gen.runtime.pipelines.flux3_action.Flux3ActionPipeline"
VAE = "sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae.Flux3VideoVAE"

REQUEST_TARGET = f"{PROTOCOL}._build_action_model_sampling_params"
LOAD_TARGET = f"{PIPELINE}._load_vae"
OBSERVATION_TARGET = f"{VAE}.encode_frame"
SAMPLER_TARGET = f"{FLUX3_STAGES}.cosmos_unipc"
POSTPROCESS_TARGET = f"{STAGES}.vla.VLAActionPostprocessStage.forward"
RESPONSE_TARGET = f"{API}.action_generation_response"

DROID_CONTENT_HW = (540, 640)

_latents = threading.local()


def request_video(sampling_params, payload: dict[str, Any], *args):
    parameters = payload.get("parameters") or {}
    sampling_params.return_video = bool(parameters.get("return_video", False))
    return sampling_params


def load_vae_with_decoder(load_vae, server_args, config):
    vae_config = dataclasses.replace(config.vae_config, load_decoder=True)
    vae = load_vae(server_args, dataclasses.replace(config, vae_config=vae_config))
    vae.model.decoder.float()
    return vae


def keep_observation_latent(latent: torch.Tensor, vae, *args) -> torch.Tensor:
    _latents.vae, _latents.observation = vae, latent
    return latent


def keep_denoised_video(samples: dict[str, torch.Tensor], *args, **kwargs):
    _latents.video = samples["video"]
    return samples


def content_hw(config) -> tuple[int, int]:
    return DROID_CONTENT_HW if config.camera_layout == "droid" else config.canvas_hw


def decode_video(config) -> torch.Tensor:
    vae = _latents.vae
    h, w = config.latent_hw
    observation = _latents.observation[..., :h, :w].float()
    video = rearrange(_latents.video[:1].float(), "b (t h w) c -> b c t h w", h=h, w=w)
    latents = torch.cat([observation, video], dim=2)
    decoder = vae.model.decoder
    normalizer = copy.deepcopy(vae.model.z_normalizer).float()
    latents = normalizer.denormalize(latents).to(next(decoder.parameters()))
    with torch.no_grad():
        frames = decoder(latents).float().clamp(-1, 1)[0].cpu()
    height, width = content_hw(config)
    frames = (frames[:, :, :height, :width] + 1) * 127.5
    return frames.round().to(torch.uint8).permute(1, 2, 3, 0).contiguous()


def encode_mp4(frames: torch.Tensor, fps: float) -> bytes:
    import imageio_ffmpeg

    _, height, width, _ = frames.shape
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "video.mp4"
        writer = imageio_ffmpeg.write_frames(
            str(path),
            (width, height),
            fps=fps,
            codec="libx264",
            quality=None,
            macro_block_size=1,
            output_params=["-crf", "20"],
        )
        writer.send(None)
        for frame in frames.numpy():
            writer.send(frame)
        writer.close()
        return path.read_bytes()


def attach_video(output, stage, batch, server_args):
    if not getattr(batch, "return_video", False):
        return output
    start = time.perf_counter()
    config = server_args.pipeline_config
    frames = decode_video(config)
    payload = output.output[0]
    payload["video"] = {
        "type": "video",
        "format": "mp4",
        "fps": config.fps,
        "shape": list(frames.shape),
        "b64_json": base64.b64encode(encode_mp4(frames, config.fps)).decode(),
    }
    if "timings" in payload:
        payload["timings"]["video_decode_ms"] = (time.perf_counter() - start) * 1000
    return output


def attach_video_to_response(
    response: dict[str, Any], output: dict[str, Any], *args, **kwargs
):
    if (video := output.get("video")) is not None:
        response["data"][0]["video"] = video
    return response


def register() -> tuple[str, ...]:
    hooks = {
        REQUEST_TARGET: (request_video, HookType.AFTER),
        LOAD_TARGET: (load_vae_with_decoder, HookType.AROUND),
        OBSERVATION_TARGET: (keep_observation_latent, HookType.AFTER),
        SAMPLER_TARGET: (keep_denoised_video, HookType.AFTER),
        POSTPROCESS_TARGET: (attach_video, HookType.AFTER),
        RESPONSE_TARGET: (attach_video_to_response, HookType.AFTER),
    }
    for target, (hook, hook_type) in hooks.items():
        HookRegistry.register(target, hook, hook_type)
    return tuple(hooks)

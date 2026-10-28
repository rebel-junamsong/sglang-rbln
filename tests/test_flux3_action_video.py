# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import io
from types import SimpleNamespace

import imageio.v3 as iio
import pytest
import torch
from einops import rearrange
from sglang.multimodal_gen.configs.models.vaes.flux3_video import Flux3VideoVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.flux3_action import (
    Flux3ActionPipelineConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae import (
    DistributedRunningStats,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln.patches.stages import flux3_action_video as video

LATENT_FRAMES = 3


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((), dtype=torch.bfloat16))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        b, _, t, h, w = latents.shape
        frames = latents[:, :3].repeat_interleave(32, -1).repeat_interleave(32, -2)
        frames = torch.cat(
            [frames[:, :, :1], frames[:, :, 1:].repeat_interleave(4, 2)], 2
        )
        return frames.tanh() * self.weight.float()


class _VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.decoder = _Decoder()
        self.model.z_normalizer = DistributedRunningStats(96)
        self.model.z_normalizer.running_mean.uniform_(-1, 1)
        self.model.z_normalizer.running_var.uniform_(0.5, 2)
        self.model.z_normalizer.to(torch.bfloat16)
        self.decoded = []
        self.model.decoder.register_forward_pre_hook(
            lambda module, args: self.decoded.append(args[0])
        )


@pytest.fixture
def config() -> Flux3ActionPipelineConfig:
    return Flux3ActionPipelineConfig()


@pytest.fixture
def latents(config):
    h, w = config.latent_hw
    vae = _VAE()
    observation = torch.randn(1, 96, 1, h + 1, w + 1, dtype=torch.bfloat16)
    predicted = torch.randn(1, LATENT_FRAMES * h * w, 96)
    video.keep_observation_latent(observation, vae, None)
    video.keep_denoised_video({"video": predicted, "action": None}, None)
    return SimpleNamespace(vae=vae, observation=observation, predicted=predicted)


def _output() -> OutputBatch:
    return OutputBatch(output=[{"actions": [[0.0]], "timings": {}}])


def test_registers_every_hook():
    targets = video.register()
    assert len(targets) == len(set(targets)) == 6
    for target in targets:
        assert any(
            hook.__module__ == video.__name__
            for _, hook, _ in HookRegistry._hooks[target]
        )


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [({"return_video": True}, True), ({"seed": 0}, False), (None, False)],
)
def test_request_opts_in_through_parameters(parameters, expected):
    sampling_params = SimpleNamespace()
    payload = {} if parameters is None else {"parameters": parameters}
    assert video.request_video(sampling_params, payload, None, None) is sampling_params
    assert sampling_params.return_video is expected


def test_loads_the_decoder_in_float32(config):
    calls = []

    def load_vae(server_args, pipeline_config):
        calls.append(pipeline_config)
        return _VAE()

    vae = video.load_vae_with_decoder(load_vae, None, config)
    assert [c.vae_config.load_decoder for c in calls] == [True]
    assert isinstance(config.vae_config, Flux3VideoVAEConfig)
    assert config.vae_config.load_decoder is False
    assert vae.model.decoder.weight.dtype == torch.float32


def test_decodes_observed_then_denoised_latents_in_float32(config, latents):
    frames = video.decode_video(config)
    h, w = config.latent_hw
    normalized = torch.cat(
        [
            latents.observation[..., :h, :w].float(),
            rearrange(latents.predicted, "b (t h w) c -> b c t h w", h=h, w=w),
        ],
        dim=2,
    )
    stats = latents.vae.model.z_normalizer
    std = stats.running_var.float().sqrt().view(1, -1, 1, 1, 1)
    mean = stats.running_mean.float().view(1, -1, 1, 1, 1)
    (decoded,) = latents.vae.decoded
    assert decoded.dtype == torch.float32
    torch.testing.assert_close(decoded, normalized * std + mean, rtol=0, atol=0)
    assert not torch.equal(decoded, stats.denormalize(normalized))
    assert frames.dtype == torch.uint8
    assert tuple(frames.shape) == (
        1 + 4 * LATENT_FRAMES,
        *video.content_hw(config),
        3,
    )


def test_attaches_an_mp4_when_requested(config, latents):
    server_args = SimpleNamespace(pipeline_config=config)
    batch = SimpleNamespace(return_video=True)
    payload = video.attach_video(_output(), None, batch, server_args).output[0]
    clip = payload["video"]
    assert {k: v for k, v in clip.items() if k != "b64_json"} == {
        "type": "video",
        "format": "mp4",
        "fps": config.fps,
        "shape": [1 + 4 * LATENT_FRAMES, *video.content_hw(config), 3],
    }
    frames = iio.imread(
        io.BytesIO(base64.b64decode(clip["b64_json"])), extension=".mp4"
    )
    assert list(frames.shape) == clip["shape"]
    assert payload["timings"]["video_decode_ms"] > 0


@pytest.mark.parametrize(
    "batch", [SimpleNamespace(), SimpleNamespace(return_video=False)]
)
def test_leaves_action_only_requests_alone(config, batch):
    output = _output()
    server_args = SimpleNamespace(pipeline_config=config)
    assert video.attach_video(output, None, batch, server_args).output[0] == {
        "actions": [[0.0]],
        "timings": {},
    }


def test_envelope_carries_the_video():
    response = {"data": [{"index": 0, "action": {}}]}
    clip = {"type": "video", "b64_json": ""}
    assert video.attach_video_to_response(response, {"video": clip}, None) == {
        "data": [{"index": 0, "action": {}, "video": clip}]
    }
    assert (
        "video"
        not in video.attach_video_to_response({"data": [{}]}, {"actions": []}, None)[
            "data"
        ][0]
    )

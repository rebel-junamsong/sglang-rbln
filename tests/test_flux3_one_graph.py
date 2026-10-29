# SPDX-License-Identifier: Apache-2.0
"""Each FLUX 3 Action DiT entry point traces as one graph.

The platform compiles the text context encoder, the conditioning stream encoder
and a whole denoise step with fullgraph=True; a graph break anywhere in the
transformer or the stage helpers fails here before it fails a serve.
"""

import functools
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.configs.models.dits.flux3 import (
    Flux3ArchConfig,
    Flux3DiTConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.flux3_action import (
    Flux3ActionPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import flux3 as dit
from sglang.multimodal_gen.runtime.models.schedulers import (
    scheduling_flow_unipc_multistep as unipc,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    flux3_action as stages,
)
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module
from sglang.srt.utils.network import get_open_port

from sglang_rbln.patches.models import flux3 as flux3_patch
from sglang_rbln.patches.stages import flux3_action_step as step


@pytest.fixture
def config():
    config = Flux3ActionPipelineConfig()
    config.camera_layout = "grid"
    config.canvas_hw = (64, 96)
    config.action_horizon = 4
    config.action_dim = config.state_dim = 3
    return config


@pytest.fixture
def server_args(monkeypatch):
    server_args = SimpleNamespace(
        attention_backend="torch_sdpa",
        attention_backend_config=None,
        comfyui_mode=False,
        disable_autocast=False,
        enable_cfg_parallel=False,
        enable_breakable_cuda_graph=False,
        enable_layerwise_nvtx_marker=False,
        enable_torch_compile=False,
        kv_gather_degree=1,
        sp_split_auto=False,
        model_loaded={},
        model_paths={},
        pipeline_config=SimpleNamespace(dit_precision="bfloat16"),
    )
    monkeypatch.setattr(server_args_module, "_global_server_args", server_args)
    return server_args


@pytest.fixture
def model(monkeypatch, config, server_args):
    for module in (dit, stages):
        monkeypatch.setattr(module, "Flux3SegmentState", flux3_patch.Flux3SegmentState)
    for stage, methods in step.BACKPORTED.items():
        for method in methods:
            monkeypatch.setattr(
                getattr(stages, stage), method.__name__, method, raising=False
            )
    if not model_parallel_is_initialized():
        for key, value in dict(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(get_open_port()),
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        ).items():
            monkeypatch.setenv(key, value)
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    action = config.action_modality
    arch = Flux3ArchConfig(
        hidden_size=64,
        num_attention_heads=4,
        depth=2,
        depth_single_blocks=2,
        axes_dim=(4, 4, 4, 4),
        context_in_dim=32,
        vec_in_dim=16,
    ).with_streams({action: 3, f"{action}_cond": 3})
    torch.manual_seed(0)
    transformer = (
        dit.Flux3Transformer(Flux3DiTConfig(arch_config=arch)).bfloat16().eval()
    )
    for p in transformer.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return transformer


class _VideoVAE(torch.nn.Module):
    def __init__(self, latent):
        super().__init__()
        self.latent = latent

    def encode_frame(self, frame):
        return self.latent.to(frame.dtype)


@pytest.fixture
def observation_stage(config, model):
    vae = _VideoVAE(torch.randn(1, stages.LATENT_CHANNELS, 1, *config.latent_hw))
    return stages.Flux3ActionObservationEncodingStage(config, model, vae)


@pytest.fixture
def denoising_stage(config, model):
    return stages.Flux3ActionDenoisingStage(
        config, model, unipc.FlowUniPCMultistepScheduler()
    )


def contexts(model):
    with set_forward_context(current_timestep=0, attn_metadata=None):
        return [
            model.encode_context(torch.full((1, 5, 32), scale), stages.text_ids(5))
            for scale in (0.1, -0.1)
        ]


def one_graph(fn):
    torch._dynamo.reset()
    return torch.compile(fn, backend="eager", fullgraph=True, dynamic=False)


def assert_same(actual, expected):
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


@torch.no_grad()
def test_backport_matches_the_installed_forwards(
    config, model, observation_stage, denoising_stage, server_args
):
    def run(encode_observation, denoise):
        batch = SimpleNamespace(
            extra={
                "vla": {
                    "flux3_observation": stages.Flux3ActionObservation(
                        canvas=torch.zeros(3, *config.canvas_hw),
                        state=torch.linspace(0, 1, config.state_dim),
                        prompt="",
                    ),
                    "flux3_contexts": contexts(model),
                    "flux3_guidance": {"video": 3.0, config.action_modality: 2.0},
                }
            },
            num_inference_steps=2,
            seed=0,
            action_horizon=None,
        )
        encode_observation(observation_stage, batch, server_args)
        denoise(denoising_stage, batch, server_args)
        return stages.vla_state(batch)

    expected = run(
        stages.Flux3ActionObservationEncodingStage.forward,
        stages.Flux3ActionDenoisingStage.forward,
    )
    actual = run(step.encode_observation, step.denoise)
    assert_same(actual["actions"], expected["actions"])
    for a, e in zip(
        actual["flux3_conditioning"], expected["flux3_conditioning"], strict=True
    ):
        assert a.name == e.name
        assert_same(step._state_tensors(a), step._state_tensors(e))


@torch.no_grad()
def test_dit_entry_points_trace_as_one_graph(
    config, model, observation_stage, denoising_stage
):
    action = config.action_modality
    video, video_ids = stages.pack_video(
        torch.randn(1, stages.LATENT_CHANNELS, 2, 2, 3), first_frame=1, fps=15.0
    )
    cond, cond_ids = stages.pack_video(
        torch.randn(1, stages.LATENT_CHANNELS, 1, 2, 3), first_frame=0, fps=15.0
    )
    act, act_ids = stages.pack_action(
        torch.randn(1, 3, 4), seconds=(torch.arange(4).float() + 1) / 15
    )
    state, state_ids = stages.pack_action(torch.randn(1, 3, 1), seconds=torch.zeros(1))
    with set_forward_context(current_timestep=0, attn_metadata=None):
        ctx = torch.randn(1, 5, 32)
        context = model.encode_context(ctx, stages.text_ids(5))
        assert_same(
            step._state_tensors(
                one_graph(model.encode_context)(ctx, stages.text_ids(5))
            ),
            step._state_tensors(context),
        )

        stream_inputs = (cond, cond_ids, state, state_ids, torch.zeros(1))
        encode_streams = observation_stage._encode_streams
        conditioning = encode_streams(*stream_inputs)
        assert_same(one_graph(encode_streams)(*stream_inputs), conditioning)

        guidance = (context, *contexts(model)[1:])
        state_names = ["video_cond", f"{action}_cond", *(c.name for c in guidance)]
        states = (*conditioning, *(t for c in guidance for t in step._state_tensors(c)))
        denoise_step = functools.partial(
            denoising_stage._denoise_step,
            order=["video", action],
            state_names=state_names,
        )
        args = (
            video.bfloat16(),
            act.bfloat16(),
            torch.full((1,), 0.7),
            model.rope(video_ids),
            model.rope(act_ids),
            *states,
        )
        assert_same(one_graph(denoise_step)(*args), denoise_step(*args))

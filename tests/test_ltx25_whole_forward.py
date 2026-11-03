# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.configs.models.base import ArchConfig
from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2Config
from sglang.multimodal_gen.configs.models.dits.ltx_2_5 import LTX25ArchConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import ltx_2 as dit
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module
from sglang.srt.utils.network import get_open_port

from sglang_rbln.patches.configs import arch_config

pytestmark = pytest.mark.usefixtures("rbln_platform")

FRAMES, HEIGHT, WIDTH, AUDIO_FRAMES, TEXT = 2, 4, 4, 8, 6


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(ArchConfig, "__getattr__", arch_config.extra_attr)
    monkeypatch.setattr(
        server_args_module,
        "_global_server_args",
        SimpleNamespace(
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
        ),
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
    arch = LTX25ArchConfig(
        num_attention_heads=2,
        attention_head_dim=64,
        num_layers=2,
        cross_attention_dim=128,
        caption_channels=64,
        audio_num_attention_heads=2,
        audio_attention_head_dim=64,
        audio_cross_attention_dim=128,
    )
    torch.manual_seed(0)
    transformer = dit.LTX2VideoTransformer3DModel(
        LTX2Config(arch_config=arch), hf_config={}
    )
    transformer = transformer.bfloat16().eval()
    for parameter in transformer.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    return transformer


def test_dit_forward_traces_as_one_graph(model):
    """--regional-compile false compiles the whole DiT: latents in, velocities
    out, with the patchify, adaLN and output layers on the device."""
    inputs = dict(
        hidden_states=torch.randn(1, FRAMES * HEIGHT * WIDTH, 128).bfloat16(),
        audio_hidden_states=torch.randn(1, AUDIO_FRAMES, 128).bfloat16(),
        encoder_hidden_states=torch.randn(1, TEXT, 128).bfloat16(),
        audio_encoder_hidden_states=torch.randn(1, TEXT, 128).bfloat16(),
        timestep=torch.full((1,), 0.5),
        num_frames=FRAMES,
        height=HEIGHT,
        width=WIDTH,
        audio_num_frames=AUDIO_FRAMES,
    )
    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph

    torch._dynamo.reset()
    with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
        expected = model(**inputs)
        compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
        outputs = compiled(**inputs)
    assert len(graphs) == 1
    for output, reference in zip(outputs, expected, strict=True):
        torch.testing.assert_close(output, reference)

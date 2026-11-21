# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.compilation import placement, residency
from sglang_rbln.patches.models import ltx_2_text_features as features

TOKENS = 16


@pytest.mark.parametrize(
    ("placed", "given", "expected"),
    [
        ((4, 5, 6, 7), None, torch.device("rbln", 1)),
        ((4, 5, 6, 7), "cpu", "cpu"),
        (None, None, None),
    ],
    ids=["placed", "explicit", "unplaced"],
)
def test_text_features_are_returned_on_the_connectors_npus(
    monkeypatch, placed, given, expected
):
    # Upstream returns them on the platform device, the host: 367 MiB down, and up
    # again into each connector.
    monkeypatch.setattr(placement, "_components", {"connectors": placed})
    monkeypatch.setattr(residency, "logical_device_on", lambda devices: 1)
    seen = []

    def upstream(stage, text, server_args, **kwargs):
        seen.append(kwargs.get("device"))

    features.encode_text(upstream, None, "a yak", None, device=given)
    assert seen == [expected]


@pytest.mark.parametrize("target", features.HOOKS)
def test_wraps_text_encoding_and_packing(target):
    assert target in features.register()
    assert any(
        hook is features.HOOKS[target] and kind is HookType.AROUND
        for kind, hook, _ in HookRegistry._hooks[target]
    )


LTX25_CONFIG = SimpleNamespace(
    dit_config=SimpleNamespace(
        arch_config=SimpleNamespace(caption_proj_before_connector=True)
    )
)
LAYERS = 4  # Gemma's hidden states: the embedding output and one per layer


def _gemma_postprocess():
    from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
        _gemma_postprocess_func,
    )

    # the hook wraps the module's function; the graph runs the one underneath it
    return getattr(_gemma_postprocess_func, "__wrapped__", _gemma_postprocess_func)


def test_host_hidden_states_are_postprocessed_by_upstream_as_is():
    seen = []

    def upstream(outputs, text_inputs, pipeline_config=None):
        seen.append({state.device.type for state in outputs.hidden_states})
        return "packed"

    outputs = SimpleNamespace(hidden_states=(torch.zeros(1, TOKENS, 8),) * 3)
    text_inputs = {"attention_mask": torch.ones(1, TOKENS)}
    assert features.postprocess_text_features(upstream, outputs, text_inputs) == (
        "packed"
    )
    assert seen == [{"cpu"}]
    assert features._postprocess_graphs == {}


def test_resident_hidden_states_are_postprocessed_in_one_graph_on_the_device(
    dummy_device, monkeypatch, tmp_path
):
    """Eager, the stack of the encoder's hidden states would gather each one to a
    single NPU; as one graph of upstream's code it reads them where they are."""
    import rebel

    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(features, "_postprocess_graphs", {})
    compile_module = torch.nn.Module.compile

    def compile_resident(module, *args, **kwargs):
        # as the server's residency hook does: the call takes its inputs to the NPUs
        compile_module(module, *args, **kwargs)
        residency.install_resident_call(module)

    monkeypatch.setattr(torch.nn.Module, "compile", compile_resident)
    # the embedding output comes from the host, every layer's from the NPUs
    hidden_states = (
        torch.randn(1, 128, 64, dtype=torch.bfloat16),
        *(
            torch.randn(1, 128, 64, dtype=torch.bfloat16).to("rbln:0")
            for _ in range(LAYERS)
        ),
    )
    outputs = SimpleNamespace(hidden_states=hidden_states)
    text_inputs = {"attention_mask": torch.ones(1, 128, dtype=torch.long)}

    packed = [
        features.postprocess_text_features(
            _gemma_postprocess(), outputs, text_inputs, LTX25_CONFIG
        )
        for _ in range(2)
    ]
    assert len(features._postprocess_graphs) == 1
    assert {(out.device.type, tuple(out.shape)) for out in packed} == {
        ("rbln", (1, 128, 64 * (LAYERS + 1)))
    }
    (artifact,) = tmp_path.rglob("*.rbln")
    stats = rebel.RBLNCompiledModel.inspect(str(artifact))["subgraph"]["stats"]
    assert [stat["device_func_count"] for stat in stats] == [1]
    host_ops = {
        op
        for stat in stats
        for ops in stat["host_op_reason"].values()
        for op in ops
        if not op.startswith("args_")
    }
    assert host_ops == set()


CROSS_SET_PROBE = """
import json, os, sys, tempfile
from types import SimpleNamespace
os.environ["SGLANG_RBLN_CACHE_DIR"] = tempfile.mkdtemp()
import torch, torch_rbln  # noqa: F401
import sglang.multimodal_gen.runtime.entrypoints.cli.serve  # noqa: F401
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import _gemma_postprocess_func
from sglang_rbln.compilation import backend, residency
from sglang_rbln.patches.models import ltx_2_text_features as features

backend.ensure_compile_backend()
residency.register()
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
HookRegistry.apply_hooks()

crossings = []
index = lambda device: torch.device(device).index or 0
original_to, original_copy = torch.Tensor.to, torch.Tensor.copy_


def to(self, *args, **kwargs):
    target = kwargs.get("device", args[0] if args else None)
    if (
        isinstance(target, (str, torch.device))
        and torch.device(target).type == "rbln"
        and self.device.type == "rbln"
        and index(target) != index(self.device)
    ):
        crossings.append(("to", str(self.device), str(target)))
    return original_to(self, *args, **kwargs)


def copy_(self, src, *args, **kwargs):
    if self.device.type == src.device.type == "rbln" and index(self) != index(src):
        crossings.append(("copy_", str(src.device), str(self.device)))
    return original_copy(self, src, *args, **kwargs)


torch.Tensor.to, torch.Tensor.copy_ = to, copy_
original_across = residency.copy_across_devices
residency.copy_across_devices = lambda tensor, device, *a, **k: (
    crossings.append(("across", str(tensor.device), str(device)))
    or original_across(tensor, device, *a, **k)
)

config = SimpleNamespace(
    dit_config=SimpleNamespace(
        arch_config=SimpleNamespace(caption_proj_before_connector=True)
    )
)
states = (torch.randn(1, 128, 64, dtype=torch.bfloat16),) + tuple(
    torch.randn(1, 128, 64, dtype=torch.bfloat16).to("rbln:1") for _ in range(4)
)
outputs = SimpleNamespace(hidden_states=states)
mask = {"attention_mask": torch.ones(1, 128, dtype=torch.long)}
with residency.DeviceBoundary():
    packed = features.postprocess_text_features(
        _gemma_postprocess_func, outputs, mask, config
    )
print(json.dumps({"device": str(packed.device), "crossings": crossings}))
"""


def test_hidden_states_on_the_encoders_npus_never_cross_to_another_set(
    rbln_platform_env,
):
    env = {
        **os.environ,
        **rbln_platform_env,
        "RBLN_DUMMY_DEVICE": "1",
        "RBLN_DEVICE_MAP": "[0],[1]",
        "PYTHONWARNINGS": "ignore",
    }
    # the map names the NPUs; a caller's narrower visible list would hide set 1
    env.pop("RBLN_VISIBLE_DEVICES", None)
    completed = subprocess.run(
        [sys.executable, "-c", CROSS_SET_PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    (line,) = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    result = json.loads(line)
    assert result == {"device": "rbln:1", "crossings": []}


LENGTH = 16
MASKS = {
    "none": [0] * LENGTH,
    "left-padded": [0] * 5 + [1] * (LENGTH - 5),
    "scattered": [1, 0, 1, 1, 0, 0, 1, 0] * 2,
    "all": [1] * LENGTH,
}


def _connector_forward():
    from sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector import (
        LTX2ConnectorTransformer1d,
    )

    forward = LTX2ConnectorTransformer1d.forward
    return LTX2ConnectorTransformer1d, getattr(forward, "__wrapped__", forward)


@pytest.fixture
def on_the_npus(monkeypatch):
    """Take the device path on the host: every tensor counts as resident, and the
    connector transformer runs eagerly instead of compiling."""
    monkeypatch.setattr(residency, "_is_resident", lambda value: True)
    monkeypatch.setattr(
        residency,
        "resident_module",
        lambda cache, key, build, tensors, devices=None: build(),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mask", sorted(MASKS))
@pytest.mark.parametrize("rope_type", ["split", "interleaved"])
def test_the_connector_graph_matches_upstream_bit_for_bit(
    on_the_npus, dtype, mask, rope_type
):
    transformer, forward = _connector_forward()
    torch.manual_seed(0)
    connector = transformer(
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=2,
        num_learnable_registers=8,
        rope_type=rope_type,
        apply_gated_attention=True,
    ).to(dtype)
    hidden = torch.randn(1, LENGTH, 32).to(dtype)
    additive = (torch.tensor([MASKS[mask]]) - 1).to(dtype) * torch.finfo(dtype).max
    with torch.no_grad():
        expected = forward(connector, hidden.clone(), additive.clone())
        actual = features.connector_transformer_forward(
            forward, connector, hidden.clone(), additive.clone()
        )
    for got, want in zip(actual, expected, strict=True):
        assert got.dtype == want.dtype and torch.equal(got, want)


def test_host_hidden_states_run_upstreams_connector_as_is(monkeypatch):
    transformer, forward = _connector_forward()
    connector = transformer(
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=1,
        num_learnable_registers=8,
    )
    monkeypatch.setattr(
        residency, "resident_module", lambda *args, **kwargs: pytest.fail("compiled")
    )
    hidden, additive = torch.randn(1, LENGTH, 32), torch.zeros(1, LENGTH)
    with torch.no_grad():
        actual = features.connector_transformer_forward(
            forward, connector, hidden, additive
        )
        expected = forward(connector, hidden, additive)
    assert all(map(torch.equal, actual, expected))


def test_the_connector_stage_compiles_one_graph_per_connector_and_projection(
    rbln_platform, dummy_device, monkeypatch, tmp_path
):
    """Eager on the NPUs, the register replacement, the output norm and block 0's
    learning pass ran op by op, each a graph on one NPU: 40 graphs at the yak shape
    where the projections and connectors are 4."""
    from sglang.multimodal_gen.configs.models.adapter.ltx_2_connector import (
        LTX2ConnectorArchConfig,
        LTX2ConnectorConfig,
    )
    from sglang.multimodal_gen.runtime import server_args as server_args_module
    from sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector import (
        LTX2TextConnectors,
        LTX2TransformerBlock1d,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.text_connector import (  # noqa: E501
        LTX2TextConnectorStage,
    )

    from sglang_rbln.patches.stages import text_encoding

    monkeypatch.setattr(
        server_args_module,
        "_global_server_args",
        SimpleNamespace(
            attention_backend="torch_sdpa",
            attention_backend_config=None,
            comfyui_mode=False,
            enable_torch_compile=True,
        ),
        raising=False,
    )
    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compile_module = torch.nn.Module.compile

    def compile_resident(module, *args, **kwargs):
        # as the server's residency hook does: the call takes its inputs to the NPUs
        compile_module(module, *args, **kwargs)
        residency.install_resident_call(module)

    monkeypatch.setattr(torch.nn.Module, "compile", compile_resident)
    transformer, forward = _connector_forward()
    monkeypatch.setattr(
        transformer,
        "forward",
        lambda self, *args, **kwargs: features.connector_transformer_forward(
            forward, self, *args, **kwargs
        ),
    )
    arch = LTX2ConnectorArchConfig(
        caption_channels=16,
        text_proj_in_factor=4,
        per_modality_projections=True,
        video_hidden_dim=64,
        audio_hidden_dim=32,
        video_connector_num_attention_heads=2,
        video_connector_attention_head_dim=32,
        audio_connector_num_attention_heads=2,
        audio_connector_attention_head_dim=16,
        video_connector_num_learnable_registers=64,
        audio_connector_num_learnable_registers=64,
        video_gated_attn=True,
        audio_gated_attn=True,
    )
    torch.manual_seed(0)
    connectors = LTX2TextConnectors(LTX2ConnectorConfig(arch_config=arch))
    connectors = connectors.to(torch.bfloat16).eval()
    stage = SimpleNamespace(connectors=connectors)
    mask = torch.tensor([[0] * 40 + [1] * 88]).to("rbln:0")
    server_args = SimpleNamespace(enable_torch_compile=True)

    eager_blocks = []
    block_forward = LTX2TransformerBlock1d.forward

    def recording_block_forward(self, *args, **kwargs):
        if not torch.compiler.is_compiling():
            eager_blocks.append(self)
        return block_forward(self, *args, **kwargs)

    monkeypatch.setattr(LTX2TransformerBlock1d, "forward", recording_block_forward)

    def encode():
        batch = SimpleNamespace(
            prompt_embeds=[torch.randn(1, 128, 64, dtype=torch.bfloat16).to("rbln:0")],
            prompt_attention_mask=[mask],
            negative_prompt_embeds=None,
            negative_attention_mask=None,
            do_classifier_free_guidance=False,
        )
        with torch.no_grad(), residency.DeviceBoundary():
            text_encoding.compile_connectors(
                LTX2TextConnectorStage.forward, stage, batch, server_args
            )
        return batch

    batch = encode()
    (video,), (audio,) = batch.prompt_embeds, batch.audio_prompt_embeds
    assert (video.device.type, tuple(video.shape)) == ("rbln", (1, 128, 64))
    assert (audio.device.type, tuple(audio.shape)) == ("rbln", (1, 128, 32))
    assert len(list(tmp_path.rglob("*.rbln"))) == 4
    encode()
    assert len(list(tmp_path.rglob("*.rbln"))) == 4
    assert eager_blocks == []

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.runtime.layers import layernorm as upstream_layernorm
from sglang.multimodal_gen.runtime.pipelines.cosmos3_pipeline import Cosmos3Pipeline
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln import device
from sglang_rbln.device import ExecutionMode
from sglang_rbln.patches.stages import cosmos3
from sglang_rbln.platform import RUNS_ON_DEVICE

pytestmark = pytest.mark.usefixtures("rbln_platform")

COSMOS3_IMPORT_PROBE = """
import importlib, json
from sglang.multimodal_gen.runtime import platforms
from sglang.multimodal_gen.runtime.platforms import plugins
plugins.load_plugins(); plugins.apply_plugin_hooks()
stages = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
cosmos3 = importlib.import_module(stages + ".cosmos3")
import sglang.multimodal_gen.runtime.models.dits.cosmos3video
import sglang.multimodal_gen.runtime.pipelines.cosmos3_pipeline
original = getattr(
    cosmos3.Cosmos3DenoisingStage._maybe_enable_torch_compile, "__wrapped__", None
)
print(json.dumps({
    "execution_mode": platforms.current_platform.execution_mode,
    "compile_hook_applied": original is not None
    and original.__qualname__.endswith("_maybe_enable_torch_compile"),
}))
"""


@pytest.mark.parametrize(
    "model_id",
    ["nvidia/Cosmos3-Nano", "nvidia/Cosmos3-Super", "nvidia/Cosmos3-Edge"],
    ids=["nano", "super", "edge"],
)
def test_cosmos3_ids_resolve_to_the_native_pipeline(model_id):
    assert registry.has_registered_diffusion_model_path(model_id)
    model_name = registry._MODEL_HF_PATH_TO_NAME[model_id]
    assert registry._CONFIG_REGISTRY[model_name].pipeline_config_cls is Cosmos3Config
    assert registry.get_pipeline_class(Cosmos3Pipeline.pipeline_name) is Cosmos3Pipeline


def test_cosmos3_tokenizes_inside_the_transformer():
    assert "text_encoder" not in Cosmos3Pipeline._required_config_modules
    assert {"transformer", "vae", "scheduler"} <= set(
        Cosmos3Pipeline._required_config_modules
    )


class _GenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, hidden):
        return self.proj(hidden)


@pytest.fixture
def transformer() -> nn.Module:
    module = nn.Module()
    module.head_dim = 128
    module.norm_moe_gen = nn.RMSNorm(8)
    module.gen_layers = nn.ModuleList([_GenLayer(), _GenLayer()])
    module.language_model = nn.Module()
    module.language_model.layers = nn.ModuleList([_GenLayer()])
    module.layer_names = ["gen_layers", "language_model.layers"]
    return module


@pytest.fixture
def recorded_compiles(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def fake_compile(module, **kwargs):
        calls.append(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    return calls


class TestGenLayerCompileHook:
    def test_registers_a_replace_hook(self):
        assert cosmos3.register() == (cosmos3.HOOK_TARGET, cosmos3.DECODE_HOOK_TARGET)
        registered = HookRegistry._hooks[cosmos3.HOOK_TARGET]
        assert any(hook is cosmos3.compile_layers for _, hook, _ in registered)

    def test_compiles_through_the_platform_backend(
        self, transformer, recorded_compiles
    ):
        cosmos3.compile_layers(
            None, transformer, SimpleNamespace(enable_torch_compile=True)
        )
        assert transformer._gen_layers_torch_compiled is True
        assert (128, True) in upstream_layernorm._QK_NORM_ROPE_DICT
        assert [call["backend"] for call in recorded_compiles] == ["rbln", "rbln"]
        assert all(call["dynamic"] is False for call in recorded_compiles)
        contexts = {
            id(call["options"]["compile_context"]) for call in recorded_compiles
        }
        assert len(contexts) == 1

    @pytest.mark.parametrize(
        ("execution_mode", "expected"),
        [
            (ExecutionMode.HOST, ["gen_layers"]),
            (ExecutionMode.DEVICE, ["gen_layers", "language_model.layers"]),
        ],
        ids=["host", "device"],
    )
    def test_language_model_compiles_only_on_device(
        self, monkeypatch, transformer, recorded_compiles, execution_mode, expected
    ):
        monkeypatch.setattr(device, "execution_mode", lambda: execution_mode)
        assert cosmos3.compiled_layer_names(transformer) == expected
        cosmos3.compile_layers(
            None, transformer, SimpleNamespace(enable_torch_compile=True)
        )
        assert len(recorded_compiles) == 2 + (execution_mode is ExecutionMode.DEVICE)

    def test_final_norm_runs_on_the_host(self, monkeypatch, transformer):
        hosted = []
        monkeypatch.setattr(cosmos3.cosmos3video, "run_on_host", hosted.append)
        cosmos3.compile_layers(
            None, transformer, SimpleNamespace(enable_torch_compile=False)
        )
        assert hosted == [transformer.norm_moe_gen]

    def test_respects_disabled_compile(self, transformer, recorded_compiles):
        cosmos3.compile_layers(
            None, transformer, SimpleNamespace(enable_torch_compile=False)
        )
        assert recorded_compiles == []
        assert not hasattr(transformer, "_gen_layers_torch_compiled")


class TestHostDecodeHook:
    @pytest.fixture
    def stage(self):
        vae = nn.Conv3d(4, 3, 1).to(torch.bfloat16)
        return SimpleNamespace(vae=vae, _latents_mean=None, _latents_std=None)

    @staticmethod
    def decode_latents(stage, latents):
        return stage.vae(latents)

    def test_registers_an_around_hook(self):
        cosmos3.register()
        assert any(
            hook is cosmos3.decode_latents_on_host
            for _, hook, _ in HookRegistry._hooks[cosmos3.DECODE_HOOK_TARGET]
        )

    def test_host_latents_pass_straight_through(self, stage):
        latents = torch.randn(1, 4, 1, 2, 2, dtype=torch.bfloat16)
        out = cosmos3.decode_latents_on_host(self.decode_latents, stage, latents)
        assert torch.equal(out, stage.vae(latents))

    @pytest.mark.xfail(not RUNS_ON_DEVICE, reason="execution mode is host")
    def test_device_latents_decode_on_the_cpu_and_return_to_the_device(self, stage):
        stage.vae.to("rbln:0")
        stage._latents_mean = torch.zeros(1)
        latents = torch.randn(1, 4, 1, 2, 2, dtype=torch.bfloat16).to("rbln:0")
        out = cosmos3.decode_latents_on_host(self.decode_latents, stage, latents)
        assert out.device == latents.device
        assert next(stage.vae.parameters()).device.type == "cpu"
        assert stage._latents_mean is None
        assert torch.equal(out.cpu(), stage.vae(latents.cpu()))


@pytest.mark.parametrize("execution_mode", [ExecutionMode.HOST, ExecutionMode.DEVICE])
def test_cosmos3_import_chain_applies_the_hook(run_probe, execution_mode):
    assert run_probe(COSMOS3_IMPORT_PROBE, execution_mode) == {
        "execution_mode": execution_mode,
        "compile_hook_applied": True,
    }

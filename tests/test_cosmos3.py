# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
from sglang.multimodal_gen.runtime.pipelines.cosmos3_pipeline import Cosmos3Pipeline
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln.device import ExecutionMode
from sglang_rbln.patches.stages import cosmos3

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
    module.gen_layers = nn.ModuleList([_GenLayer(), _GenLayer()])
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
        assert cosmos3.register() == cosmos3.HOOK_TARGET
        registered = HookRegistry._hooks[cosmos3.HOOK_TARGET]
        assert any(hook is cosmos3.compile_gen_layers for _, hook, _ in registered)

    def test_compiles_through_the_platform_backend(
        self, transformer, recorded_compiles
    ):
        cosmos3.compile_gen_layers(
            None, transformer, SimpleNamespace(enable_torch_compile=True)
        )
        assert transformer._gen_layers_torch_compiled is True
        assert [call["backend"] for call in recorded_compiles] == ["rbln", "rbln"]
        assert all(call["dynamic"] is False for call in recorded_compiles)
        contexts = {
            id(call["options"]["compile_context"]) for call in recorded_compiles
        }
        assert len(contexts) == 1

    def test_respects_disabled_compile(self, transformer, recorded_compiles):
        cosmos3.compile_gen_layers(
            None, transformer, SimpleNamespace(enable_torch_compile=False)
        )
        assert recorded_compiles == []
        assert not hasattr(transformer, "_gen_layers_torch_compiled")


@pytest.mark.parametrize("execution_mode", [ExecutionMode.HOST, ExecutionMode.DEVICE])
def test_cosmos3_import_chain_applies_the_hook(run_probe, execution_mode):
    assert run_probe(COSMOS3_IMPORT_PROBE, execution_mode) == {
        "execution_mode": execution_mode,
        "compile_hook_applied": True,
    }

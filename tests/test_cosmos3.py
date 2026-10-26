# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen import registry
from sglang.multimodal_gen.runtime.pipelines.cosmos3_pipeline import Cosmos3Pipeline
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln import device
from sglang_rbln.hooks import cosmos3_compile

COSMOS3_MODEL_IDS = [
    "nvidia/Cosmos3-Nano",
    "nvidia/Cosmos3-Super",
    "nvidia/Cosmos3-Edge",
]
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


@pytest.fixture(autouse=True)
def rbln_platform(fresh_platform, monkeypatch):
    monkeypatch.setenv(device.FORCE_NPU_NAME_ENV, "RBLN-CA25")
    monkeypatch.setenv("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "rbln")
    from sglang.multimodal_gen.runtime import platforms

    assert platforms.current_platform.device_name == "rbln"


@pytest.mark.parametrize("model_id", COSMOS3_MODEL_IDS)
def test_cosmos3_ids_resolve_to_the_native_pipeline(model_id):
    from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config

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


class TestGenLayerCompileHook:
    def test_registers_a_replace_hook(self):
        assert cosmos3_compile.register() == cosmos3_compile.HOOK_TARGET
        registered = HookRegistry._hooks[cosmos3_compile.HOOK_TARGET]
        assert any(
            hook is cosmos3_compile.compile_gen_layers for _, hook, _ in registered
        )

    def test_compiles_through_the_platform_backend(self, monkeypatch):
        compile_calls = []

        def fake_compile(module, **kwargs):
            compile_calls.append(kwargs)
            return module

        monkeypatch.setattr(torch, "compile", fake_compile)
        transformer = nn.Module()
        transformer.gen_layers = nn.ModuleList([_GenLayer(), _GenLayer()])
        server_args = SimpleNamespace(enable_torch_compile=True)
        cosmos3_compile.compile_gen_layers(None, transformer, server_args)
        assert len(compile_calls) == 2
        assert transformer._gen_layers_torch_compiled is True
        for kwargs in compile_calls:
            assert kwargs["backend"] == "rbln"
            assert kwargs["dynamic"] is False
            assert "compile_context" in kwargs["options"]
        assert (
            compile_calls[0]["options"]["compile_context"]
            is compile_calls[1]["options"]["compile_context"]
        )

    def test_respects_disabled_compile(self, monkeypatch):
        monkeypatch.setattr(
            torch, "compile", lambda *a, **k: pytest.fail("compiled while disabled")
        )
        transformer = nn.Module()
        transformer.gen_layers = nn.ModuleList([_GenLayer()])
        cosmos3_compile.compile_gen_layers(
            None, transformer, SimpleNamespace(enable_torch_compile=False)
        )
        assert not hasattr(transformer, "_gen_layers_torch_compiled")


@pytest.mark.parametrize("execution_mode", ["host", "device"])
def test_cosmos3_import_chain_applies_the_hook(execution_mode):
    probe_env = {
        **os.environ,
        device.FORCE_NPU_NAME_ENV: "RBLN-CA25",
        device.EXECUTION_MODE_ENV: execution_mode,
        "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "rbln",
        "PYTHONWARNINGS": "ignore",
    }
    probe_run = subprocess.run(
        [sys.executable, "-c", COSMOS3_IMPORT_PROBE],
        env=probe_env,
        capture_output=True,
        text=True,
    )
    assert probe_run.returncode == 0, probe_run.stderr[-2000:]
    report = json.loads(probe_run.stdout.strip().splitlines()[-1])
    assert report == {"execution_mode": execution_mode, "compile_hook_applied": True}

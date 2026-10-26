# SPDX-License-Identifier: Apache-2.0
import pytest
from sglang.multimodal_gen import registry
from sglang.multimodal_gen.runtime.pipelines.cosmos3_pipeline import Cosmos3Pipeline

from sglang_rbln import device

COSMOS3_IDS = [
    "nvidia/Cosmos3-Nano",
    "nvidia/Cosmos3-Super",
    "nvidia/Cosmos3-Edge",
]


@pytest.fixture(autouse=True)
def rbln_platform(fresh_platform, monkeypatch):
    monkeypatch.setenv(device.FORCE_NPU_NAME_ENV, "RBLN-CA25")
    monkeypatch.setenv("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "rbln")
    from sglang.multimodal_gen.runtime import platforms

    assert platforms.current_platform.device_name == "rbln"


@pytest.mark.parametrize("model_id", COSMOS3_IDS)
def test_cosmos3_ids_are_registered(model_id):
    assert registry.has_registered_diffusion_model_path(model_id)


def test_cosmos3_resolves_to_the_native_pipeline():
    registry._ensure_registry_initialized()
    assert registry.get_pipeline_class(Cosmos3Pipeline.pipeline_name) is Cosmos3Pipeline


def test_cosmos3_pipeline_declares_its_modules():
    assert "text_encoder" not in Cosmos3Pipeline._required_config_modules
    assert {"transformer", "vae", "scheduler"} <= set(
        Cosmos3Pipeline._required_config_modules
    )

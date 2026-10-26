# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys
from types import ModuleType

import pytest
from sglang.multimodal_gen import registry
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import LTX2Pipeline

from sglang_rbln import device
from sglang_rbln.hooks import triton_norm

MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
NORM_TRITON = "sglang.kernels.ops.diffusion.norm.norm_triton"


@pytest.fixture(autouse=True)
def rbln_platform(fresh_platform, monkeypatch):
    monkeypatch.setenv(device.FORCE_NPU_NAME_ENV, "RBLN-CA25")
    monkeypatch.setenv("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "rbln")
    from sglang.multimodal_gen.runtime import platforms

    assert platforms.current_platform.device_name == "rbln"


def test_ltx25_resolves_to_the_native_pipeline():
    from sglang.multimodal_gen.configs.pipeline_configs.ltx_2_5 import (
        LTX25PipelineConfig,
    )

    assert registry.has_registered_diffusion_model_path(MODEL_ID)
    name = registry._MODEL_HF_PATH_TO_NAME[MODEL_ID]
    assert registry._CONFIG_REGISTRY[name].pipeline_config_cls is LTX25PipelineConfig
    assert registry.get_pipeline_class(LTX2Pipeline.pipeline_name) is LTX2Pipeline


def test_ltx25_pipeline_declares_audio_modules():
    assert {"transformer", "vae", "audio_vae", "vocoder", "connectors"} <= set(
        LTX2Pipeline._required_config_modules
    )


class TestTritonNormSeed:
    def test_binds_only_missing_names(self):
        module = ModuleType("fake")
        module.rms_norm_fn = real = object()
        assert triton_norm.seed(module) == ["norm_infer"]
        assert module.rms_norm_fn is real
        with pytest.raises(NotImplementedError):
            module.norm_infer()

    def test_is_idempotent(self):
        module = ModuleType("fake")
        assert triton_norm.seed(module) == list(triton_norm.NAMES)
        assert triton_norm.seed(module) == []


IMPORT_PROBE = """
import json, sys
from sglang.multimodal_gen.runtime import platforms
from sglang.multimodal_gen.runtime.platforms import plugins
plugins.load_plugins(); plugins.apply_plugin_hooks()
import sglang.multimodal_gen.runtime.models.dits.ltx_2
import sglang.multimodal_gen.runtime.models.vaes.ltx_2_audio
import sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline
import sglang.kernels.ops.diffusion as kd
try:
    kd.norm_infer(); raise SystemExit("norm_infer did not raise")
except NotImplementedError:
    pass
print(json.dumps({
    "platform": type(platforms.current_platform).__name__,
    "norm_triton_imported": __NORM_TRITON__ in sys.modules,
}))
"""


def test_ltx25_imports_without_the_triton_norms():
    code = IMPORT_PROBE.replace("__NORM_TRITON__", repr(NORM_TRITON))
    env = {
        **os.environ,
        device.FORCE_NPU_NAME_ENV: "RBLN-CA25",
        "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "rbln",
        "PYTHONWARNINGS": "ignore",
    }
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {"platform": "RblnPlatform", "norm_triton_imported": False}

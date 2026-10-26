# SPDX-License-Identifier: Apache-2.0
import json
import os
import subprocess
import sys

import pytest
from sglang.multimodal_gen import registry
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import LTX2Pipeline

from sglang_rbln import device

LTX25_MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
NORM_TRITON_MODULE = "sglang.kernels.ops.diffusion.norm.norm_triton"
LTX25_IMPORT_PROBE = """
import json, sys
from sglang.multimodal_gen.runtime import platforms
from sglang.multimodal_gen.runtime.platforms import plugins
plugins.load_plugins(); plugins.apply_plugin_hooks()
import sglang.multimodal_gen.runtime.entrypoints.cli.serve
import sglang.multimodal_gen.runtime.models.dits.ltx_2
import sglang.multimodal_gen.runtime.models.vaes.ltx_2_audio
import sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline
import sglang.kernels.ops.diffusion as diffusion_exports
from sglang.multimodal_gen.runtime.layers.attention.backends import video_sparse_attn
try:
    diffusion_exports.norm_infer(); raise SystemExit("norm_infer did not raise")
except NotImplementedError:
    pass
print(json.dumps({
    "platform": type(platforms.current_platform).__name__,
    "execution_mode": platforms.current_platform.execution_mode,
    "norm_triton_imported": __NORM_TRITON_MODULE__ in sys.modules,
    "vsa_kernel_bound": video_sparse_attn.video_sparse_attn is not None,
}))
"""


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

    assert registry.has_registered_diffusion_model_path(LTX25_MODEL_ID)
    model_name = registry._MODEL_HF_PATH_TO_NAME[LTX25_MODEL_ID]
    config_info = registry._CONFIG_REGISTRY[model_name]
    assert config_info.pipeline_config_cls is LTX25PipelineConfig
    assert registry.get_pipeline_class(LTX2Pipeline.pipeline_name) is LTX2Pipeline


def test_ltx25_pipeline_declares_audio_modules():
    assert {"transformer", "vae", "audio_vae", "vocoder", "connectors"} <= set(
        LTX2Pipeline._required_config_modules
    )


@pytest.mark.parametrize("execution_mode", ["host", "device"])
def test_serve_import_chain_loads_without_cuda(execution_mode):
    probe = LTX25_IMPORT_PROBE.replace(
        "__NORM_TRITON_MODULE__", repr(NORM_TRITON_MODULE)
    )
    probe_env = {
        **os.environ,
        device.FORCE_NPU_NAME_ENV: "RBLN-CA25",
        device.EXECUTION_MODE_ENV: execution_mode,
        "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "rbln",
        "PYTHONWARNINGS": "ignore",
    }
    probe_run = subprocess.run(
        [sys.executable, "-c", probe], env=probe_env, capture_output=True, text=True
    )
    assert probe_run.returncode == 0, probe_run.stderr[-2000:]
    report = json.loads(probe_run.stdout.strip().splitlines()[-1])
    assert report == {
        "platform": "RblnPlatform",
        "execution_mode": execution_mode,
        "norm_triton_imported": False,
        "vsa_kernel_bound": False,
    }

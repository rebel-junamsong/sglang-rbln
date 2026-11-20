# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2_5 import LTX25PipelineConfig
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import LTX2Pipeline

pytestmark = pytest.mark.usefixtures("rbln_platform")

LTX25_MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
NORM_TRITON_MODULE = "sglang.kernels.ops.diffusion.norm.norm_triton"
LTX25_IMPORT_PROBE = f"""
import json, sys
from sglang.multimodal_gen.runtime import platforms
from sglang.multimodal_gen.runtime.platforms import plugins
plugins.load_plugins(); plugins.apply_plugin_hooks()
import sglang.multimodal_gen.runtime.entrypoints.cli.serve
import sglang.multimodal_gen.runtime.models.dits.ltx_2
import sglang.multimodal_gen.runtime.models.vaes.ltx_2_audio
import sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline
from sglang.multimodal_gen.runtime.layers.attention.backends import video_sparse_attn
print(json.dumps({{
    "platform": type(platforms.current_platform).__name__,
    "device_type": platforms.current_platform.device_type,
    "norm_triton_imported": {NORM_TRITON_MODULE!r} in sys.modules,
    "vsa_kernel_bound": video_sparse_attn.video_sparse_attn is not None,
}}))
"""


def test_ltx25_resolves_to_the_native_pipeline():
    assert registry.has_registered_diffusion_model_path(LTX25_MODEL_ID)
    model_name = registry._MODEL_HF_PATH_TO_NAME[LTX25_MODEL_ID]
    config_info = registry._CONFIG_REGISTRY[model_name]
    assert config_info.pipeline_config_cls is LTX25PipelineConfig
    assert registry.get_pipeline_class(LTX2Pipeline.pipeline_name) is LTX2Pipeline


def test_ltx25_pipeline_declares_audio_modules():
    assert {"transformer", "vae", "audio_vae", "vocoder", "connectors"} <= set(
        LTX2Pipeline._required_config_modules
    )


def test_serve_import_chain_loads_without_cuda(run_probe):
    assert run_probe(LTX25_IMPORT_PROBE) == {
        "platform": "RBLNPlatform",
        "device_type": "cpu",
        "norm_triton_imported": False,
        "vsa_kernel_bound": False,
    }

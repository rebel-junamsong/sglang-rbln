# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import pytest

from sglang_rbln import device

STAGES = "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"


@dataclass(frozen=True, slots=True)
class PatchDebt:
    patch: str
    module: str
    upstream_pattern: str
    upstream_fix: str


PATCH_DEBTS = [
    PatchDebt(
        patch="configs/arch_config",
        module="sglang.multimodal_gen.configs.models.base",
        upstream_pattern='d = object.__getattribute__(self, "__dict__")',
        upstream_fix="read extra_attrs through self.__dict__ so dynamo can trace it",
    ),
    PatchDebt(
        patch="layers/layernorm",
        module="sglang.multimodal_gen.runtime.layers.layernorm",
        upstream_pattern=(
            "if not _is_cpu:\n"
            "    from sglang.kernels.ops.diffusion import norm_infer, rms_norm_fn"
        ),
        upstream_fix="gate the Triton norm import on has_triton()",
    ),
    PatchDebt(
        patch="layers/attention/video_sparse_attn",
        module="sglang.multimodal_gen.runtime.layers.attention.backends"
        ".video_sparse_attn",
        upstream_pattern=(
            "try:\n    from vsa import video_sparse_attn\nexcept ImportError:"
        ),
        upstream_fix="also catch the CUDA RuntimeError vsa raises at import",
    ),
    PatchDebt(
        patch="srt/layers/quantization/fp4_utils",
        module="sglang.srt.layers.quantization.fp4_utils",
        upstream_pattern=(
            "    from flashinfer import fp4_quantize as _flashinfer_fp4_quantize"
        ),
        upstream_fix="catch the AssertionError flashinfer raises without CUDA",
    ),
    PatchDebt(
        patch="models/ltx_2",
        module="sglang.multimodal_gen.runtime.models.dits.ltx_2",
        upstream_pattern=(
            "cos_freqs = torch.swapaxes(cos_freq, 1, 2)\n"
            "            sin_freqs = torch.swapaxes(sin_freq, 1, 2)"
        ),
        upstream_fix="return the rotary cos/sin tables contiguous",
    ),
    PatchDebt(
        patch="models/ltx_2 (fused ada values)",
        module="sglang.multimodal_gen.runtime.models.dits.ltx_2",
        upstream_pattern="or not current_platform.tensor_on_device(timestep)",
        upstream_fix="gate the Triton ada-values kernel on Triton, not on the device",
    ),
    PatchDebt(
        patch="models/ltx_2 (connector rope)",
        module="sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector",
        upstream_pattern=(
            "cos_freqs = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)\n"
            "            sin_freqs = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)"
        ),
        upstream_fix="return the connector rotary cos/sin tables contiguous",
    ),
    PatchDebt(
        patch="models/ltx_2_text_features",
        module="sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector",
        upstream_pattern="hidden_states[i, binary_attn_mask[i].bool(), :]",
        upstream_fix="replace the padded tokens over a fixed length",
    ),
    PatchDebt(
        patch="models/ltx_2_vae_assembly",
        module="sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae",
        upstream_pattern="            result_rows.append(torch.cat(result_row, dim=4))",
        upstream_fix="join the decoded tiles on the device that decoded them",
    ),
    PatchDebt(
        patch="stages/ltx_2/vocoder",
        module=f"{STAGES}.ltx_2.decoding_av",
        upstream_pattern="                    waveform = self.vocoder(spectrogram)",
        upstream_fix="compile the vocoder through the platform backend",
    ),
    PatchDebt(
        patch="stages/text_encoding",
        module="sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding",
        upstream_pattern="class TextEncodingStage(ConditionEncodingStage):",
        upstream_fix="compile the text encoder layer stacks like DenoisingStage does",
    ),
    PatchDebt(
        patch="stages/text_encoding (capture hooks)",
        module="transformers.utils.output_capturing",
        upstream_pattern="        collected_outputs = _active_collector.get()",
        upstream_fix="read the output collector in a way dynamo can trace",
    ),
    PatchDebt(
        patch="stages/text_encoding (ltx_2 connectors)",
        module=f"{STAGES}.ltx_2.text_connector",
        upstream_pattern="class LTX2TextConnectorStage(PipelineStage):",
        upstream_fix="compile the connector layer stacks like DenoisingStage does",
    ),
    PatchDebt(
        patch="models/ltx_2 (diffusion decoder keyframe tag)",
        module="sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder",
        upstream_pattern=(
            "self.conv_in = nn.Linear(arch.latent_channels, stage_channels[0], "
            "bias=True)\n\n        self.det_stages = nn.ModuleList()"
        ),
        upstream_fix="declare the checkpoint's type_emb (sglang 0.5.22)",
    ),
    PatchDebt(
        patch="models/ltx_2_tile_ranks",
        module="sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder",
        upstream_pattern="world_size = get_decode_parallel_world_size()",
        upstream_fix="let a platform supply decode-parallel ranks inside one process",
    ),
    PatchDebt(
        patch="stages/ltx_2/decoding_av",
        module=f"{STAGES}.ltx_2.decoding_av",
        upstream_pattern="decode_output = vae.decode(latents",
        upstream_fix="compile the video decoder like DenoisingStage compiles the DiT",
    ),
    PatchDebt(
        patch="stages/ltx_2/image_encoding",
        module="sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding",
        upstream_pattern="latent_dist = vae.encode(video_condition)",
        upstream_fix="compile the video encoder like DenoisingStage compiles the DiT",
    ),
    PatchDebt(
        patch="stages/ltx_2/duration",
        module=f"{STAGES}.ltx_2.duration",
        upstream_pattern="num_frames = duration_head.predict_num_frames(",
        upstream_fix="compile the duration head through the platform backend",
    ),
    PatchDebt(
        patch="stages/ltx_2/upsampling",
        module=f"{STAGES}.ltx_2.upsampling",
        upstream_pattern="latents = spatial_upsampler(latents)",
        upstream_fix="compile the latent upsampler through the platform backend",
    ),
    pytest.param(
        PatchDebt(
            patch="torch_rbln/ops_utils",
            module="torch_rbln._internal.ops_utils",
            upstream_pattern=(
                "promoted_dtype = torch.promote_types(promoted_dtype, tensor.dtype)"
            ),
            upstream_fix="promote like torch.result_type (zero-dim tensors do not"
            " raise the result dtype within a category)",
        ),
        id="torch_rbln/ops_utils",
        marks=pytest.mark.xfail(
            not device.torch_rbln_available(), reason="torch-rbln is not installed"
        ),
    ),
]


def _upstream_source(module: str) -> str:
    spec = find_spec(module)
    assert spec is not None and spec.origin, f"{module} is not installed"
    return Path(spec.origin).read_text()


@pytest.mark.parametrize(
    "debt",
    PATCH_DEBTS,
    ids=lambda debt: debt.patch if isinstance(debt, PatchDebt) else None,
)
def test_upstream_still_needs_the_patch(debt: PatchDebt):
    assert debt.upstream_pattern in _upstream_source(debt.module), (
        f"sglang no longer has the code patches/{debt.patch} works around "
        f"({debt.upstream_fix}); delete the patch"
    )

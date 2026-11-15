# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import pytest

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
        patch="utils/precision",
        module="sglang.multimodal_gen.runtime.utils.precision",
        upstream_pattern=(
            "return torch.autocast(\n        device_type=current_platform.device_type,"
        ),
        upstream_fix="let the platform name every device type autocast covers",
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

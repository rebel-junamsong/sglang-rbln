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
        patch="layers/layernorm",
        module="sglang.multimodal_gen.runtime.layers.layernorm",
        upstream_pattern=(
            "if not _is_cpu:\n"
            "    from sglang.kernels.ops.diffusion import norm_infer, rms_norm_fn"
        ),
        upstream_fix="gate the Triton norm import on has_triton()",
    ),
    PatchDebt(
        patch="layers/layernorm (qk-norm rope)",
        module="sglang.multimodal_gen.runtime.layers.layernorm",
        upstream_pattern=(
            "    rotary_emb = _QK_NORM_ROPE_DICT.get(rope_key)\n"
            "    if rotary_emb is None:"
        ),
        upstream_fix="build the QK-norm RotaryEmbedding outside the traced call",
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
        patch="models/ltx_2 (connector rope)",
        module="sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector",
        upstream_pattern=(
            "cos_freqs = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)\n"
            "            sin_freqs = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)"
        ),
        upstream_fix="return the connector rotary cos/sin tables contiguous",
    ),
    PatchDebt(
        patch="stages/ltx_2/vocoder",
        module=f"{STAGES}.ltx_2.decoding_av",
        upstream_pattern="                    waveform = self.vocoder(spectrogram)",
        upstream_fix="compile the vocoder through the platform backend",
    ),
    PatchDebt(
        patch="stages/cosmos3",
        module=f"{STAGES}.cosmos3",
        upstream_pattern='compile_kwargs = {\n                "mode": "default",',
        upstream_fix="build the kwargs through resolve_torch_compile_kwargs",
    ),
    PatchDebt(
        patch="stages/cosmos3 (decode)",
        module=f"{STAGES}.cosmos3",
        upstream_pattern="        video = self.vae.decode(latents)",
        upstream_fix="let the platform choose the VAE decode device",
    ),
    PatchDebt(
        patch="models/cosmos3video",
        module="sglang.multimodal_gen.runtime.models.dits.cosmos3video",
        upstream_pattern="        t_scaled = t * self.timestep_scale",
        upstream_fix="let the platform run float32 timestep embeddings on the host",
    ),
    PatchDebt(
        patch="models/cosmos3video (qk-norm rope)",
        module="sglang.multimodal_gen.runtime.models.dits.cosmos3video",
        upstream_pattern="use_fused_qk_norm_rope = T > 1 or (",
        upstream_fix="take the split QK-norm RoPE path when no fused kernel exists",
    ),
    PatchDebt(
        patch="models/cosmos3video (text kv)",
        module="sglang.multimodal_gen.runtime.models.dits.cosmos3video",
        upstream_pattern="        return out, k_gen, v",
        upstream_fix="return the cached text K/V with one layout on every request",
    ),
    PatchDebt(
        patch="models/cosmos3video (rope)",
        module="sglang.multimodal_gen.runtime.models.dits.cosmos3video",
        upstream_pattern=(
            "    q_out = torch.empty_like(q)\n"
            "    q_out[..., :half] = q1 * cos - q2 * sin"
        ),
        upstream_fix="build the rotated halves with torch.cat, not slice assignment",
    ),
    PatchDebt(
        patch="models/cosmos3video (position ids)",
        module="sglang.multimodal_gen.runtime.models.dits.cosmos3video",
        upstream_pattern="            real_len = int(text_lengths[b].item())",
        upstream_fix="compute the mRoPE offsets as tensors instead of with .item()",
    ),
    PatchDebt(
        patch="stages/cosmos3 (final norm)",
        module="sglang.multimodal_gen.runtime.models.dits.cosmos3video",
        upstream_pattern="        hidden_gen = self.norm_moe_gen(hidden_gen)",
        upstream_fix="let the platform run float32 norms on the host",
    ),
    PatchDebt(
        patch="models/wanvae",
        module="sglang.multimodal_gen.runtime.models.vaes.wanvae",
        upstream_pattern='feat_cache = contextvars.ContextVar("feat_cache"',
        upstream_fix="keep the Wan feature cache on the module instead of ContextVars",
    ),
    PatchDebt(
        patch="stages/flux3_action",
        module=f"{STAGES}.flux3_action",
        upstream_pattern="class Flux3ActionDenoisingStage(PipelineStage):",
        upstream_fix="compile each denoise step as one graph",
    ),
    PatchDebt(
        patch="stages/flux3_action (observation)",
        module=f"{STAGES}.flux3_action",
        upstream_pattern="class Flux3ActionObservationEncodingStage(PipelineStage):",
        upstream_fix="compile the conditioning stream encoder as one graph",
    ),
    PatchDebt(
        patch="stages/flux3_action_step",
        module=f"{STAGES}.flux3_action",
        upstream_pattern=(
            "                def denoise(ctx: Flux3SegmentState)"
            " -> tuple[torch.Tensor, ...]:"
        ),
        upstream_fix="run each denoise step as a tensor-only _denoise_step",
    ),
    PatchDebt(
        patch="stages/flux3_action_step (observation)",
        module=f"{STAGES}.flux3_action",
        upstream_pattern='            state["flux3_conditioning"] = [',
        upstream_fix="encode the conditioning streams in a tensor-only _encode_streams",
    ),
    PatchDebt(
        patch="stages/flux3_action (text)",
        module=f"{STAGES}.flux3_action",
        upstream_pattern="class Flux3ActionTextEncodingStage(PipelineStage):",
        upstream_fix="compile the text context encoder as one graph",
    ),
    PatchDebt(
        patch="models/flux3 (segment state)",
        module="sglang.multimodal_gen.runtime.models.dits.flux3",
        upstream_pattern="class Flux3SegmentState(msgspec.Struct, frozen=True):",
        upstream_fix="use a segment state torch.compile can build (a NamedTuple)",
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
        patch="stages/flux3_action_video (request)",
        module="sglang.multimodal_gen.runtime.entrypoints.action.protocol",
        upstream_pattern=(
            '    for name in ("seed", "guidance_scale", "guidance_scale_action"):'
        ),
        upstream_fix="forward parameters.return_video to the sampling params",
    ),
    PatchDebt(
        patch="stages/flux3_action_video (decoder)",
        module="sglang.multimodal_gen.configs.pipeline_configs.flux3_action",
        upstream_pattern=(
            "default_factory=lambda: Flux3VideoVAEConfig(load_decoder=False)"
        ),
        upstream_fix="load the video VAE decoder when the pipeline returns video",
    ),
    PatchDebt(
        patch="stages/flux3_action_video (observation latent)",
        module=f"{STAGES}.flux3_action",
        upstream_pattern=(
            "            latent = self.vae.encode_frame(frame)[..., :h, :w]\n"
            "        video, video_ids = pack_video("
        ),
        upstream_fix="keep the observation latent in the VLA state",
    ),
    PatchDebt(
        patch="stages/flux3_action_video (denoised video)",
        module=f"{STAGES}.flux3_action",
        upstream_pattern=(
            "targets = result[cfg.action_modality][0].float() / cfg.action_scale"
        ),
        upstream_fix="keep the denoised video latent in the VLA state",
    ),
    PatchDebt(
        patch="stages/flux3_action_video (content region)",
        module="sglang.multimodal_gen.configs.pipeline_configs.flux3_action",
        upstream_pattern="            content = (540, 640)",
        upstream_fix="expose the canvas content region on the pipeline config",
    ),
    PatchDebt(
        patch="stages/flux3_action_video (decode)",
        module="sglang.multimodal_gen.runtime.pipelines_core.stages.vla",
        upstream_pattern=(
            "        payload = {\n"
            '            "request_id": batch.request_id,\n'
            '            "actions": actions_out,\n'
            "        }"
        ),
        upstream_fix="decode the predicted video in a stage when the request asks",
    ),
    PatchDebt(
        patch="stages/flux3_action_video (response)",
        module="sglang.multimodal_gen.runtime.entrypoints.action.protocol",
        upstream_pattern=(
            '        for name in ("action_mode", "domain_id", "raw_action_dim"):'
        ),
        upstream_fix="return the predicted video in the action envelope",
    ),
    PatchDebt(
        patch="models/flux3_video_vae (attention)",
        module="sglang.multimodal_gen.runtime.models.vaes.flux3_video_vae",
        upstream_pattern=(
            "out = neighborhood_attention(q2, k2, v2, kernel_size=kernel)"
        ),
        upstream_fix="let the platform choose the neighborhood attention kernel",
    ),
    PatchDebt(
        patch="models/flux3_video_vae (compile)",
        module="sglang.multimodal_gen.runtime.pipelines.flux3_action",
        upstream_pattern=(
            'return vae.to(resolve_component_precision(server_args, "vae"))'
        ),
        upstream_fix="compile the video VAE blocks like DenoisingStage does",
    ),
    PatchDebt(
        patch="stages/ltx_2/decoding_av",
        module=f"{STAGES}.ltx_2.decoding_av",
        upstream_pattern="decode_output = vae.decode(latents",
        upstream_fix="compile the video decoder like DenoisingStage compiles the DiT",
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

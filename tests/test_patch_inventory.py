# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import pytest
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import patches

RUNTIME = "sglang.multimodal_gen.runtime"
CORE_STAGES = f"{RUNTIME}.pipelines_core.stages"
STAGES = f"{CORE_STAGES}.model_specific_stages"
VAE = f"{RUNTIME}.models.vaes.ltx_2_vae"
RESIDENCY = f"{RUNTIME}.managers.memory_managers.component_residency_strategies"


@dataclass(frozen=True, slots=True)
class PatchDebt:
    """An upstream line a patch works around. `target` is the hook the patch
    registers on it, None for a workaround that is not a hook."""

    patch: str
    target: str | None
    module: str
    upstream_pattern: str
    upstream_fix: str


PATCH_DEBTS = [
    PatchDebt(
        patch="configs/arch_config",
        target="sglang.multimodal_gen.configs.models.base.ArchConfig.__getattr__",
        module="sglang.multimodal_gen.configs.models.base",
        upstream_pattern='d = object.__getattribute__(self, "__dict__")',
        upstream_fix="read extra_attrs through self.__dict__ so dynamo can trace it",
    ),
    PatchDebt(
        patch="utils/precision",
        target=f"{RUNTIME}.utils.precision.autocast_context",
        module="sglang.multimodal_gen.runtime.utils.precision",
        upstream_pattern=(
            "return torch.autocast(\n        device_type=current_platform.device_type,"
        ),
        upstream_fix="let the platform name every device type autocast covers",
    ),
    PatchDebt(
        patch="layers/attention/video_sparse_attn",
        target=None,
        module="sglang.multimodal_gen.runtime.layers.attention.backends"
        ".video_sparse_attn",
        upstream_pattern=(
            "try:\n    from vsa import video_sparse_attn\nexcept ImportError:"
        ),
        upstream_fix="also catch the CUDA RuntimeError vsa raises at import",
    ),
    PatchDebt(
        patch="models/ltx_2",
        target=f"{RUNTIME}.models.dits.ltx_2.LTX2AudioVideoRotaryPosEmbed.forward",
        module="sglang.multimodal_gen.runtime.models.dits.ltx_2",
        upstream_pattern=(
            "cos_freqs = torch.swapaxes(cos_freq, 1, 2)\n"
            "            sin_freqs = torch.swapaxes(sin_freq, 1, 2)"
        ),
        upstream_fix="return the rotary cos/sin tables contiguous",
    ),
    PatchDebt(
        patch="models/ltx_2 (fused ada values)",
        target=f"{RUNTIME}.models.dits.ltx_2._ltx2_try_fused_ada_values9",
        module="sglang.multimodal_gen.runtime.models.dits.ltx_2",
        upstream_pattern="or not current_platform.tensor_on_device(timestep)",
        upstream_fix="gate the Triton ada-values kernel on Triton, not on the device",
    ),
    PatchDebt(
        patch="models/ltx_2 (connector rope)",
        target=f"{RUNTIME}.models.adapter.ltx_2_connector.LTX2RotaryPosEmbed1d.forward",
        module="sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector",
        upstream_pattern=(
            "cos_freqs = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)\n"
            "            sin_freqs = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)"
        ),
        upstream_fix="return the connector rotary cos/sin tables contiguous",
    ),
    PatchDebt(
        patch="models/ltx_2_text_features",
        target=f"{RUNTIME}.models.adapter.ltx_2_connector.LTX2ConnectorTransformer1d.forward",
        module="sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector",
        upstream_pattern="hidden_states[i, binary_attn_mask[i].bool(), :]",
        upstream_fix="replace the padded tokens over a fixed length",
    ),
    PatchDebt(
        patch="models/ltx_2_vae_assembly",
        target=f"{VAE}.AutoencoderKLLTX2Video.tiled_decode",
        module="sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae",
        upstream_pattern="            result_rows.append(torch.cat(result_row, dim=4))",
        upstream_fix="join the decoded tiles on the device that decoded them",
    ),
    PatchDebt(
        patch="stages/ltx_2/vocoder",
        target=f"{STAGES}.ltx_2.decoding_av.LTX2AVDecodingStage.forward",
        module=f"{STAGES}.ltx_2.decoding_av",
        upstream_pattern="                    waveform = self.vocoder(spectrogram)",
        upstream_fix="compile the vocoder through the platform backend",
    ),
    PatchDebt(
        patch="stages/text_encoding",
        target=f"{CORE_STAGES}.text_encoding.TextEncodingStage.forward",
        module="sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding",
        upstream_pattern="class TextEncodingStage(ConditionEncodingStage):",
        upstream_fix="compile the text encoder layer stacks like DenoisingStage does",
    ),
    PatchDebt(
        patch="stages/text_encoding (capture hooks)",
        target=None,
        module="transformers.utils.output_capturing",
        upstream_pattern="        collected_outputs = _active_collector.get()",
        upstream_fix="read the output collector in a way dynamo can trace",
    ),
    PatchDebt(
        patch="stages/text_encoding (ltx_2 connectors)",
        target=f"{STAGES}.ltx_2.text_connector.LTX2TextConnectorStage.forward",
        module=f"{STAGES}.ltx_2.text_connector",
        upstream_pattern="class LTX2TextConnectorStage(PipelineStage):",
        upstream_fix="compile the connector layer stacks like DenoisingStage does",
    ),
    PatchDebt(
        patch="models/ltx_2 (diffusion decoder keyframe tag)",
        target=(
            f"{RUNTIME}.models.decoders.ltx_2_5_diffusion_decoder."
            "LTX2VideoDiffusionDecoder3d.__init__"
        ),
        module="sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder",
        upstream_pattern=(
            "self.conv_in = nn.Linear(arch.latent_channels, stage_channels[0], "
            "bias=True)\n\n        self.det_stages = nn.ModuleList()"
        ),
        upstream_fix="declare the checkpoint's type_emb (sglang 0.5.22)",
    ),
    PatchDebt(
        patch="stages/ltx_2/decoding_av",
        target=f"{STAGES}.ltx_2.decoding_av.LTX2AVDecodingStage.forward",
        module=f"{STAGES}.ltx_2.decoding_av",
        upstream_pattern="decode_output = vae.decode(latents",
        upstream_fix="compile the video decoder like DenoisingStage compiles the DiT",
    ),
    PatchDebt(
        patch="stages/ltx_2/image_encoding",
        target=f"{CORE_STAGES}.image_encoding.LTX2ImageEncodingStage.forward",
        module="sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding",
        upstream_pattern="latent_dist = vae.encode(video_condition)",
        upstream_fix="compile the video encoder like DenoisingStage compiles the DiT",
    ),
    PatchDebt(
        patch="stages/ltx_2/duration",
        target=f"{STAGES}.ltx_2.duration.LTX2DurationStage.forward",
        module=f"{STAGES}.ltx_2.duration",
        upstream_pattern="num_frames = duration_head.predict_num_frames(",
        upstream_fix="compile the duration head through the platform backend",
    ),
    PatchDebt(
        patch="stages/ltx_2/upsampling",
        target=f"{STAGES}.ltx_2.upsampling.LTX2UpsampleStage._upsample_video_latents",
        module=f"{STAGES}.ltx_2.upsampling",
        upstream_pattern="latents = spatial_upsampler(latents)",
        upstream_fix="compile the latent upsampler through the platform backend",
    ),
    PatchDebt(
        patch="models/ltx_2_text_features (packing)",
        target="sglang.multimodal_gen.configs.pipeline_configs.ltx_2._gemma_postprocess_func",
        module="sglang.multimodal_gen.configs.pipeline_configs.ltx_2",
        upstream_pattern="hidden_states = torch.stack(outputs.hidden_states, dim=-1)",
        upstream_fix="pack the hidden states in one graph where they live",
    ),
    PatchDebt(
        patch="models/ltx_2_text_features (features device)",
        target=f"{CORE_STAGES}.text_encoding.TextEncodingStage.encode_text",
        module=f"{CORE_STAGES}.text_encoding",
        upstream_pattern=(
            "target_device = device if device is not None else get_local_torch_device()"
        ),
        upstream_fix="return the features on the device of the module that reads them",
    ),
    PatchDebt(
        patch="stages/text_encoding (host inputs)",
        target=f"{CORE_STAGES}.text_encoding.TextEncodingStage._forward_text_encoder",
        module=f"{CORE_STAGES}.text_encoding",
        upstream_pattern=(
            "text_inputs = host_inputs.to(target_device, non_blocking=True)"
        ),
        upstream_fix=(
            "give the encoder its inputs on the encoder's device, not the output's"
        ),
    ),
    PatchDebt(
        patch="models/ltx_2 (blend_h)",
        target=f"{VAE}.AutoencoderKLLTX2Video.blend_h",
        module=VAE,
        upstream_pattern=(
            "for x in range(blend_extent):\n            b[:, :, :, :, x] ="
        ),
        upstream_fix="blend the tiles in whole-slab ops",
    ),
    PatchDebt(
        patch="models/ltx_2 (blend_v)",
        target=f"{VAE}.AutoencoderKLLTX2Video.blend_v",
        module=VAE,
        upstream_pattern=(
            "for y in range(blend_extent):\n            b[:, :, :, y, :] ="
        ),
        upstream_fix="blend the tiles in whole-slab ops",
    ),
    PatchDebt(
        patch="models/ltx_2 (blend_t)",
        target=f"{VAE}.AutoencoderKLLTX2Video.blend_t",
        module=VAE,
        upstream_pattern=(
            "for x in range(blend_extent):\n            b[:, :, x, :, :] ="
        ),
        upstream_fix="blend the frames in whole-slab ops",
    ),
    PatchDebt(
        patch="models/ltx_2_vae_assembly (temporal)",
        target=f"{VAE}.AutoencoderKLLTX2Video._temporal_tiled_decode",
        module=VAE,
        upstream_pattern="dec = torch.cat(result_row, dim=2)[:, :, :num_sample_frames]",
        upstream_fix="join each chunk into the previous one as it comes",
    ),
    PatchDebt(
        patch="models/ltx_2_vae_assembly (postprocess)",
        target="diffusers.video_processor.VideoProcessor.postprocess_video",
        module="diffusers.video_processor",
        upstream_pattern=(
            "batch_output = self.postprocess(batch_vid, output_type, **kwargs)"
        ),
        upstream_fix="take the frames post-processed beside the decode",
    ),
    PatchDebt(
        patch="stages/ltx_2/decoding_av (smaller tiles)",
        target=f"{VAE}.AutoencoderKLLTX2Video.decode",
        module=VAE,
        upstream_pattern="decoded = self._decode(z, temb, causal=causal).sample",
        upstream_fix="retry a decode that does not fit on smaller tiles",
    ),
    PatchDebt(
        patch="stages/ltx_2/upsampling (packing)",
        target=f"{STAGES}.ltx_2.upsampling.LTX2UpsampleStage._pack_video_latents",
        module=f"{STAGES}.ltx_2.upsampling",
        upstream_pattern="batch.raw_latent_shape = latents.shape",
        upstream_fix="pack the latents in one graph where they live",
    ),
    PatchDebt(
        patch="stages/ltx_2/detailing (reference tokens)",
        target=f"{STAGES}.ltx_2.upsampling.LTX2UpsampleStage.forward",
        module=f"{STAGES}.ltx_2.upsampling",
        upstream_pattern=(
            "latents = self._upsample_video_latents(batch.latents, server_args, device)"
        ),
        upstream_fix="keep the stage-1 latent as stage-2 IC-LoRA reference tokens",
    ),
    PatchDebt(
        patch="stages/ltx_2/detailing (append reference)",
        target=f"{STAGES}.ltx_2.denoising.LTX2DenoisingStage._prepare_denoising_loop",
        module=f"{STAGES}.ltx_2.denoising",
        upstream_pattern=(
            "base_ctx = super()._prepare_denoising_loop(batch, server_args)"
        ),
        upstream_fix="append IC-LoRA reference tokens to the stage-2 latents",
    ),
    PatchDebt(
        patch="stages/ltx_2/detailing (reference positions)",
        target=f"{STAGES}.ltx_2.denoising.LTX2DenoisingStage._prepare_ltx2_model_inputs",
        module=f"{STAGES}.ltx_2.denoising",
        upstream_pattern=(
            "video_coords, audio_coords = _prepare_ltx2_rope_coords_for_bcg("
        ),
        upstream_fix="give IC-LoRA reference tokens their RoPE positions",
    ),
    PatchDebt(
        patch="stages/ltx_2/detailing (drop reference)",
        target=f"{STAGES}.ltx_2.denoising_av.LTX2AVDenoisingStage._post_denoising_loop",
        module=f"{STAGES}.ltx_2.denoising_av",
        upstream_pattern=(
            "latents = self._truncate_sp_padded_token_latents(batch, latents)"
        ),
        upstream_fix="drop IC-LoRA reference tokens after denoising",
    ),
    PatchDebt(
        patch="stages/ltx_2/detailing (LoRA phase)",
        target=f"{RUNTIME}.pipelines.ltx_2_pipeline.LTX2TwoStagePipeline.switch_lora_phase",
        module=f"{RUNTIME}.pipelines.ltx_2_pipeline",
        upstream_pattern="lora_nicknames, lora_paths, lora_strengths, lora_targets = (",
        upstream_fix="set the detailing IC-LoRA per phase on the resident weights",
    ),
    PatchDebt(
        patch="compilation/placement",
        target=f"{RUNTIME}.loader.component_loaders.component_loader.PipelineComponentLoader.load_component",
        module=f"{RUNTIME}.loader.component_loaders.component_loader",
        upstream_pattern="allow_native_fallback=loader_cls is None,",
        upstream_fix="let the platform place a component on devices of its own",
    ),
    PatchDebt(
        patch="compilation/residency (ready)",
        target=f"{RESIDENCY}._module_ready_on_local_device",
        module=RESIDENCY,
        upstream_pattern="if tensor.device != get_local_torch_device():",
        upstream_fix="count a component on its own devices as ready",
    ),
    PatchDebt(
        patch="compilation/residency (move)",
        target=f"{RESIDENCY}._module_to_local_device",
        module=RESIDENCY,
        upstream_pattern="module.to(device, dtype=dtype, non_blocking=True)",
        upstream_fix="leave a component on its own devices",
    ),
    PatchDebt(
        patch="compilation/residency (stage boundary)",
        target=f"{CORE_STAGES}.base.PipelineStage.__call__",
        module=f"{CORE_STAGES}.base",
        upstream_pattern="result = self.forward(batch, server_args)",
        upstream_fix="let the platform wrap each stage's forward",
    ),
    PatchDebt(
        patch="stages/compile_only",
        target=f"{CORE_STAGES}.base.PipelineStage._run_verification",
        module=f"{CORE_STAGES}.base",
        upstream_pattern="raise StageVerificationError(error_msg)",
        upstream_fix="skip the value checks of a run that computes no values",
    ),
    PatchDebt(
        patch="compilation/residency (compile)",
        target="torch.nn.modules.module.Module.compile",
        module="torch.nn.modules.module",
        upstream_pattern=(
            "self._compiled_call_impl = torch.compile(self._call_impl, *args, **kwargs)"
        ),
        upstream_fix="let a compile backend install its own call",
    ),
    PatchDebt(
        patch="compilation/regions",
        target=f"{RUNTIME}.utils.torch_compile.compile_matching_submodules",
        module=f"{RUNTIME}.utils.torch_compile",
        upstream_pattern="submodule.compile(**compile_kwargs)",
        upstream_fix="compile consecutive blocks as one region",
    ),
    PatchDebt(
        patch="managers/gpu_worker",
        target=f"{RUNTIME}.managers.gpu_worker.GPUWorker.execute_forward",
        module=f"{RUNTIME}.managers.gpu_worker",
        upstream_pattern=(
            "forward_fn=lambda: self.pipeline.forward(req, self.server_args),"
        ),
        upstream_fix="key the request shape and profile one request",
    ),
    PatchDebt(
        patch="entrypoints/prompt_enhancement",
        target=f"{RUNTIME}.entrypoints.openai.prompt_enhancement.maybe_enhance_prompt",
        module=f"{RUNTIME}.entrypoints.openai.prompt_enhancement",
        upstream_pattern='"Prompt enhancement requires --prompt-enhancer-config"',
        upstream_fix="serve the checkpoint's own prompt enhancer",
    ),
    PatchDebt(
        patch="entrypoints/frames",
        target=f"{RUNTIME}.entrypoints.utils._sample_to_uint8_frames",
        module=f"{RUNTIME}.entrypoints.utils",
        upstream_pattern="arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)",
        upstream_fix="convert float frames to uint8 on all host threads",
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


@pytest.fixture
def hook_targets(monkeypatch: pytest.MonkeyPatch, rbln_platform) -> set[str]:
    """Every target the plugin hooks, the opt-in settings' hooks included."""
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_ONLY", "1")
    monkeypatch.setenv("SGLANG_RBLN_LOG_TIMELINE", "1")
    targets: set[str] = set()
    monkeypatch.setattr(
        HookRegistry,
        "register",
        classmethod(
            lambda cls, target, hook, hook_type=HookType.AFTER: targets.add(target)
        ),
    )
    patches.register()
    return targets


def test_every_hook_has_a_patch_debt(hook_targets: set[str]):
    debts = [debt for debt in PATCH_DEBTS if debt.target]
    assert hook_targets == {debt.target for debt in debts}
    assert [d.target for d in debts if not d.target.startswith(f"{d.module}.")] == []

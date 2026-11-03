# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import glob
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    LTX2AudioVideoRotaryPosEmbed,
)
from sglang.multimodal_gen.runtime.platforms.plugins import HookType

from sglang_rbln.patches.stages.ltx_2 import detailing

FRAMES, HEIGHT, WIDTH, CHANNELS, FPS = 3, 2, 4, 8, 24.0


@pytest.fixture
def lora(monkeypatch, tmp_path):
    path = tmp_path / "ic_lora.safetensors"
    save_file(
        {"w.lora_A.weight": torch.zeros(2, 2)},
        path,
        metadata={"reference_downscale_factor": "2"},
    )
    monkeypatch.setenv("SGLANG_RBLN_LTX2_DETAILING_LORA", str(path))
    return path


@pytest.fixture
def model():
    rope = LTX2AudioVideoRotaryPosEmbed(dim=64, modality="video")
    return SimpleNamespace(rope=rope)


def pack(latents, batch_size, batch):
    return latents.flatten(2).transpose(1, 2)


@pytest.fixture
def server_args():
    return SimpleNamespace(pipeline_config=SimpleNamespace(maybe_pack_latents=pack))


@pytest.fixture
def stage1_latents():
    return torch.randn(1, CHANNELS, FRAMES, HEIGHT, WIDTH)


@pytest.fixture
def upsampled(lora, server_args, stage1_latents):
    def upsample(stage, batch, server_args):
        batch.latents = torch.randn(1, FRAMES * HEIGHT * WIDTH * 4, CHANNELS)
        batch.height, batch.width = batch.height * 2, batch.width * 2
        return batch

    batch = SimpleNamespace(
        latents=stage1_latents, fps=FPS, height=64, width=128, extra={}
    )
    return detailing.keep_reference(upsample, None, batch, server_args)


def test_registers_around_the_two_stage_path():
    assert set(detailing.HOOKS) == {
        detailing.SWITCH_PHASE,
        detailing.UPSAMPLE,
        detailing.PREPARE_LOOP,
        detailing.MODEL_INPUTS,
        detailing.POST_LOOP,
    }
    registered = []
    original = detailing.HookRegistry.register
    detailing.HookRegistry.register = lambda target, hook, kind: registered.append(kind)
    try:
        detailing.register()
    finally:
        detailing.HookRegistry.register = original
    assert set(registered) == {HookType.AROUND}


@pytest.mark.parametrize(("phase", "strength"), [("stage1", 0.0), ("stage2", 0.5)])
def test_keeps_the_lora_dynamic_in_both_stages(lora, phase, strength):
    switched, calls = [], []
    pipeline = SimpleNamespace(set_lora=lambda *a, **k: calls.append((a, k)))
    detailing.switch_lora_phase(
        lambda p, ph, **k: switched.append(ph), pipeline, phase, batch=None
    )
    assert switched == [phase]
    assert calls == [
        (
            (detailing.ADAPTER, str(lora)),
            {"target": "transformer", "strength": strength, "merge_weights": False},
        )
    ]


def test_leaves_the_pipeline_alone_without_the_lora(monkeypatch):
    monkeypatch.delenv("SGLANG_RBLN_LTX2_DETAILING_LORA", raising=False)
    calls = []
    pipeline = SimpleNamespace(set_lora=lambda *a, **k: calls.append(a))
    detailing.switch_lora_phase(lambda *a, **k: None, pipeline, "stage2")
    assert calls == []
    batch = SimpleNamespace(latents=torch.zeros(1), extra={})
    assert detailing.keep_reference(lambda s, b, a: b, None, batch, None) is batch
    assert batch.extra == {}


def test_keeps_the_stage1_latent_as_tokens(upsampled, stage1_latents):
    tokens, grid = upsampled.extra[detailing.REFERENCE]
    torch.testing.assert_close(tokens, pack(stage1_latents, 1, None))
    assert grid == (FRAMES, HEIGHT, WIDTH, FPS, 2)


def test_reference_lands_on_the_target_grid(model, upsampled):
    """ltx-core scales a reference token's height and width by the downscale
    factor, so reference cell (f, i, j) starts where target cell (f, 2i, 2j) does."""
    reference = upsampled.extra[detailing.REFERENCE]
    coords = detailing.reference_coords(model, reference, 1, "cpu")
    target = model.rope.prepare_video_coords(
        1, FRAMES, 2 * HEIGHT, 2 * WIDTH, "cpu", FPS
    )
    grid = coords.view(3, FRAMES, HEIGHT, WIDTH, 2)
    target_grid = target.view(3, FRAMES, 2 * HEIGHT, 2 * WIDTH, 2)
    torch.testing.assert_close(grid[..., 0], target_grid[:, :, ::2, ::2, 0])
    # A reference cell spans two target cells.
    torch.testing.assert_close(grid[1:, ..., 1], target_grid[1:, :, 1::2, 1::2, 1])


def stage2_context(latents):
    return SimpleNamespace(
        stage="stage2",
        latents=latents,
        denoise_mask=None,
        clean_latent=None,
        latent_num_frames_for_model=FRAMES,
        latent_height=2 * HEIGHT,
        latent_width=2 * WIDTH,
    )


def test_appends_clean_reference_tokens(upsampled):
    noisy = torch.randn(1, FRAMES * HEIGHT * WIDTH * 4, CHANNELS)
    upsampled.did_sp_shard_latents = False
    ctx = detailing.append_reference(
        lambda *a: stage2_context(noisy), None, upsampled, None
    )
    tokens = upsampled.extra[detailing.REFERENCE][0]
    count = noisy.shape[1]
    assert upsampled.extra[detailing.TARGET_TOKENS] == count
    torch.testing.assert_close(ctx.latents, torch.cat([noisy, tokens], dim=1))
    torch.testing.assert_close(ctx.clean_latent[:, count:], tokens)
    # Timestep = sigma * mask per token: the reference stays at timestep 0.
    assert ctx.denoise_mask[:, :count].eq(1).all()
    assert ctx.denoise_mask[:, count:].eq(0).all()
    assert ctx.denoise_mask.shape == (1, count + tokens.shape[1], 1)


def test_conditions_only_stage2(upsampled):
    ctx = stage2_context(torch.zeros(1, 4, CHANNELS))
    ctx.stage = "stage1"
    assert detailing.append_reference(lambda *a: ctx, None, upsampled, None) is ctx
    assert ctx.latents.shape[1] == 4


def test_model_sees_reference_positions(model, upsampled):
    noisy = torch.randn(1, FRAMES * HEIGHT * WIDTH * 4, CHANNELS)
    upsampled.did_sp_shard_latents = False
    ctx = detailing.append_reference(
        lambda *a: stage2_context(noisy), None, upsampled, None
    )
    inputs = SimpleNamespace(latent_model_input=ctx.latents, video_coords=None)
    step = SimpleNamespace(current_model=model)
    inputs = detailing.reference_positions(
        lambda *a: inputs, None, ctx, step, upsampled
    )
    reference = upsampled.extra[detailing.REFERENCE]
    expected = torch.cat(
        [
            model.rope.prepare_video_coords(
                1, FRAMES, 2 * HEIGHT, 2 * WIDTH, "cpu", FPS
            ),
            detailing.reference_coords(model, reference, 1, "cpu"),
        ],
        dim=2,
    )
    torch.testing.assert_close(inputs.video_coords, expected)
    assert inputs.video_coords.shape[2] == ctx.latents.shape[1]


def test_decodes_only_the_target_tokens(upsampled):
    upsampled.extra[detailing.TARGET_TOKENS] = 5
    seen = []
    detailing.drop_reference(
        lambda stage, batch, latents, **k: seen.append(latents),
        None,
        upsampled,
        torch.zeros(1, 9, CHANNELS),
        server_args=None,
    )
    assert seen[0].shape == (1, 5, CHANNELS)
    assert upsampled.extra == {}


class _Blocks(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        from sglang.multimodal_gen.runtime.layers.linear import ColumnParallelLinear

        self.projections = torch.nn.ModuleList(
            ColumnParallelLinear(
                width, width, gather_output=True, params_dtype=torch.bfloat16
            )
            for _ in range(2)
        )

    def forward(self, x):
        for projection in self.projections:
            x = torch.nn.functional.gelu(projection(x)[0])
        return x


def _kernel_bytes(cache) -> list[int]:
    import rebel

    return [
        sum(rebel.RBLNCompiledModel(path).get_alloc_per_node_by_key().get("Kernel", []))
        for path in sorted(glob.glob(f"{cache}/*.rbln"), key=os.path.getmtime)
    ]


@pytest.mark.parametrize(
    ("stage1_strength", "shares_pool"),
    [
        pytest.param(0.0, True, id="strength-0"),
        pytest.param(None, False, id="deactivated"),
    ],
)
def test_both_stages_compile_against_one_weight_pool(
    monkeypatch, tmp_path, npu_name, stage1_strength, shares_pool
):
    """A stage-1 graph without the LoRA weights is another weight group than the
    stage-2 graph with them, so stage 2 lays out a second copy of the base
    weights. Keeping the LoRA at strength 0 in stage 1 keeps one pool."""
    import rebel
    import rebel.core.torch_compile  # noqa: F401
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.runtime.layers.lora.linear import (
        replace_submodule,
        wrap_with_lora_layer,
    )
    from sglang.srt.utils.network import get_open_port

    if not model_parallel_is_initialized():
        for key, value in dict(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(get_open_port()),
            RANK="0",
            LOCAL_RANK="0",
            WORLD_SIZE="1",
        ).items():
            monkeypatch.setenv(key, value)
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    monkeypatch.setenv("RBLN_FORCE_NPU_NAME", npu_name)
    width, rank = 256, 8
    torch.manual_seed(0)
    model = _Blocks(width).eval()
    adapter = {}
    for index in range(2):
        name = f"projections.{index}"
        layer = wrap_with_lora_layer(model.get_submodule(name), rank, rank)
        replace_submodule(model, name, layer)
        adapter[layer] = (
            torch.randn(rank, width, dtype=torch.bfloat16),
            torch.randn(width, rank, dtype=torch.bfloat16),
        )
    compiled = torch.compile(
        model,
        backend="rbln",
        dynamic=False,
        options={
            "compile_context": rebel.CompileContext(use_weight_sharing=True),
            "mode": ["compile_only"],
            "cache_dir": str(tmp_path),
            "num_devices": 4,
        },
    )
    with torch.no_grad():
        for stage, strength in ((1, stage1_strength), (2, 0.5)):
            for layer, (a, b) in adapter.items():
                if strength is None:
                    layer.disable_lora = True
                else:
                    layer.set_lora_weights(
                        a,
                        b,
                        strength=strength,
                        merge_weights=False,
                        clear_existing=True,
                    )
            compiled(torch.randn(1, 64 * stage, width, dtype=torch.bfloat16))
    stage1, stage2 = _kernel_bytes(tmp_path)
    assert stage1 > 0
    assert (stage2 == 0) is shares_pool


class UpsampleStage:
    """Stands in for LTX2UpsampleStage, the one target both plugins hook."""

    def __init__(self, upsampler):
        self.spatial_upsampler = upsampler

    def forward(self, batch, server_args):
        batch.latents = torch.randn(1, FRAMES * HEIGHT * WIDTH * 4, CHANNELS)
        return batch


def test_the_upsampler_compiles_and_the_reference_is_kept_in_one_forward(
    lora, stage1_latents, monkeypatch
):
    # compiling is a BEFORE hook so the detailing LoRA keeps the only AROUND one; both
    # have to take effect on the same call
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry

    from sglang_rbln.patches.stages.ltx_2 import upsampling

    compiled = []
    monkeypatch.setattr(
        upsampling, "compile_on_one_device", lambda module, name: compiled.append(name)
    )
    monkeypatch.setenv("SGLANG_RBLN_UPSAMPLER_ON_NPU", "1")
    monkeypatch.setattr(HookRegistry, "_hooks", type(HookRegistry._hooks)(list))
    monkeypatch.setattr(HookRegistry, "_patched", set())
    # records the unhooked forward as the value to restore, before the hooks replace it
    monkeypatch.setattr(UpsampleStage, "forward", UpsampleStage.forward)
    target = f"{__name__}.UpsampleStage.forward"
    HookRegistry.register(target, upsampling.compile_upsampler, HookType.BEFORE)
    HookRegistry.register(target, detailing.keep_reference, HookType.AROUND)
    HookRegistry.apply_hooks()

    stage = UpsampleStage(SimpleNamespace())
    batch = SimpleNamespace(
        latents=stage1_latents, fps=FPS, height=64, width=128, extra={}
    )
    args = SimpleNamespace(
        enable_torch_compile=True,
        pipeline_config=SimpleNamespace(maybe_pack_latents=pack),
    )
    out = stage.forward(batch, args)

    assert compiled == ["latent upsampler"]
    assert detailing.REFERENCE in out.extra

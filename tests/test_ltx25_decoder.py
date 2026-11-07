# SPDX-License-Identifier: Apache-2.0
import logging
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.configs.models.decoders.ltx_2_5_diffusion_decoder import (
    LTX25DiffusionDecoderConfig,
)
from sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder import (
    LTX2VideoDiffusionDecoderModel,
)
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import LTX2VideoDecoder3d
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln.compilation import backend as compilation
from sglang_rbln.compilation import placement, regions
from sglang_rbln.patches.models import ltx_2_tile_ranks as tile_ranks
from sglang_rbln.patches.stages.ltx_2 import decoding_av

pytestmark = pytest.mark.usefixtures("rbln_platform")


@pytest.fixture
def stage():
    return SimpleNamespace(
        vae=SimpleNamespace(decoder=nn.Linear(4, 4), use_framewise_decoding=False),
        diffusion_decoder=None,
    )


@pytest.fixture
def compiled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs))
    )
    return calls


def decode_video(stage, batch, server_args, on_frames=None):
    return "video", on_frames


def test_registers_around_the_video_decode():
    targets = decoding_av.register()
    assert targets == (decoding_av.HOOK_TARGET, decoding_av.DIFFUSION_DECODE_TARGET)
    hooks = [hook for _, hook, _ in HookRegistry._hooks[decoding_av.HOOK_TARGET]]
    assert decoding_av.compile_decoders in hooks
    hooks = [
        hook for _, hook, _ in HookRegistry._hooks[decoding_av.DIFFUSION_DECODE_TARGET]
    ]
    assert decoding_av.decode_in_decoder_dtype in hooks


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "eager"])
def test_a_compiled_diffusion_decoder_gets_latents_in_its_dtype(compiled):
    decoder = nn.Linear(4, 4).to(torch.bfloat16)
    if compiled:
        setattr(decoder, decoding_av.COMPILED, True)
    seen = []

    def decode(stage, module, latents, batch, server_args):
        seen.append(latents.dtype)
        return "video"

    latents = torch.randn(1, 4)
    output = decoding_av.decode_in_decoder_dtype(
        decode, None, decoder, latents, None, None
    )
    assert output == "video"
    assert seen == [torch.bfloat16 if compiled else torch.float32]


def test_one_hook_compiles_the_decoder_and_the_vocoder(stage, compiled, monkeypatch):
    monkeypatch.setenv("SGLANG_RBLN_VOCODER_ON_NPU", "1")
    stage.vocoder = nn.Linear(4, 4)
    batch = SimpleNamespace(use_diffusion_decoder=False)
    server_args = SimpleNamespace(enable_torch_compile=True)
    output = decoding_av.compile_decoders(decode_video, stage, batch, server_args)
    assert output == ("video", None)
    assert [module for module, _ in compiled] == [stage.vae.decoder, stage.vocoder]


def test_compiles_the_decoder_once(stage, compiled):
    batch = SimpleNamespace(use_diffusion_decoder=False)
    server_args = SimpleNamespace(enable_torch_compile=True)
    outputs = [
        decoding_av.compile_video_decoder(
            decode_video, stage, batch, server_args, on_frames=print
        )
        for _ in range(2)
    ]
    assert outputs == [("video", print)] * 2
    assert [module for module, _ in compiled] == [stage.vae.decoder]
    assert compiled[0][1]["backend"] == "rbln"
    assert compiled[0][1]["dynamic"] is False
    assert stage.vae.use_framewise_decoding
    assert compiled[0][1]["options"]["compile_context"].state.use_weight_sharing


def test_decoder_spans_the_tensor_parallel_devices(stage, compiled, monkeypatch):
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    batch = SimpleNamespace(use_diffusion_decoder=False)
    server_args = SimpleNamespace(enable_torch_compile=True)
    decoding_av.compile_video_decoder(decode_video, stage, batch, server_args)
    options = compiled[0][1]["options"]
    assert options["num_devices"] == 4
    assert "device" not in options


def test_decoder_shares_the_transformer_context(stage, compiled, monkeypatch):
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    monkeypatch.setattr(compilation, "_device_set_contexts", {})
    transformer = compilation.compile_options(nn.Linear(4, 4))["compile_context"]
    batch = SimpleNamespace(use_diffusion_decoder=False)
    server_args = SimpleNamespace(enable_torch_compile=True)
    decoding_av.compile_video_decoder(decode_video, stage, batch, server_args)
    assert compiled[0][1]["options"]["compile_context"] is transformer


def test_decoder_keeps_the_devices_declared_for_its_component(
    stage, compiled, monkeypatch
):
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    placement.place(stage.vae.decoder, (7,))
    batch = SimpleNamespace(use_diffusion_decoder=False)
    server_args = SimpleNamespace(enable_torch_compile=True)
    decoding_av.compile_video_decoder(decode_video, stage, batch, server_args)
    options = compiled[0][1]["options"]
    assert (options["num_devices"], options["device"]) == (1, [7])


@pytest.mark.parametrize(
    ("enable_torch_compile", "use_diffusion_decoder"),
    [(False, False), (True, True)],
    ids=["compile-disabled", "diffusion-decoder"],
)
def test_leaves_the_decoder_eager(
    stage, compiled, enable_torch_compile, use_diffusion_decoder
):
    batch = SimpleNamespace(use_diffusion_decoder=use_diffusion_decoder)
    server_args = SimpleNamespace(enable_torch_compile=enable_torch_compile)
    decoding_av.compile_video_decoder(decode_video, stage, batch, server_args)
    assert compiled == []
    assert not stage.vae.use_framewise_decoding


@pytest.fixture
def diffusion_decoder():
    config = LTX25DiffusionDecoderConfig()
    arch = config.arch_config
    arch.decoder_stage_channels = [64, 32, 16, 16, 8]
    arch.decoder_stage_depths = [2, 1, 1, 1, 1]
    arch.decoder_head_dim = 8
    return LTX2VideoDiffusionDecoderModel(config)


@pytest.fixture
def torch_compiled(monkeypatch):
    calls = []

    def fake_compile(fn, **kwargs):
        calls.append((fn, kwargs))
        return fn

    monkeypatch.setattr(decoding_av.torch, "compile", fake_compile)
    return calls


def test_diffusion_decoder_tiles_shrink_to_the_token_budget(diffusion_decoder):
    decoding_av.fit_diffusion_decoder_tiles(diffusion_decoder)
    tiles = {
        name: getattr(diffusion_decoder, f"tile_sample_{name}")
        for name in (
            "min_height",
            "stride_height",
            "min_num_frames",
            "stride_num_frames",
        )
    }
    assert tiles == {
        "min_height": 384,
        "stride_height": 256,
        "min_num_frames": 16,
        "stride_num_frames": 8,
    }
    assert diffusion_decoder.tile_sample_min_width == 384
    tokens = decoding_av.diffusion_decoder_tile_tokens(diffusion_decoder)
    assert tokens <= decoding_av.DIFFUSION_DECODER_TILE_TOKENS


def test_leaves_the_diffusion_decoder_eager_unless_opted_in(
    stage, compiled, torch_compiled, diffusion_decoder, monkeypatch
):
    monkeypatch.delenv("SGLANG_RBLN_COMPILE_DIFFUSION_DECODER", raising=False)
    stage.diffusion_decoder = diffusion_decoder
    batch = SimpleNamespace(use_diffusion_decoder=True)
    server_args = SimpleNamespace(enable_torch_compile=True)
    output = decoding_av.compile_video_decoder(decode_video, stage, batch, server_args)
    assert output == ("video", None)
    assert compiled == torch_compiled == []
    assert not getattr(diffusion_decoder, decoding_av.COMPILED, False)


@pytest.mark.parametrize(
    ("component_devices", "stage_options", "tile_devices"),
    [
        ("", {"num_devices": 4}, (0, 1, 2, 3)),
        (
            "diffusion_decoder=0,1,2,3",
            {"num_devices": 4, "device": [0, 1, 2, 3]},
            (0, 1, 2, 3),
        ),
        ("diffusion_decoder=2", {"num_devices": 1, "device": [2]}, (2,)),
    ],
    ids=["server-devices", "all-four", "one-device"],
)
def test_diffusion_decoder_compiles_on_its_devices(
    stage,
    compiled,
    torch_compiled,
    diffusion_decoder,
    monkeypatch,
    component_devices,
    stage_options,
    tile_devices,
):
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_DIFFUSION_DECODER", "1")
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    monkeypatch.setenv("SGLANG_RBLN_COMPONENT_DEVICES", component_devices)
    loaded = diffusion_decoder
    placement.place_component(
        lambda *args, **kwargs: (loaded, 0.0), "diffusion_decoder"
    )
    stage.diffusion_decoder = diffusion_decoder
    batch = SimpleNamespace(use_diffusion_decoder=True)
    server_args = SimpleNamespace(enable_torch_compile=True)
    decoding_av.compile_video_decoder(decode_video, stage, batch, server_args)

    # Stages 1-3 on the whole device set, in its one context.
    stages = [kwargs["options"] for _, kwargs in compiled]
    for option in stages:
        assert {key: option.get(key) for key in stage_options} == stage_options
    assert len({id(option["compile_context"]) for option in stages}) == 1
    # Each tile phase once per device, on that device alone, in the stages' context.
    assert getattr(diffusion_decoder, tile_ranks.TILE_DEVICES) == tile_devices
    tiles = [kwargs["options"] for _, kwargs in torch_compiled]
    assert len(tiles) == len(decoding_av.DIFFUSION_DECODER_GRAPHS) * len(tile_devices)
    for phase in decoding_av.DIFFUSION_DECODER_GRAPHS:
        dispatch = getattr(diffusion_decoder.decoder, phase)
        assert isinstance(dispatch, tile_ranks.RankDispatch)
        assert len(dispatch.compiled) == len(tile_devices)
    assert all(option["num_devices"] == 1 for option in tiles)
    assert sorted({option["device"] for option in tiles}) == list(tile_devices)
    contexts = {id(option["compile_context"]) for option in stages + tiles}
    assert len(contexts) == 1


def test_tile_ranks_without_the_cache_keep_a_context_each(
    stage, compiled, torch_compiled, diffusion_decoder, monkeypatch
):
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_DIFFUSION_DECODER", "1")
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    monkeypatch.setenv("SGLANG_RBLN_DISABLE_COMPILE_CACHE", "1")
    stage.diffusion_decoder = diffusion_decoder
    batch = SimpleNamespace(use_diffusion_decoder=True)
    server_args = SimpleNamespace(enable_torch_compile=True)
    decoding_av.compile_video_decoder(decode_video, stage, batch, server_args)
    by_device = {}
    for _, kwargs in torch_compiled:
        options = kwargs["options"]
        context = id(options["compile_context"])
        by_device.setdefault(options["device"], set()).add(context)
    assert sorted(by_device) == [0, 1, 2, 3]
    assert all(len(contexts) == 1 for contexts in by_device.values())
    assert len(set().union(*by_device.values())) == 4


def test_compiles_the_diffusion_decoder_once(
    stage, compiled, torch_compiled, diffusion_decoder, monkeypatch
):
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_DIFFUSION_DECODER", "1")
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "2")
    stage.diffusion_decoder = diffusion_decoder
    batch = SimpleNamespace(use_diffusion_decoder=True)
    server_args = SimpleNamespace(enable_torch_compile=True)
    for _ in range(2):
        decoding_av.compile_video_decoder(decode_video, stage, batch, server_args)
    decoder = diffusion_decoder.decoder
    phases = [fn.__name__ for fn, _ in torch_compiled]
    assert phases == [
        name for name in decoding_av.DIFFUSION_DECODER_GRAPHS for _ in range(2)
    ]
    modules = [module for module, _ in compiled]
    assert modules[0] is decoder.conv_in
    assert all(upsample in modules for upsample in decoder.upsamples[:-1])
    assert decoder.upsamples[-1] not in modules
    assert all(
        isinstance(block.forward, regions.DeferredBlock)
        for blocks in decoder.det_stages[:-1]
        if len(blocks) > 1
        for block in blocks
    )
    assert not stage.vae.use_framewise_decoding


@pytest.mark.npu
def test_video_decoder_runs_on_the_npu(caplog, monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compilation.ensure_compile_backend()
    torch.manual_seed(0)
    decoder = (
        LTX2VideoDecoder3d(
            block_out_channels=(64, 128, 128, 256),
            spatio_temporal_scaling=(True, True, True, True),
            layers_per_block=(1, 1, 1, 1, 1),
            inject_noise=(False,) * 5,
            upsample_residual=(False,) * 4,
            upsample_factor=(2, 2, 1, 2),
            upsample_type=("spatiotemporal", "spatiotemporal", "temporal", "spatial"),
            spatial_padding_mode="zeros",
        )
        .to(torch.bfloat16)
        .eval()
    )
    latents = torch.randn(1, 128, 2, 2, 3, dtype=torch.bfloat16)
    with torch.no_grad():
        reference = decoder(latents)
        compiled = torch.compile(
            decoder,
            backend=compilation.COMPILE_BACKEND_NAME,
            options=compilation.compile_options(decoder),
            fullgraph=True,
            dynamic=False,
        )
        with caplog.at_level(logging.WARNING, logger="rebel-compiler"):
            output = compiled(latents)
    assert "Fallback to eager" not in caplog.text
    assert output.shape == reference.shape == (1, 3, 9, 64, 96)
    error = (output.float() - reference.float()).abs().mean() / reference.abs().mean()
    assert error < 2e-2

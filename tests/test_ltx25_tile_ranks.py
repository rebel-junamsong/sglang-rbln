# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from sglang.multimodal_gen.configs.models.decoders.ltx_2_5_diffusion_decoder import (
    LTX25DiffusionDecoderConfig,
)
from sglang.multimodal_gen.runtime.models.decoders import ltx_2_5_diffusion_decoder
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry

from sglang_rbln.patches.models import ltx_2_tile_ranks as tile_ranks

# These need a device to name, not the hardware: placement and graph structure
# only. A dummy device refuses to execute a compiled graph and its eager outputs
# are garbage, so nothing here may assert on values from one.
pytestmark = pytest.mark.usefixtures("dummy_device")

MODEL = ltx_2_5_diffusion_decoder.LTX2VideoDiffusionDecoderModel
DECODER = ltx_2_5_diffusion_decoder.LTX2VideoDiffusionDecoder3d


@pytest.fixture(name="model")
def model_fixture():
    config = LTX25DiffusionDecoderConfig()
    arch = config.arch_config
    arch.decoder_stage_channels = [64, 32, 16, 16, 8]
    arch.decoder_stage_depths = [1, 1, 1, 1, 1]
    arch.decoder_head_dim = 8
    arch.decoder_stage_kernels = [[3, 3, 3]] * 4
    arch.decoder_stage5_kernel = [3, 3, 3]
    torch.manual_seed(0)
    model = MODEL(config).eval()
    model.use_tiling = True
    # Cells of 8 px: tiles of 8 x 8 cells every 4, so the grid below has
    # several tiles per spatial axis, edge tiles included.
    model.tile_sample_min_height = model.tile_sample_min_width = 64
    model.tile_sample_stride_height = model.tile_sample_stride_width = 32
    return model


@pytest.fixture(name="ranked")
def ranked_fixture(monkeypatch):
    """Route the upstream methods through the plugin hooks, as registration does."""
    for target, hook in (
        ("tiled_decode", tile_ranks.tiled_decode_over_ranks),
        ("_tile_shard", tile_ranks.tile_shard_of_rank),
    ):
        original = getattr(MODEL, target)
        monkeypatch.setattr(
            MODEL,
            target,
            lambda self, *a, _o=original, _h=hook, **k: _h(_o, self, *a, **k),
        )
    gather = ltx_2_5_diffusion_decoder._all_gather_tiles
    monkeypatch.setattr(
        ltx_2_5_diffusion_decoder,
        "_all_gather_tiles",
        lambda *a, **k: tile_ranks.gather_rank_tiles(gather, *a, **k),
    )
    stages = DECODER.forward_stages_1_to_3
    monkeypatch.setattr(
        DECODER,
        "forward_stages_1_to_3",
        lambda self, *a, **k: tile_ranks.shared_stages_1_to_3(stages, self, *a, **k),
    )


def _record_ranks(model, world_size):
    """Give each phase one callable per rank that logs which rank ran it."""
    log = []
    decoder = model.decoder
    for name in ("forward_stage_4", "forward_diffusion_step"):
        phase = getattr(decoder, name)

        def per_rank(device, phase=phase, name=name):
            def run(*args, **kwargs):
                rank = tile_ranks.current_rank()
                log.append((name, device, rank[0] if rank else None))
                return phase(*args, **kwargs)

            return run

        setattr(
            decoder,
            name,
            tile_ranks.RankDispatch([per_rank(d) for d in range(world_size)]),
        )
    setattr(model, tile_ranks.TILE_DEVICES, tuple(range(world_size)))
    return log


@pytest.mark.usefixtures("ranked")
@pytest.mark.parametrize("world_size", [2, 3])
def test_tiles_spread_over_ranks_match_one_rank(model, world_size):
    latents = torch.randn(1, 128, 2, 4, 6)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        expected = model(latents, generator=generator)
    expected_state = generator.get_state()

    log = _record_ranks(model, world_size)
    stage_calls = []
    stages = tile_ranks.shared_stages_1_to_3

    def counting(forward, decoder, hidden_states, *args, **kwargs):
        stage_calls.append(tile_ranks.current_rank()[0])
        return forward(decoder, hidden_states, *args, **kwargs)

    tile_ranks.shared_stages_1_to_3 = lambda f, d, h, *a, **k: stages(
        lambda *x, **y: counting(f, *x, **y), d, h, *a, **k
    )
    generator = torch.Generator().manual_seed(0)
    try:
        with torch.no_grad():
            actual = model(latents, generator=generator)
    finally:
        tile_ranks.shared_stages_1_to_3 = stages

    assert torch.equal(actual, expected)
    assert torch.equal(generator.get_state(), expected_state)
    tiles = [entry for entry in log if entry[0] == "forward_stage_4"]
    assert len(tiles) > world_size
    # Every rank ran on its own device, and only the tiles that are its share.
    assert all(device == rank for _, device, rank in log)
    assert sorted(rank for _, _, rank in tiles) == sorted(
        index % world_size for index in range(len(tiles))
    )
    assert stage_calls == [0]


@pytest.mark.usefixtures("ranked")
def test_ranks_compile_under_the_callers_dynamo_config(model):
    seen = []
    for name in ("forward_stage_4", "forward_diffusion_step"):
        phase = getattr(model.decoder, name)

        def run(*args, _phase=phase, **kwargs):
            seen.append(torch._dynamo.config.recompile_limit)
            return _phase(*args, **kwargs)

        setattr(model.decoder, name, run)
    setattr(model, tile_ranks.TILE_DEVICES, (0, 1))
    with torch._dynamo.config.patch(recompile_limit=1234), torch.no_grad():
        model(torch.randn(1, 128, 2, 4, 6), generator=torch.Generator())
    assert seen
    assert set(seen) == {1234}


def test_registers_around_the_upstream_tiling():
    targets = tile_ranks.register()
    for target, hook in zip(
        targets,
        (
            tile_ranks.tiled_decode_over_ranks,
            tile_ranks.tile_shard_of_rank,
            tile_ranks.gather_rank_tiles,
            tile_ranks.shared_stages_1_to_3,
        ),
        strict=True,
    ):
        assert hook in [h for _, h, _ in HookRegistry._hooks[target]]


def test_one_device_keeps_the_upstream_tiled_decode(model):
    calls = []
    setattr(model, tile_ranks.TILE_DEVICES, (0,))

    def tiled_decode(self, *args, **kwargs):
        calls.append(tile_ranks.current_rank())
        return "decoded"

    assert tile_ranks.tiled_decode_over_ranks(tiled_decode, model, None) == "decoded"
    assert calls == [None]

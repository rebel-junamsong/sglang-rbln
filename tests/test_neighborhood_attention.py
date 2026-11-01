# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from sglang.multimodal_gen.runtime.models.vaes import flux3_neighborhood_attention

from sglang_rbln.layers.attention import neighborhood


@pytest.fixture(scope="module")
def flex_reference():
    torch.manual_seed(0)
    return flux3_neighborhood_attention.neighborhood_attention


@pytest.mark.parametrize(
    ("grid", "kernel", "causal", "tile"),
    [
        ((3, 10, 12), [5, 5, 5], None, None),
        ((7, 9, 13), [5, 5, 5], [True, False, False], [2, 4, 5]),
        ((9, 17, 20), [5, 5, 5], None, [3, 16, 16]),
        ((2, 6, 6), [5, 5, 5], None, None),
        ((1, 33, 40), [5, 5, 5], [True, False, False], None),
        ((17, 20), [5, 5], None, None),
        ((5, 7), [5, 5], None, [5, 7]),
        ((1, 17, 23), [5, 5, 5], [True, False, False], None),
        ((2, 34, 40), [5, 5, 5], [True, False, False], None),
        ((4, 46, 34), [5, 5, 5], [True, False, False], None),
        ((13, 15, 11), [5, 5, 5], [True, False, False], None),
        ((17, 23), [5, 5], None, None),
        ((47, 33), [5, 5], None, None),
        ((12, 17, 20), [5, 5, 5], None, None),
        ((12, 34, 40), [5, 5, 5], [True, False, False], None),
    ],
    ids=str,
)
def test_matches_the_flex_attention_windows(flex_reference, grid, kernel, causal, tile):
    q, k, v = (torch.randn(2, *grid, 4, 64) for _ in range(3))
    expected = flex_reference(q, k, v, kernel_size=kernel, is_causal=causal)
    actual = neighborhood.neighborhood_attention(
        q, k, v, kernel_size=kernel, is_causal=causal, tile_size=tile
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    ("length", "kernel", "size"), [(10, 5, 8), (17, 5, 1), (3, 5, 8), (176, 11, 4)]
)
def test_every_window_lies_in_its_halo(length, kernel, size, causal):
    classes = neighborhood.axis_tiles(length, kernel, causal, size, torch.device("cpu"))
    window = min(kernel, length)
    positions = sorted(p for tiles in classes for p in tiles.positions)
    assert positions == list(range(-(-length // min(size, length))))
    for tiles in classes:
        halo = torch.tensor(tiles.halo).view(tiles.count, tiles.span)
        queries = torch.tensor(tiles.queries).view(tiles.count, tiles.size)
        assert tiles.mask.shape == (tiles.size, tiles.span)
        assert int(halo.max()) < length and int(halo.min()) >= 0
        for tile, rows in enumerate(queries.tolist()):
            for row, query in enumerate(rows):
                keys = halo[tile][tiles.mask[row]].tolist()
                assert len(keys) == (min(query + 1, window) if causal else window)
                assert query in keys


@pytest.mark.parametrize(
    ("length", "kernel", "size", "expected"),
    [(176, 11, 4, 5), (192, 11, 8, 3), (136, 5, 8, 3), (31, 11, 31, 1)],
)
def test_interior_tiles_share_one_mask(length, kernel, size, expected):
    classes = neighborhood.axis_tiles(length, kernel, False, size, torch.device("cpu"))
    assert len(classes) == expected
    assert max(tiles.count for tiles in classes) == -(-length // size) - expected + 1


@pytest.mark.parametrize(
    ("grid", "kernel", "expected"),
    [
        ((12, 136, 160), [5, 5, 5], [12, 4, 8]),
        ((16, 15, 16), [5, 5, 5], [1, 15, 8]),
        ((9, 17, 20), [3, 5, 5], [9, 4, 8]),
        ((17, 20), [5, 5], [8, 8]),
        ((14, 9), [5, 3], [14, 9]),
    ],
)
def test_axes_up_to_three_windows_long_are_one_tile(grid, kernel, expected):
    assert neighborhood.default_tile_lengths(grid, kernel) == expected


def test_index_tables_are_runs_of_consecutive_positions():
    classes = neighborhood.axis_tiles(136, 5, False, 8, torch.device("cpu"))
    for tiles in classes:
        halo = torch.tensor(tiles.halo).view(tiles.count, tiles.span)
        assert bool((halo.diff(dim=1) == 1).all())
    queries = sorted(query for tiles in classes for query in tiles.queries)
    assert queries == list(range(136))


@pytest.mark.parametrize(
    ("grid", "kernel", "causal"),
    [
        ((4, 34, 40), [5, 5, 5], [True, False, False]),
        ((13, 34, 36), [11, 11, 11], None),
    ],
    ids=str,
)
@pytest.mark.parametrize("budget", [1, 1 << 16])
def test_calls_split_under_the_score_budget_stay_exact(
    monkeypatch, flex_reference, grid, kernel, causal, budget
):
    q, k, v = (torch.randn(1, *grid, 2, 32) for _ in range(3))
    expected = flex_reference(q, k, v, kernel_size=kernel, is_causal=causal)
    calls = []
    sdpa = neighborhood.F.scaled_dot_product_attention
    monkeypatch.setattr(neighborhood, "MAX_SCORES_PER_CALL", budget)
    monkeypatch.setattr(
        neighborhood.F,
        "scaled_dot_product_attention",
        lambda *args, **kwargs: calls.append(args[0].shape) or sdpa(*args, **kwargs),
    )
    actual = neighborhood.neighborhood_attention(
        q, k, v, kernel_size=kernel, is_causal=causal
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert len(calls) > 1
    if budget == 1:
        assert {shape[1] for shape in calls} == {2}


@pytest.mark.parametrize(
    ("grid", "kernel"), [((13, 34, 36), [11, 11, 11]), ((17, 23), [5, 5])], ids=str
)
def test_traces_as_one_graph(monkeypatch, grid, kernel):
    monkeypatch.setattr(neighborhood, "MAX_SCORES_PER_CALL", 1 << 16)
    q, k, v = (torch.randn(1, *grid, 2, 32) for _ in range(3))
    expected = neighborhood.neighborhood_attention(q, k, v, kernel_size=kernel)
    compiled = torch.compile(
        neighborhood.neighborhood_attention, backend="eager", fullgraph=True
    )
    torch.testing.assert_close(compiled(q, k, v, kernel_size=kernel), expected)

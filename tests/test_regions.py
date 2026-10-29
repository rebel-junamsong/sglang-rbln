# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln.compilation import regions

pytestmark = pytest.mark.usefixtures("rbln_platform")


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x, y, scale, *, shift):
        return torch.tanh(self.linear(x) * scale + shift), y + x.mean()


class Model(nn.Module):
    _compile_conditions = [lambda name, module: isinstance(module, Block)]

    def __init__(self, depth: int = 7, width: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(width) for _ in range(depth))
        self.head = nn.Linear(width, width)

    def forward(self, x):
        y = torch.zeros(())
        late = len(self.blocks) // 2
        for index, block in enumerate(self.blocks):
            x, y = block(x, y, 2.0 if index >= late else 1.0, shift=index * 0.1)
        return self.head(x) + y


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x, key, value, *, residual=None):
        residual = x if residual is None else x + residual
        return torch.tanh(self.linear(residual) * key + value), residual


class ResidualModel(nn.Module):
    _compile_conditions = [lambda name, module: isinstance(module, ResidualBlock)]

    def __init__(self, depth: int = 6, width: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(ResidualBlock(width) for _ in range(depth))
        self.cache = [
            (torch.full((width,), 1.0 + i / 10), torch.full((width,), i / 10))
            for i in range(depth)
        ]

    def forward(self, x, rescale: bool = False):
        residual = None
        for index, layer in enumerate(self.layers):
            key, value = self.cache[index]
            x, residual = layer(x, key, value, residual=residual)
            if rescale:
                x = x * 2
        return x + residual


@pytest.fixture
def graphs():
    captured = []

    def backend(graph, example_inputs):
        captured.append(graph)
        return graph

    return captured, {"backend": backend, "dynamic": False}


def _plain_compile(module, *, compile_kwargs):
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        compile_matching_submodules,
    )

    return compile_matching_submodules(module, compile_kwargs=compile_kwargs)


@pytest.mark.parametrize("size", [2, 3, 7])
def test_grouped_blocks_match_eager(monkeypatch, graphs, size):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", str(size))
    captured, compile_kwargs = graphs
    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)
        count = regions.compile_block_groups(
            _plain_compile, model, compile_kwargs=compile_kwargs
        )
        outputs = [model(x) for _ in range(3)]
    assert count == math.ceil(len(model.blocks) / size)
    for output in outputs:
        torch.testing.assert_close(output, expected)
    assert len(captured) == count


def test_default_groups_four_blocks(monkeypatch, graphs):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    captured, compile_kwargs = graphs
    model = Model().eval()
    with torch.no_grad():
        count = regions.compile_block_groups(
            _plain_compile, model, compile_kwargs=compile_kwargs
        )
        for _ in range(2):
            model(torch.randn(2, 8))
    assert count == 2
    assert len(captured) == 2


@pytest.mark.parametrize("size", [2, 4, 6])
def test_outputs_feed_keyword_and_positional_slots(monkeypatch, graphs, size):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", str(size))
    captured, compile_kwargs = graphs
    torch.manual_seed(0)
    model = ResidualModel().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)
        count = regions.compile_block_groups(
            _plain_compile, model, compile_kwargs=compile_kwargs
        )
        outputs = [model(x) for _ in range(3)]
    for output in outputs:
        torch.testing.assert_close(output, expected)
    assert len(captured) == count == math.ceil(len(model.layers) / size)


def test_outputs_changed_between_blocks_are_rejected(monkeypatch, graphs):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", "2")
    _, compile_kwargs = graphs
    model = ResidualModel(depth=2).eval()
    regions.compile_block_groups(_plain_compile, model, compile_kwargs=compile_kwargs)
    with torch.no_grad(), pytest.raises(RuntimeError, match="not passed unchanged"):
        model(torch.randn(2, 8), rescale=True)


def test_one_block_per_region_keeps_upstream_behaviour(monkeypatch, graphs):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", "1")
    captured, compile_kwargs = graphs
    model = Model(depth=3).eval()
    with torch.no_grad():
        count = regions.compile_block_groups(
            _plain_compile, model, compile_kwargs=compile_kwargs
        )
        model(torch.randn(2, 8))
    assert count == 3
    assert not any(isinstance(m.forward, regions.DeferredBlock) for m in model.blocks)


def test_groups_stay_inside_one_module_list(monkeypatch, graphs):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", "4")
    _, compile_kwargs = graphs
    model = Model(depth=3).eval()
    model.extra = nn.ModuleList(Block(8) for _ in range(2))
    count = regions.compile_block_groups(
        _plain_compile, model, compile_kwargs=compile_kwargs
    )
    assert count == 2
    assert model.blocks[2].forward.group is not model.extra[0].forward.group


def test_out_of_order_block_call_is_rejected(monkeypatch, graphs):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", "2")
    _, compile_kwargs = graphs
    model = Model(depth=2).eval()
    regions.compile_block_groups(_plain_compile, model, compile_kwargs=compile_kwargs)
    with torch.no_grad(), pytest.raises(RuntimeError, match="out of order"):
        model.blocks[1](torch.randn(2, 8), torch.zeros(()), 1.0, shift=0.0)


def test_registers_an_around_hook():
    assert regions.register() == regions.HOOK_TARGET
    assert any(
        hook is regions.compile_block_groups
        for _, hook, _ in HookRegistry._hooks[regions.HOOK_TARGET]
    )

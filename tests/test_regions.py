# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import contextvars
import math
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln.compilation import regions, shapes

pytestmark = pytest.mark.usefixtures("rbln_platform")


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()


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
        assert torch.equal(output, expected)
    assert len(captured) == count


@pytest.mark.parametrize("keep_outputs", [False, True], ids=["last", "kept"])
def test_first_call_returns_the_compiled_region_output(keep_outputs):
    runs = []

    def backend(graph, example_inputs):
        def run(*args):
            runs.append(len(runs))
            return [output + 1 for output in graph(*args)]

        return run

    torch.manual_seed(0)
    model = (RecordingModel if keep_outputs else Model)(depth=4).eval()
    x = torch.randn(2, 8)
    regions.group_blocks(
        list(model.blocks), None, {"backend": backend, "dynamic": False}, keep_outputs
    )
    with torch.no_grad():
        first, second = model(x), model(x)
    assert runs == [0, 1]
    if keep_outputs:
        assert all(map(torch.equal, first, second))
    else:
        assert torch.equal(first, second)


_steps: contextvars.ContextVar[list] = contextvars.ContextVar("steps")


class KeyCache:
    def __init__(self) -> None:
        self.keys: dict[int, torch.Tensor] = {}

    def update(self, layer: int, key: torch.Tensor) -> torch.Tensor:
        if layer in self.keys:
            key = torch.cat([self.keys[layer], key], dim=0)
        self.keys[layer] = key
        return key


class KeyCarryingBlock(nn.Module):
    def __init__(self, layer: int, width: int) -> None:
        super().__init__()
        self.layer = layer
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)

    def forward(self, x, *, cache):
        keys = cache.update(self.layer, self.key(x))
        steps = _steps.get()
        steps.append(keys.shape[0])
        scores = torch.softmax(self.query(x) @ keys.T, dim=-1)
        return x + scores @ keys * len(steps)


class KeyCarryingModel(nn.Module):
    def __init__(self, depth: int = 3, width: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(KeyCarryingBlock(i, width) for i in range(depth))

    def forward(self, x):
        token = _steps.set([])
        try:
            cache = KeyCache()
            for block in self.blocks:
                x = block(x, cache=cache)
            return (
                x,
                {layer: key.shape for layer, key in cache.keys.items()},
                _steps.get(),
            )
        finally:
            _steps.reset(token)


def test_first_call_of_key_carrying_blocks_runs_compiled_and_matches_eager():
    runs = []

    def backend(graph, example_inputs):
        def run(*args):
            runs.append(len(runs))
            return graph(*args)

        return run

    torch.manual_seed(0)
    model = KeyCarryingModel().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)
        regions.group_blocks(
            list(model.blocks), None, {"backend": backend, "dynamic": False}
        )
        first = model(x)
        compiled_runs = len(runs)
        second = model(x)
    assert compiled_runs > 0
    for output in (first, second):
        assert torch.equal(output[0], expected[0])
        assert output[1:] == expected[1:]
    assert expected[1] == {layer: torch.Size([2, 8]) for layer in range(3)}


def test_default_compiles_each_block_list_as_one_region(monkeypatch, graphs):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    captured, compile_kwargs = graphs
    model = Model().eval()
    with torch.no_grad():
        count = regions.compile_block_groups(
            _plain_compile, model, compile_kwargs=compile_kwargs
        )
        for _ in range(2):
            model(torch.randn(2, 8))
    assert count == 1
    assert len(captured) == 1


def test_a_module_compiled_by_two_stages_keeps_one_region(monkeypatch, graphs):
    """Each denoising stage of a two-stage pipeline compiles the shared transformer."""
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    captured, compile_kwargs = graphs
    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)
        counts = [
            regions.compile_block_groups(
                _plain_compile, model, compile_kwargs=compile_kwargs
            )
            for _ in range(2)
        ]
        outputs = [model(x) for _ in range(2)]
    assert counts == [1, 1]
    for output in outputs:
        assert torch.equal(output, expected)
    assert len(captured) == 1


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
        assert torch.equal(output, expected)
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


BLOCK_BYTES = 100


def _blocks_in(graph: torch.fx.GraphModule) -> int:
    return sum(node.target is torch.tanh for node in graph.graph.nodes)


@pytest.fixture
def device_memory(monkeypatch):
    import rebel

    queried = []
    free = {"bytes": 0}

    def get_device_memory_info(index):
        queried.append(index)
        return SimpleNamespace(free=free["bytes"], granularity=BLOCK_BYTES // 10)

    monkeypatch.setattr(rebel, "get_device_memory_info", get_device_memory_info)
    return free, queried


@pytest.fixture
def sized_graphs(device_memory):
    free, _ = device_memory
    sizes = []

    def backend(graph, example_inputs):
        blocks = _blocks_in(graph)
        sizes.append(blocks)
        compiled = SimpleNamespace(get_alloc_per_node=lambda: [blocks * BLOCK_BYTES])
        regions.check_fits(compiled)
        return graph

    return free, sizes, {"backend": backend, "dynamic": False}


def _grouped_run(model, compile_kwargs):
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)
        regions.compile_block_groups(
            _plain_compile, model, compile_kwargs=compile_kwargs
        )
        outputs = [model(x) for _ in range(2)]
    for output in outputs:
        assert torch.equal(output, expected)


@pytest.mark.parametrize(
    ("room", "sizes", "splits"),
    [
        pytest.param(4, [4], 0, id="fits"),
        pytest.param(2, [4, 2, 2], 1, id="one-halving"),
        pytest.param(1, [4, 2, 1, 1, 2, 1, 1], 3, id="halved-to-one"),
    ],
)
def test_region_halves_until_the_device_holds_it(
    monkeypatch, sized_graphs, caplog, room, sizes, splits
):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    free, compiled, compile_kwargs = sized_graphs
    free["bytes"] = room * BLOCK_BYTES + BLOCK_BYTES // 10
    torch.manual_seed(0)
    _grouped_run(Model(depth=4).eval(), compile_kwargs)
    assert compiled == sizes
    logged = [
        record for record in caplog.records if "does not fit" in record.getMessage()
    ]
    assert len(logged) == splits


@contextlib.contextmanager
def request_shape(height: int):
    dims = (("height", height), ("width", height), ("num_frames", 1))
    token = shapes.current_shape_key.set(shapes.ShapeKey("Pipeline", dims))
    try:
        yield
    finally:
        shapes.current_shape_key.reset(token)


def _room(blocks: int) -> int:
    return blocks * BLOCK_BYTES + BLOCK_BYTES // 10


@pytest.fixture
def two_shape_model(sized_graphs):
    free, compiled, compile_kwargs = sized_graphs
    torch.manual_seed(0)
    model = Model(depth=4).eval()
    inputs = {2: torch.randn(2, 8), 6: torch.randn(6, 8)}
    with torch.no_grad():
        expected = {rows: model(x) for rows, x in inputs.items()}
    regions.compile_block_groups(_plain_compile, model, compile_kwargs=compile_kwargs)
    return model, inputs, expected, free, compiled


def test_a_later_shape_that_does_not_fit_splits_only_itself(two_shape_model, caplog):
    model, inputs, expected, free, compiled = two_shape_model
    free["bytes"] = _room(4)
    with torch.no_grad(), request_shape(2):
        first = model(inputs[2])
    free["bytes"] = _room(2)
    with torch.no_grad(), request_shape(6):
        second = [model(inputs[6]) for _ in range(2)]
    with torch.no_grad(), request_shape(2):
        first_again = model(inputs[2])
    assert compiled == [4, 4, 2, 2]
    assert all(torch.equal(output, expected[2]) for output in (first, first_again))
    assert all(torch.equal(output, expected[6]) for output in second)
    splits = [r for r in caplog.records if "does not fit" in r.getMessage()]
    assert len(splits) == 1
    assert "height=6" in splits[0].getMessage()


def test_a_later_shape_that_fits_keeps_the_whole_region(two_shape_model):
    model, inputs, expected, free, compiled = two_shape_model
    free["bytes"] = _room(4)
    with torch.no_grad():
        for rows in (2, 6, 2, 6):
            with request_shape(rows):
                assert torch.equal(model(inputs[rows]), expected[rows])
    assert compiled == [4, 4]


def test_a_second_tensor_shape_under_one_shape_key_is_fit_checked(two_shape_model):
    model, inputs, expected, free, compiled = two_shape_model
    free["bytes"] = _room(4)
    with torch.no_grad(), request_shape(2):
        model(inputs[2])
        free["bytes"] = _room(2)
        assert torch.equal(model(inputs[6]), expected[6])
    assert compiled == [4, 4, 2, 2]


class RecordingModel(Model):
    def forward(self, x):
        y, seen = torch.zeros(()), []
        for index, block in enumerate(self.blocks):
            x, y = block(x, y, 1.0, shift=index * 0.1)
            seen.append(x)
        return seen


@pytest.mark.parametrize("shape", [(2, 8), (3, 16)], ids=str)
@pytest.mark.parametrize("room", [4, 2, 1], ids=["fits", "halved", "single"])
def test_kept_outputs_reach_every_block_caller(sized_graphs, room, shape):
    free, compiled, compile_kwargs = sized_graphs
    free["bytes"] = room * BLOCK_BYTES + BLOCK_BYTES // 10
    torch.manual_seed(0)
    model = RecordingModel(depth=4, width=shape[-1]).eval()
    x = torch.randn(shape)
    with torch.no_grad():
        expected = model(x)
        regions.group_blocks(list(model.blocks), None, compile_kwargs, True)
        runs = [model(x) for _ in range(2)]
    assert compiled[0] == 4
    for seen in runs:
        assert len(seen) == len(expected)
        assert all(map(torch.equal, seen, expected))


@pytest.mark.parametrize(
    "message",
    [
        "Lack of device memory, key=Kernel",
        "(System) code=510 SYS_ENOMEM: Out of memory. (Failed to allocate device "
        "memory, key=Input, size=12478054400, node_id=0)",
    ],
)
def test_load_time_out_of_memory_halves_the_region(monkeypatch, message):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    sizes = []

    def backend(graph, example_inputs):
        sizes.append(_blocks_in(graph))
        if sizes[-1] > 2:
            raise RuntimeError(message)
        return graph

    torch.manual_seed(0)
    _grouped_run(Model(depth=4).eval(), {"backend": backend, "dynamic": False})
    assert sizes == [4, 2, 2]


def test_a_block_that_never_fits_raises(monkeypatch, sized_graphs):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    free, compiled, compile_kwargs = sized_graphs
    free["bytes"] = BLOCK_BYTES // 2
    model = Model(depth=2).eval()
    regions.compile_block_groups(_plain_compile, model, compile_kwargs=compile_kwargs)
    with torch.no_grad(), pytest.raises(Exception, match="free"):
        model(torch.randn(2, 8))
    assert compiled == [2, 1]


def test_other_compile_errors_do_not_split(monkeypatch):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    sizes = []

    def backend(graph, example_inputs):
        sizes.append(_blocks_in(graph))
        raise RuntimeError("unsupported operator")

    model = Model(depth=4).eval()
    regions.compile_block_groups(
        _plain_compile, model, compile_kwargs={"backend": backend, "dynamic": False}
    )
    with torch.no_grad(), pytest.raises(Exception, match="unsupported operator"):
        model(torch.randn(2, 8))
    assert sizes == [4]


def test_installed_check_covers_a_module_no_block_group_compiled(
    device_memory, monkeypatch
):
    """A VAE coder or vocoder reaches the device through Module.compile, which
    `checking_fit` never wrapped: it used to fail with a bare SYS_ENOMEM."""
    import rebel

    free, _ = device_memory
    free["bytes"] = BLOCK_BYTES
    monkeypatch.setattr(regions, "_installed", False)
    original = rebel.RBLNCompiledModel.create_runtime
    monkeypatch.setattr(
        rebel.RBLNCompiledModel, "create_runtime", original, raising=False
    )
    assert regions.install_fit_check()
    compiled = SimpleNamespace(get_alloc_per_node=lambda: [4 * BLOCK_BYTES])
    with pytest.raises(regions.RegionDoesNotFit) as error:
        rebel.RBLNCompiledModel.create_runtime(compiled, device=0)
    assert error.value.needed == [4 * BLOCK_BYTES]
    assert not regions.install_fit_check()


def test_runtime_creation_checks_every_node_it_loads_on(device_memory):
    import rebel

    free, queried = device_memory
    free["bytes"] = 3 * BLOCK_BYTES
    original = rebel.RBLNCompiledModel.create_runtime
    compiled = SimpleNamespace(
        get_alloc_per_node=lambda: [BLOCK_BYTES, 4 * BLOCK_BYTES]
    )
    with regions.checking_fit(), pytest.raises(regions.RegionDoesNotFit) as error:
        rebel.RBLNCompiledModel.create_runtime(compiled, device=2)
    assert queried == [2, 3]
    assert error.value.needed == [BLOCK_BYTES, 4 * BLOCK_BYTES]
    assert rebel.RBLNCompiledModel.create_runtime is original

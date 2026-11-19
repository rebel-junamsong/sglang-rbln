# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import contextvars
import gc
import math
import weakref
from types import SimpleNamespace

import pytest
import torch
from rebel.core.exception import RBLNDeviceMemoryError
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln.compilation import backend as compilation
from sglang_rbln.compilation import placement, regions, residency, shapes

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
def capturing_backend():
    captured = []

    def backend(graph, example_inputs):
        captured.append(graph)
        return graph

    return captured, {"backend": backend, "dynamic": False}


def _upstream_compile(module, *, compile_kwargs):
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        compile_matching_submodules,
    )

    return compile_matching_submodules(module, compile_kwargs=compile_kwargs)


@pytest.mark.parametrize("size", [2, 3, 7])
def test_grouped_blocks_match_eager(monkeypatch, capturing_backend, size):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", str(size))
    captured, compile_kwargs = capturing_backend
    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)
        count = regions.compile_block_groups(
            _upstream_compile, model, compile_kwargs=compile_kwargs
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
    model = (EveryOutputModel if keep_outputs else Model)(depth=4).eval()
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


def test_default_compiles_each_block_list_as_one_region(monkeypatch, capturing_backend):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    captured, compile_kwargs = capturing_backend
    model = Model().eval()
    with torch.no_grad():
        count = regions.compile_block_groups(
            _upstream_compile, model, compile_kwargs=compile_kwargs
        )
        for _ in range(2):
            model(torch.randn(2, 8))
    assert count == 1
    assert len(captured) == 1


def test_a_module_compiled_by_two_stages_keeps_one_region(
    monkeypatch, capturing_backend
):
    """Each denoising stage of a two-stage pipeline compiles the shared transformer."""
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    captured, compile_kwargs = capturing_backend
    torch.manual_seed(0)
    model = Model().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)
        counts = [
            regions.compile_block_groups(
                _upstream_compile, model, compile_kwargs=compile_kwargs
            )
            for _ in range(2)
        ]
        outputs = [model(x) for _ in range(2)]
    assert counts == [1, 1]
    for output in outputs:
        assert torch.equal(output, expected)
    assert len(captured) == 1


@pytest.mark.parametrize("size", [2, 4, 6])
def test_outputs_feed_keyword_and_positional_slots(
    monkeypatch, capturing_backend, size
):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", str(size))
    captured, compile_kwargs = capturing_backend
    torch.manual_seed(0)
    model = ResidualModel().eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)
        count = regions.compile_block_groups(
            _upstream_compile, model, compile_kwargs=compile_kwargs
        )
        outputs = [model(x) for _ in range(3)]
    for output in outputs:
        assert torch.equal(output, expected)
    assert len(captured) == count == math.ceil(len(model.layers) / size)


def test_outputs_changed_between_blocks_are_rejected(monkeypatch, capturing_backend):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", "2")
    _, compile_kwargs = capturing_backend
    model = ResidualModel(depth=2).eval()
    regions.compile_block_groups(
        _upstream_compile, model, compile_kwargs=compile_kwargs
    )
    with torch.no_grad(), pytest.raises(RuntimeError, match="not passed unchanged"):
        model(torch.randn(2, 8), rescale=True)


def test_one_block_per_region_keeps_upstream_behaviour(monkeypatch, capturing_backend):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", "1")
    captured, compile_kwargs = capturing_backend
    model = Model(depth=3).eval()
    with torch.no_grad():
        count = regions.compile_block_groups(
            _upstream_compile, model, compile_kwargs=compile_kwargs
        )
        model(torch.randn(2, 8))
    assert count == 3
    assert not any(isinstance(m.forward, regions.DeferredBlock) for m in model.blocks)


def test_groups_stay_inside_one_module_list(monkeypatch, capturing_backend):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", "4")
    _, compile_kwargs = capturing_backend
    model = Model(depth=3).eval()
    model.extra = nn.ModuleList(Block(8) for _ in range(2))
    count = regions.compile_block_groups(
        _upstream_compile, model, compile_kwargs=compile_kwargs
    )
    assert count == 2
    assert model.blocks[2].forward.group is not model.extra[0].forward.group


def test_out_of_order_block_call_is_rejected(monkeypatch, capturing_backend):
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", "2")
    _, compile_kwargs = capturing_backend
    model = Model(depth=2).eval()
    regions.compile_block_groups(
        _upstream_compile, model, compile_kwargs=compile_kwargs
    )
    with torch.no_grad(), pytest.raises(RuntimeError, match="out of order"):
        model.blocks[1](torch.randn(2, 8), torch.zeros(()), 1.0, shift=0.0)


def test_registers_around_the_upstream_block_compile():
    assert regions.register() == regions.HOOK_TARGET
    assert any(
        hook is regions.compile_block_groups
        for _, hook, _ in HookRegistry._hooks[regions.HOOK_TARGET]
    )


BLOCK_BYTES = 100


def _room_for(blocks: int) -> int:
    """Free memory that holds `blocks` blocks and no more."""
    return blocks * BLOCK_BYTES + BLOCK_BYTES // 10


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
    monkeypatch.setattr(rebel, "get_evictable_memory", lambda index: 0)
    return free, queried


@pytest.fixture
def fit_checked_backend(device_memory):
    free, _ = device_memory
    sizes = []

    def backend(graph, example_inputs):
        blocks = _blocks_in(graph)
        sizes.append(blocks)
        compiled = SimpleNamespace(get_alloc_per_node=lambda: [blocks * BLOCK_BYTES])
        regions.check_fits(compiled)
        return graph

    return free, sizes, {"backend": backend, "dynamic": False}


def _assert_grouped_run_matches_eager(model, compile_kwargs):
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)
        regions.compile_block_groups(
            _upstream_compile, model, compile_kwargs=compile_kwargs
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
    monkeypatch, fit_checked_backend, caplog, room, sizes, splits
):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    free, compiled, compile_kwargs = fit_checked_backend
    free["bytes"] = _room_for(room)
    torch.manual_seed(0)
    _assert_grouped_run_matches_eager(Model(depth=4).eval(), compile_kwargs)
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


@pytest.fixture
def two_shape_model(fit_checked_backend):
    free, compiled, compile_kwargs = fit_checked_backend
    torch.manual_seed(0)
    model = Model(depth=4).eval()
    inputs = {2: torch.randn(2, 8), 6: torch.randn(6, 8)}
    with torch.no_grad():
        expected = {rows: model(x) for rows, x in inputs.items()}
    regions.compile_block_groups(
        _upstream_compile, model, compile_kwargs=compile_kwargs
    )
    return model, inputs, expected, free, compiled


def test_a_later_shape_that_does_not_fit_splits_only_itself(two_shape_model, caplog):
    model, inputs, expected, free, compiled = two_shape_model
    free["bytes"] = _room_for(4)
    with torch.no_grad(), request_shape(2):
        first = model(inputs[2])
    free["bytes"] = _room_for(2)
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
    free["bytes"] = _room_for(4)
    with torch.no_grad():
        for rows in (2, 6, 2, 6):
            with request_shape(rows):
                assert torch.equal(model(inputs[rows]), expected[rows])
    assert compiled == [4, 4]


def test_a_second_tensor_shape_under_one_shape_key_is_fit_checked(two_shape_model):
    model, inputs, expected, free, compiled = two_shape_model
    free["bytes"] = _room_for(4)
    with torch.no_grad(), request_shape(2):
        model(inputs[2])
        free["bytes"] = _room_for(2)
        assert torch.equal(model(inputs[6]), expected[6])
    assert compiled == [4, 4, 2, 2]


class EveryOutputModel(Model):
    def forward(self, x):
        y, seen = torch.zeros(()), []
        for index, block in enumerate(self.blocks):
            x, y = block(x, y, 1.0, shift=index * 0.1)
            seen.append(x)
        return seen


@pytest.mark.parametrize("shape", [(2, 8), (3, 16)], ids=str)
@pytest.mark.parametrize("room", [4, 2, 1], ids=["fits", "halved", "single"])
def test_kept_outputs_reach_every_block_caller(fit_checked_backend, room, shape):
    free, compiled, compile_kwargs = fit_checked_backend
    free["bytes"] = _room_for(room)
    torch.manual_seed(0)
    model = EveryOutputModel(depth=4, width=shape[-1]).eval()
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
    ["Lack of device memory", "SYS_ENOMEM"],
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
    _assert_grouped_run_matches_eager(
        Model(depth=4).eval(), {"backend": backend, "dynamic": False}
    )
    assert sizes == [4, 2, 2]


@pytest.mark.parametrize("wrapped", [False, True], ids=["raised", "wrapped"])
def test_a_typed_device_memory_error_halves_the_region(monkeypatch, wrapped):
    # Codegen refuses a graph past a node's DRAM with the typed error, whatever
    # its text; dynamo hands it on as the cause of its own error.
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    sizes = []

    def backend(graph, example_inputs):
        sizes.append(_blocks_in(graph))
        if sizes[-1] > 2:
            refusal = RBLNDeviceMemoryError("node 0 needs 20310117248 B")
            if not wrapped:
                raise refusal
            raise RuntimeError("backend failed") from refusal
        return graph

    torch.manual_seed(0)
    _assert_grouped_run_matches_eager(
        Model(depth=4).eval(), {"backend": backend, "dynamic": False}
    )
    assert sizes == [4, 2, 2]


def test_a_block_that_never_fits_raises(monkeypatch, fit_checked_backend):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    free, compiled, compile_kwargs = fit_checked_backend
    free["bytes"] = BLOCK_BYTES // 2
    model = Model(depth=2).eval()
    regions.compile_block_groups(
        _upstream_compile, model, compile_kwargs=compile_kwargs
    )
    with torch.no_grad(), pytest.raises(Exception, match="free"):
        model(torch.randn(2, 8))
    assert compiled == [2, 1]


def test_a_region_that_ran_out_after_loading_releases_its_graphs(monkeypatch):
    # An abandoned attempt must not keep what it loaded, or one oversized shape
    # leaves the DiT's NPUs full and every later request fails.
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    released = []
    monkeypatch.setattr(
        regions, "release_graphs", lambda group: released.append(len(group.blocks))
    )

    def backend(graph, example_inputs):
        if _blocks_in(graph) > 1:
            regions.runtimes_created += 1  # loaded, then out of memory at the run
            raise RuntimeError("SYS_ENOMEM")
        return graph

    torch.manual_seed(0)
    _assert_grouped_run_matches_eager(
        Model(depth=4).eval(), {"backend": backend, "dynamic": False}
    )
    assert released == [4, 2, 2]  # each failed group, before its halves


def test_a_released_attempt_holds_no_runtime(monkeypatch):
    # A caught error keeps the failed call's frames, and the runtime they hold,
    # alive. Releasing before they are dropped would free nothing, and the
    # abandoned attempts' buffers (1.4 GiB) would pile up beside the smaller graphs.
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    runtimes, alive = [], []

    class Runtime:
        pass

    def backend(graph, example_inputs):
        if _blocks_in(graph) > 1:
            runtime = Runtime()
            runtimes.append(weakref.ref(runtime))
            regions.runtimes_created += 1
            raise RuntimeError("SYS_ENOMEM")
        return graph

    def release(group):
        gc.collect()
        alive.append(runtimes[-1]() is not None)

    monkeypatch.setattr(regions, "release_graphs", release)
    torch.manual_seed(0)
    _assert_grouped_run_matches_eager(
        Model(depth=4).eval(), {"backend": backend, "dynamic": False}
    )
    assert alive == [False, False, False]


def test_a_refusal_that_loaded_nothing_releases_nothing(
    monkeypatch, fit_checked_backend
):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    released = []
    monkeypatch.setattr(regions, "release_graphs", released.append)
    free, _, compile_kwargs = fit_checked_backend
    free["bytes"] = BLOCK_BYTES // 2
    model = Model(depth=2).eval()
    regions.compile_block_groups(
        _upstream_compile, model, compile_kwargs=compile_kwargs
    )
    with torch.no_grad(), pytest.raises(Exception, match="free"):
        model(torch.randn(2, 8))
    assert released == []


def test_other_compile_errors_do_not_split(monkeypatch):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    sizes = []

    def backend(graph, example_inputs):
        sizes.append(_blocks_in(graph))
        raise RuntimeError("unsupported operator")

    model = Model(depth=4).eval()
    regions.compile_block_groups(
        _upstream_compile, model, compile_kwargs={"backend": backend, "dynamic": False}
    )
    with torch.no_grad(), pytest.raises(Exception, match="unsupported operator"):
        model(torch.randn(2, 8))
    assert sizes == [4]


def test_modules_compiled_outside_block_regions_are_fit_checked_too(
    device_memory, monkeypatch
):
    """A module compiled outside any block region, such as a VAE coder or vocoder,
    gets the same fit check and raises RegionDoesNotFit with the memory it needs."""
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
    create_runtime = regions._checked_create_runtime(
        rebel.RBLNCompiledModel.create_runtime
    )
    compiled = SimpleNamespace(
        get_alloc_per_node=lambda: [BLOCK_BYTES, 4 * BLOCK_BYTES]
    )
    with pytest.raises(regions.RegionDoesNotFit) as error:
        create_runtime(compiled, device=2)
    assert queried == [2, 3]
    assert error.value.needed == [BLOCK_BYTES, 4 * BLOCK_BYTES]


def test_a_region_runs_on_the_npus_of_its_blocks():
    model = Model(depth=4).eval()
    placement.place(model, (4, 5, 6, 7))
    groups = regions.group_blocks(
        list(model.blocks), 2, {"backend": "eager", "dynamic": False}
    )
    assert [placement.devices_of(region.group) for region in groups] == [
        (4, 5, 6, 7)
    ] * 2
    assert (
        placement.devices_of(
            regions.Region([Block(8)], {"backend": "eager", "dynamic": False}).group
        )
        is None
    )


def test_kept_outputs_stay_on_the_region_npus(monkeypatch, dummy_device):
    # The text encoder keeps every layer's output for the connector, which reads
    # them on the NPUs; a round trip through the host would cost a copy each way.

    def backend(graph, example_inputs):
        return lambda *args: [out.to("rbln:0") for out in graph(*args)]

    model = EveryOutputModel(depth=4).eval()
    regions.group_blocks(
        list(model.blocks), None, {"backend": backend, "dynamic": False}, True
    )
    with torch.no_grad():
        model(torch.randn(2, 8))  # block 0 runs on fake tensors while learning
        with residency.DeviceBoundary():
            seen = model(torch.randn(2, 8))
    assert [out.device.type for out in seen] == ["rbln"] * 4


class SessionRegion(SimpleNamespace):
    """A compiled region whose reported size leaves out memory it shares with others."""

    def __init__(self, counted: int, intermediate: int, session: str = "s") -> None:
        super().__init__(counted=counted, intermediate=intermediate, session=session)

    def get_alloc_per_node(self):
        return [self.counted]

    def get_alloc_per_node_by_key(self):
        return {"Kernel": [self.counted], "Intermediate": [self.intermediate]}

    def select_executor(self):
        return SimpleNamespace(get_context_id=lambda: self.session)


@pytest.fixture
def no_shared_context_memory(monkeypatch):
    monkeypatch.setattr(regions, "_pools", {}, raising=False)


def test_a_region_needing_more_than_free_memory_at_run_does_not_fit(
    device_memory, no_shared_context_memory
):
    # A region whose reported size fits in free memory, but which needs more once it
    # runs, is refused before it loads.
    free, _ = device_memory
    create_runtime = regions._checked_create_runtime(lambda model, **kwargs: model)
    held, counted, run_time = 2_279_484_288, 1_545_601_024, 3_939_358_336
    free["bytes"] = 4_000_000_000
    create_runtime(SessionRegion(0, held), device=0)
    free["bytes"] = 2_057_306_112 + BLOCK_BYTES // 10
    with pytest.raises(regions.RegionDoesNotFit) as error:
        create_runtime(SessionRegion(counted, run_time), device=0)
    assert error.value.needed == [counted + run_time - held]


def test_a_region_reusing_memory_already_held_needs_no_more(
    device_memory, no_shared_context_memory
):
    # It needs more than is free, but memory already held by its compile context
    # covers it.
    free, _ = device_memory
    create_runtime = regions._checked_create_runtime(lambda model, **kwargs: model)
    free["bytes"] = 6 * BLOCK_BYTES
    create_runtime(SessionRegion(0, 5 * BLOCK_BYTES), device=0)
    free["bytes"] = 2 * BLOCK_BYTES
    create_runtime(SessionRegion(BLOCK_BYTES, 5 * BLOCK_BYTES), device=0)
    # a region in another compile context gets no such credit
    with pytest.raises(regions.RegionDoesNotFit):
        create_runtime(SessionRegion(BLOCK_BYTES, 5 * BLOCK_BYTES, "t"), device=0)


def test_memory_the_runtime_can_free_counts_as_room(device_memory, monkeypatch):
    # Otherwise a region that would fit is split and recompiled needlessly.
    import rebel

    free, _ = device_memory
    free["bytes"] = 2 * BLOCK_BYTES + BLOCK_BYTES // 10
    region = SimpleNamespace(get_alloc_per_node=lambda: [5 * BLOCK_BYTES])
    with pytest.raises(regions.RegionDoesNotFit):  # nothing the runtime can free
        regions.check_fits(region, 0)
    asked = []
    monkeypatch.setattr(
        rebel,
        "get_evictable_memory",
        lambda index: asked.append(index) or 3 * BLOCK_BYTES,
    )
    regions.check_fits(region, 0)
    assert asked == [0]


def test_a_device_without_memory_info_has_nothing_to_check(monkeypatch):
    # On the dummy device of a compile-only run there is no memory information, so
    # there is nothing to check.
    import rebel

    monkeypatch.setattr(rebel, "get_device_memory_info", lambda index: None)
    region = SimpleNamespace(get_alloc_per_node=lambda: [5 * BLOCK_BYTES])
    assert regions.check_fits(region, 0) is None


def test_a_split_region_reuses_its_parents_compile_context(monkeypatch):
    monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    context = object()

    def backend(graph, example_inputs):
        if _blocks_in(graph) > 2:
            raise regions.RegionDoesNotFit([4 * BLOCK_BYTES], [2 * BLOCK_BYTES])
        return graph

    compiled = []
    original = nn.Module.compile

    def compile_recording(module, **kwargs):
        compiled.append(kwargs["options"]["compile_context"])
        return original(module, backend=backend, dynamic=False)

    monkeypatch.setattr(nn.Module, "compile", compile_recording)
    model = Model(depth=4).eval()
    regions.group_blocks(
        list(model.blocks), None, {"options": {"compile_context": context}}
    )
    with torch.no_grad():
        model(torch.randn(2, 8))
    assert len(compiled) == 3 and all(each is context for each in compiled)


def test_a_watched_output_still_reaches_its_placeholder():
    # A split region's output is the next region's input, which dynamo watches weakly.
    value = torch.arange(6.0).reshape(2, 3)
    value._rbln_producer_view = "view"
    watched = weakref.ref(value)
    placeholder = torch.empty(2, 3, device="meta")
    regions._become(placeholder, value)
    assert watched() is value
    assert placeholder.data_ptr() == value.data_ptr()
    assert torch.equal(placeholder, value)
    assert placeholder._rbln_producer_view == "view"


class RecordingBlock(Block):
    """A block that records the tensor types it runs on."""

    seen: list[type] = []

    def forward(self, x, y, scale, *, shift):
        RecordingBlock.seen.append(type(x))
        return super().forward(x, y, scale, shift=shift)


class DataReadingBlock(Block):
    def forward(self, x, y, scale, *, shift):
        scale = scale * float(x.abs().max().item() > 0)
        return super().forward(x, y, scale, shift=shift)


def _first_call(block_type, monkeypatch, capturing_backend):
    """The eager output of a 4-block model, then its output on the region's first
    call."""
    monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", "4")
    _, compile_kwargs = capturing_backend
    torch.manual_seed(0)
    model = Model(depth=4).eval()
    for index, block in enumerate(model.blocks):
        model.blocks[index] = block_type(8)
        model.blocks[index].load_state_dict(block.state_dict())
    x = torch.randn(2, 8)
    with torch.no_grad():
        expected = model(x)
        RecordingBlock.seen = []
        regions.compile_block_groups(
            _upstream_compile, model, compile_kwargs=compile_kwargs
        )
        return expected, model(x)


def test_block_zero_learns_its_slots_on_fake_tensors(monkeypatch, capturing_backend):
    from torch._subclasses.fake_tensor import FakeTensor

    expected, actual = _first_call(RecordingBlock, monkeypatch, capturing_backend)
    assert torch.equal(actual, expected)
    # the learning pass; the region's trace records after it
    assert RecordingBlock.seen[0] is FakeTensor


def test_a_block_that_reads_its_data_learns_its_slots_eagerly(
    monkeypatch, capturing_backend
):
    expected, actual = _first_call(DataReadingBlock, monkeypatch, capturing_backend)
    assert torch.equal(actual, expected)


class EagerDeviceOps(torch.overrides.TorchFunctionMode):
    """The torch calls run eagerly on rbln tensors, by name."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # traced or under fake dispatch, an op runs nothing
        if torch.compiler.is_compiling() or residency._under_fake_mode():
            return func(*args, **kwargs)
        if any(
            isinstance(leaf, torch.Tensor) and leaf.device.type == "rbln"
            for leaf in [*args, *kwargs.values()]
        ):
            self.names.append(getattr(func, "__name__", str(func)))
        return func(*args, **kwargs)


class TanhBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, x, *, shift):
        return torch.tanh(self.linear(x) + shift)


def test_a_region_first_call_runs_no_block_op_on_the_npus(
    dummy_device, monkeypatch, tmp_path
):
    """Eager on the NPUs, block 0's learning pass compiled each of its ops as a
    single-op graph on one NPU: 29 graphs for the two LTX-2 connectors."""
    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compilation.ensure_compile_backend()
    torch.manual_seed(0)
    blocks = [TanhBlock(64).to(torch.bfloat16).to("rbln:0") for _ in range(4)]
    regions.group_blocks(blocks, None, compilation.static_compile_kwargs(blocks[0]))
    eager = EagerDeviceOps()
    h = torch.randn(2, 64, dtype=torch.bfloat16).to("rbln:0")
    with torch.no_grad(), residency.DeviceBoundary():
        # block 0 learns; a mode traced into the last block's compile would split it
        with eager:
            for index, block in enumerate(blocks[:-1]):
                h = block(h, shift=index * 0.1)
        h = blocks[-1](h, shift=0.3)
    assert (h.device.type, tuple(h.shape)) == ("rbln", (2, 64))
    assert len(list(tmp_path.rglob("*.rbln"))) == 1
    assert not {"clone", "linear", "tanh", "add"} & set(eager.names)

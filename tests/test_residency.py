# SPDX-License-Identifier: Apache-2.0
import os
import sys
import types

import pytest
import torch
from torch import nn

from sglang_rbln import device
from sglang_rbln.compilation import placement, residency
from sglang_rbln.compilation.backend import COMPILE_BACKEND_NAME

# These need a device to name, not the hardware: placement and graph structure
# only. A dummy device refuses to execute a compiled graph and its eager outputs
# are garbage, so nothing here may assert on values from one.
pytestmark = pytest.mark.usefixtures("dummy_device")


@pytest.mark.parametrize(
    ("mode", "num_devices", "expected"),
    [
        ("resident", "4", True),
        ("resident", "1", False),
        ("resident", None, False),
        ("staged", "4", False),
    ],
)
def test_resident_regions_need_device_mode_and_several_npus(
    monkeypatch, mode, num_devices, expected
):
    monkeypatch.setenv("SGLANG_RBLN_EXECUTION_MODE", mode)
    if num_devices is None:
        monkeypatch.delenv("SGLANG_RBLN_NUM_DEVICES", raising=False)
    else:
        monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", num_devices)
    assert device.resident_regions() is expected


def _compiled_block() -> nn.Module:
    block = nn.Linear(4, 4)
    block.compile(backend="eager")
    return block


def test_install_wraps_a_compiled_call_once():
    block = _compiled_block()
    residency.install(block)
    wrapped = block._compiled_call_impl
    residency.install(block)
    assert block._compiled_call_impl is wrapped
    assert wrapped._rbln_resident


def test_install_skips_an_uncompiled_module():
    block = nn.Linear(4, 4)
    residency.install(block)
    assert block._compiled_call_impl is None


@pytest.mark.parametrize(
    ("backend", "wrapped"), [(COMPILE_BACKEND_NAME, True), ("eager", False)]
)
def test_only_rbln_compiles_become_resident(backend, wrapped):
    block = _compiled_block()
    original = block._compiled_call_impl
    residency.install_after_compile(None, block, backend=backend)
    assert (block._compiled_call_impl is not original) is wrapped


def test_boundary_leaves_host_values_alone():
    hidden = torch.arange(6.0)
    with residency.HostBoundary():
        doubled = hidden * 2
    assert doubled.device.type == "cpu"
    torch.testing.assert_close(doubled, torch.arange(0.0, 12.0, 2.0))


@pytest.mark.resident_mode
def test_boundary_brings_device_values_to_the_host():
    hidden = torch.arange(6.0, dtype=torch.bfloat16)
    resident = hidden.to("rbln:0")
    with residency.HostBoundary():
        shape = resident.shape
        summed = resident + hidden
    assert shape == hidden.shape
    assert summed.device.type == "cpu"
    torch.testing.assert_close(summed, hidden * 2)


@pytest.mark.resident_mode
def test_a_resident_call_moves_module_and_inputs_under_the_boundary():
    block = nn.Linear(4, 4)
    seen = []
    wrapped = residency.resident_call(
        block,
        lambda hidden: seen.append((hidden.device.type, block.weight.device.type)),
    )
    with residency.HostBoundary():
        wrapped(torch.ones(4))
    assert seen == [("rbln", "rbln")]


GROUP_CASES = {
    "eight": ({}, 8, [(0, 1, 2, 3), (4, 5, 6, 7)]),
    "six": ({}, 6, [(0, 1, 2, 3), (4,), (5,)]),
    "placed": (
        {"SGLANG_RBLN_COMPONENT_DEVICES": "text_encoder=4,5,6,7;vocoder=8"},
        10,
        [(0, 1, 2, 3), (4, 5, 6, 7), (8,), (9,)],
    ),
    "placed-over-spare": (
        {"SGLANG_RBLN_COMPONENT_DEVICES": "vae=4"},
        8,
        [(0, 1, 2, 3), (4,), (5,), (6,), (7,)],
    ),
}


@pytest.fixture
def layout(monkeypatch):
    def set_layout(env: dict, visible: int) -> None:
        monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
        monkeypatch.delenv("SGLANG_RBLN_COMPONENT_DEVICES", raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("RBLN_VISIBLE_DEVICES", ",".join(map(str, range(visible))))

    return set_layout


@pytest.mark.parametrize("case", GROUP_CASES)
def test_device_groups_put_each_component_set_on_its_own_device(layout, case):
    env, visible, groups = GROUP_CASES[case]
    layout(env, visible)
    assert residency.device_groups() == groups


def test_component_sets_may_not_share_npus(layout):
    layout({"SGLANG_RBLN_COMPONENT_DEVICES": "text_encoder=3,4"}, 8)
    with pytest.raises(ValueError, match="share"):
        residency.device_groups()


@pytest.mark.parametrize(
    ("given", "device_map"),
    [
        ({}, "[0,1,2,3],[4,5,6,7]"),
        ({"RBLN_DEVICE_MAP": "[0,1],[2,3]"}, "[0,1],[2,3]"),
        ({"RBLN_NPUS_PER_DEVICE": "2"}, None),
    ],
    ids=["from-layout", "map-given", "width-given"],
)
def test_device_map_follows_the_layout_unless_given(
    layout, monkeypatch, given, device_map
):
    layout({}, 8)
    for name in ("RBLN_DEVICE_MAP", "RBLN_NPUS_PER_DEVICE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in given.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setitem(sys.modules, "torch_rbln", types.ModuleType("torch_rbln"))
    residency.configure_device_group()
    assert os.environ.get("RBLN_DEVICE_MAP") == device_map


def test_a_placed_module_runs_on_the_device_of_its_npus(monkeypatch):
    rsd = types.ModuleType("torch_rbln._internal.rsd_utils")
    rsd.get_physical_device_ids = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}.get
    monkeypatch.setitem(sys.modules, "torch_rbln._internal.rsd_utils", rsd)
    monkeypatch.setattr(
        torch, "rbln", types.SimpleNamespace(device_count=lambda: 2), raising=False
    )
    encoder, dit = nn.Linear(2, 2), nn.Linear(2, 2)
    placement.place(encoder, (4, 5, 6, 7))
    assert residency.logical_device_of(encoder) == 1
    assert residency.logical_device_of(dit) == 0
    placement.place(dit, (4, 5))
    with pytest.raises(RuntimeError, match="no logical device"):
        residency.logical_device_of(dit)


def test_a_host_value_moves_once_per_version(monkeypatch):
    moves = []
    to = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor, "to", lambda self, *args: moves.append(args) or to(self, *args)
    )
    transfers, value, meta = residency._Transfers(), torch.ones(2), torch.device("meta")
    assert transfers.to(value, torch.device("cpu")) is value
    first = transfers.to(value, meta)
    assert transfers.to(value, meta) is first
    value.add_(1)
    assert transfers.to(value, meta) is not first
    assert len(moves) == 2


def test_a_resident_transfer_takes_an_inference_tensor(dummy_device):
    # a compiled call under @torch.inference_mode() -- the prompt enhancer's decode --
    # hands _Transfers a tensor with no version counter
    with torch.inference_mode():
        value = torch.zeros(4)
    residency._transfers.to(value, torch.device("rbln"))


def test_an_inference_tensor_moves_on_every_call(monkeypatch):
    # it has no version counter, so a cached copy could not tell it was updated in place
    moves = []
    to = torch.Tensor.to
    monkeypatch.setattr(
        torch.Tensor, "to", lambda self, *args: moves.append(args) or to(self, *args)
    )
    transfers, meta = residency._Transfers(), torch.device("meta")
    with torch.inference_mode():
        value = torch.ones(2)
        transfers.to(value, meta)
        value.add_(1)
        transfers.to(value, meta)
    assert len(moves) == 2


@pytest.mark.npu  # it runs an eager copy between NPU sets, which a dummy device refuses
@pytest.mark.resident_mode
def test_a_value_crosses_npu_sets_on_the_device():
    if torch.rbln.device_count() < 2:
        pytest.skip("needs two logical devices")
    value = torch.arange(4096, dtype=torch.bfloat16).reshape(64, 64).to("rbln:0")
    copied = residency._Transfers().to(value.t(), torch.device("rbln", 1))
    assert copied.device == torch.device("rbln", 1)
    assert torch.equal(copied.cpu(), value.t().cpu())


@pytest.fixture
def meta_is_resident(monkeypatch):
    monkeypatch.setattr(
        residency,
        "_is_resident",
        lambda value: isinstance(value, torch.Tensor) and value.device.type == "meta",
    )


@pytest.mark.parametrize("resident", [True, False], ids=["on-device", "on-host"])
def test_the_component_manager_leaves_resident_components_in_place(
    meta_is_resident, resident
):
    module = nn.Linear(2, 2, device="meta" if resident else "cpu")
    moved = []
    residency.keep_resident_placement(
        lambda m, **kwargs: moved.append(kwargs), module, dtype=torch.float32
    )
    assert moved == ([] if resident else [{"dtype": torch.float32}])
    assert residency.resident_is_ready(lambda m, **kwargs: False, module) is resident


def test_registers_the_placement_hooks():
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    targets = residency.register()
    for target, hook in [
        (residency.MOVE_TARGET, residency.keep_resident_placement),
        (residency.READY_TARGET, residency.resident_is_ready),
    ]:
        assert target in targets
        assert any(
            registered is hook and kind is HookType.AROUND
            for kind, registered, _ in HookRegistry._hooks[target]
        )


@pytest.mark.resident_mode
@pytest.mark.parametrize("hooked", [True, False], ids=["hooked", "upstream"])
def test_a_resident_component_survives_the_manager_move(monkeypatch, hooked):
    from sglang.multimodal_gen.runtime.managers.memory_managers import (
        component_residency_strategies as strategies,
    )

    # With resident regions the platform's local device is the host.
    monkeypatch.setattr(
        strategies, "get_local_torch_device", lambda: torch.device("cpu")
    )
    module = nn.Linear(4, 4).to("rbln:0")
    if hooked:
        residency.keep_resident_placement(strategies._module_to_local_device, module)
    else:
        strategies._module_to_local_device(module)
    assert (module.weight.device.type == "rbln") is hooked


SHARDED_ROWS = '{"on_device": {"allocs": [{"node_id": 0}, {"node_id": 1}], "axis": 0}}'
SHARDED_COLS = '{"on_device": {"allocs": [{"node_id": 0}, {"node_id": 1}], "axis": 1}}'


@pytest.mark.parametrize(
    ("destination_npus", "config", "read_as", "kept"),
    [
        (4, SHARDED_ROWS, None, True),
        (4, SHARDED_ROWS, SHARDED_ROWS, True),
        (4, SHARDED_ROWS, SHARDED_COLS, False),
        (2, SHARDED_ROWS, None, False),
        (4, None, None, False),
    ],
    ids=[
        "first-read",
        "same-layout",
        "other-shard-axis",
        "other-set-size",
        "no-config",
    ],
)
def test_a_cross_set_copy_keeps_only_a_layout_its_reader_reads(
    monkeypatch, destination_npus, config, read_as, kept
):
    """A layout crosses NPU sets only when the destination set is as large and the graph
    reading it there reads that layout; anything else comes over dense, with no tags,
    so no graph ever reads a buffer as a layout it does not have."""
    sizes = {0: 4, 1: destination_npus}
    monkeypatch.setattr(
        residency, "npus_of", lambda device: tuple(range(sizes[device.index or 0]))
    )
    tensor = torch.ones(2, 2)
    if config is not None:
        setattr(tensor, residency.PRODUCER_CONFIG, types.SimpleNamespace(json=config))
    kept_config = residency.layout_to_keep(tensor, torch.device("cpu", 1), read_as)
    assert (kept_config is not None) is kept


class _Stage:
    def __init__(self, work) -> None:
        self.work = work


@pytest.mark.resident_mode
def test_crossings_count_each_stage_download_by_op_and_upload(monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_RBLN_LOG_CROSSINGS", "1")
    resident = torch.zeros(4, 8, dtype=torch.bfloat16).to("rbln:0")
    host = torch.ones(16, dtype=torch.float32)

    def work():
        resident + 1
        resident * 2
        residency._Transfers().to(host, torch.device("rbln:0"))
        return residency.crossings.of_stage("_Stage")

    with caplog.at_level("INFO", logger=residency.__name__):
        seen = residency.stage_on_host_boundary(
            lambda stage: stage.work(), _Stage(work)
        )
    assert seen == {("down", "add"): (1, 64), ("up", "host tensor"): (1, 64)}
    assert residency.crossings.of_stage("_Stage") == {}
    assert "rbln crossings in _Stage: " in caplog.text


@pytest.mark.resident_mode
def test_a_stage_with_no_crossings_reports_none(monkeypatch, caplog):
    monkeypatch.setenv("SGLANG_RBLN_LOG_CROSSINGS", "1")
    with caplog.at_level("INFO", logger=residency.__name__):
        residency.stage_on_host_boundary(lambda stage: torch.ones(2) * 2, _Stage(None))
    assert "rbln crossings in _Stage: none" in caplog.text


@pytest.mark.resident_mode
def test_casting_a_resident_module_downloads_no_parameter():
    # Upstream casts the VAE around each decode: it brought the whole decoder down
    # (776.6 MiB) to compare tensor types.
    module = nn.Linear(8, 8).to("rbln:0")

    def cast():
        module.to(torch.float32)
        return residency.crossings.of_stage("_Stage")

    seen = residency.stage_on_host_boundary(lambda stage: stage.work(), _Stage(cast))
    assert seen == {}
    assert module.weight.device.type == "rbln"

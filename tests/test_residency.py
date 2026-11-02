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


@pytest.mark.parametrize(
    ("mode", "num_devices", "expected"),
    [
        ("device", "4", True),
        ("device", "1", False),
        ("device", None, False),
        ("host", "4", False),
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


@pytest.mark.device_mode
def test_boundary_brings_device_values_to_the_host():
    hidden = torch.arange(6.0, dtype=torch.bfloat16)
    resident = hidden.to("rbln:0")
    with residency.HostBoundary():
        shape = resident.shape
        summed = resident + hidden
    assert shape == hidden.shape
    assert summed.device.type == "cpu"
    torch.testing.assert_close(summed, hidden * 2)


@pytest.mark.device_mode
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


@pytest.mark.device_mode
def test_a_value_crosses_npu_sets_on_the_device():
    if torch.rbln.device_count() < 2:
        pytest.skip("needs two logical devices")
    value = torch.arange(4096, dtype=torch.bfloat16).reshape(64, 64).to("rbln:0")
    copied = residency._Transfers().to(value.t(), torch.device("rbln", 1))
    assert copied.device == torch.device("rbln", 1)
    assert torch.equal(copied.cpu(), value.t().cpu())

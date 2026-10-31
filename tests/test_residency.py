# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

from sglang_rbln import device
from sglang_rbln.compilation import residency
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
    resident = hidden.to(residency.DEVICE)
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

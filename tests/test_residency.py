# SPDX-License-Identifier: Apache-2.0
import os
import sys
import types

import pytest
import torch
from torch import nn

from sglang_rbln.compilation import placement, residency
from sglang_rbln.compilation.backend import COMPILE_BACKEND_NAME

# These need a device to name, not the hardware. A dummy device has no real
# outputs, so only placement and structure are asserted.
pytestmark = pytest.mark.usefixtures("dummy_device")


@pytest.fixture
def transfers(monkeypatch):
    """(from, to) device types of every tensor the boundary moved, in order."""
    seen = []
    move = residency._transfers.to

    def recording(value, device, *args, **kwargs):
        if isinstance(value, torch.Tensor) and value.device != device:
            seen.append((value.device.type, device.type))
        return move(value, device, *args, **kwargs)

    monkeypatch.setattr(residency._transfers, "to", recording)
    return seen


def _compiled_block() -> nn.Module:
    block = nn.Linear(4, 4)
    block.compile(backend="eager")
    return block


def test_install_wraps_a_compiled_call_once():
    block = _compiled_block()
    residency.install_resident_call(block)
    wrapped = block._compiled_call_impl
    residency.install_resident_call(block)
    assert block._compiled_call_impl is wrapped
    assert wrapped._rbln_resident


def test_install_skips_an_uncompiled_module():
    block = nn.Linear(4, 4)
    residency.install_resident_call(block)
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
    with residency.DeviceBoundary():
        doubled = hidden * 2
    assert doubled.device.type == "cpu"
    torch.testing.assert_close(doubled, torch.arange(0.0, 12.0, 2.0))


def test_boundary_takes_host_operands_to_the_resident_one(transfers):
    hidden = torch.arange(6.0, dtype=torch.bfloat16)
    resident = hidden.to("rbln:0")
    with residency.DeviceBoundary():
        shape = resident.shape
        summed = resident + hidden
        scaled = resident * torch.tensor(2.0)
    assert shape == hidden.shape
    assert summed.device.type == scaled.device.type == "rbln"
    assert transfers == [("cpu", "rbln")]  # the 0-dim scalar stays on the host


def test_boundary_writes_into_a_host_buffer_on_the_host():
    buffer = torch.zeros(4, dtype=torch.bfloat16)
    resident = torch.ones(2, dtype=torch.bfloat16).to("rbln:0")
    with residency.DeviceBoundary():
        buffer[1:3] = resident
    assert buffer.device.type == "cpu"
    assert residency._target_device(
        torch.Tensor.__setitem__, (buffer, slice(1, 3), resident), {}
    ) == torch.device("cpu")


def test_a_fake_tensor_probe_leaves_the_upload_cache_alone(transfers):
    # The audio chain finds its waveform length under FakeTensorMode before its
    # first run, while its buffers are still on the host. A probe that uploaded
    # them would cache fake tensors, and the next real op on a buffer would
    # return a fake.
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    buffer = torch.ones(8, dtype=torch.bfloat16)
    resident = torch.randn(8, dtype=torch.bfloat16).to("rbln:0")
    with residency.DeviceBoundary():
        with FakeTensorMode(allow_non_fake_inputs=True) as fake:
            fake.from_tensor(resident) * buffer
        assert transfers == []
        assert not isinstance(resident * buffer, FakeTensor)


def test_a_resident_call_moves_module_and_inputs_under_the_boundary():
    block = nn.Linear(4, 4)
    seen = []
    wrapped = residency.resident_call(
        block,
        lambda hidden: seen.append((hidden.device.type, block.weight.device.type)),
    )
    with residency.DeviceBoundary():
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
def npu_layout(monkeypatch):
    def set_layout(env: dict, visible: int) -> None:
        monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
        monkeypatch.delenv("SGLANG_RBLN_COMPONENT_DEVICES", raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("RBLN_VISIBLE_DEVICES", ",".join(map(str, range(visible))))

    return set_layout


@pytest.mark.parametrize("case", GROUP_CASES)
def test_device_groups_put_each_component_set_on_its_own_device(npu_layout, case):
    env, visible, groups = GROUP_CASES[case]
    npu_layout(env, visible)
    assert residency.device_groups() == groups


def test_component_sets_may_not_share_npus(npu_layout):
    npu_layout({"SGLANG_RBLN_COMPONENT_DEVICES": "text_encoder=3,4"}, 8)
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
    npu_layout, monkeypatch, given, device_map
):
    npu_layout({}, 8)
    for name in ("RBLN_DEVICE_MAP", "RBLN_NPUS_PER_DEVICE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in given.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setitem(sys.modules, "torch_rbln", types.ModuleType("torch_rbln"))
    residency.configure_device_group()
    assert os.environ.get("RBLN_DEVICE_MAP") == device_map


def test_a_placed_module_runs_on_the_device_of_its_npus(monkeypatch):
    physical_ids = types.ModuleType("torch_rbln._internal.rsd_utils")
    physical_ids.get_physical_device_ids = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}.get
    monkeypatch.setitem(sys.modules, "torch_rbln._internal.rsd_utils", physical_ids)
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


@pytest.mark.npu  # it asserts the values an eager copy between NPU sets carries
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


ROW_FORM = '{"on_device": {"allocs": [{"node_id": 0}, {"node_id": 1}], "axis": 0}}'
COLUMN_FORM = '{"on_device": {"allocs": [{"node_id": 0}, {"node_id": 1}], "axis": 1}}'


@pytest.mark.parametrize(
    ("destination_npus", "config", "read_as", "kept"),
    [
        (4, ROW_FORM, None, True),
        (4, ROW_FORM, ROW_FORM, True),
        (4, ROW_FORM, COLUMN_FORM, False),
        (2, ROW_FORM, None, False),
        (4, None, None, False),
    ],
    ids=[
        "first-read",
        "same-form",
        "other-form",
        "other-set-size",
        "no-config",
    ],
)
def test_a_cross_set_copy_keeps_its_form_only_for_a_reader_that_accepts_it(
    monkeypatch, destination_npus, config, read_as, kept
):
    """A tensor copied to another NPU set keeps its multi-NPU form only when that set
    is as large and the graph reading it accepts that form; otherwise it arrives as a
    plain tensor."""
    sizes = {0: 4, 1: destination_npus}
    monkeypatch.setattr(
        residency, "npus_of", lambda device: tuple(range(sizes[device.index or 0]))
    )
    tensor = torch.ones(2, 2)
    if config is not None:
        setattr(tensor, residency.PRODUCER_CONFIG, types.SimpleNamespace(json=config))
    kept_config = residency.layout_to_keep(tensor, torch.device("cpu", 1), read_as)
    assert (kept_config is not None) is kept


def test_a_strided_host_tensor_uploads_dense():
    # LTX-2 packs its latents as a transposed view. Uploading it dense keeps the
    # first request's graphs identical to those a compile-only run prepared.
    host = (
        torch.arange(128 * 60, dtype=torch.float32).reshape(1, 128, 60).transpose(1, 2)
    )
    uploaded = residency._Transfers().to(host, torch.device("rbln:0"))
    assert uploaded.device.type == "rbln"
    assert uploaded.is_contiguous()


def test_casting_a_resident_module_moves_no_parameter(transfers):
    # Upstream casts the VAE around each decode; that must not bring the whole
    # decoder (776.6 MiB) to the host.
    module = nn.Linear(8, 8).to("rbln:0")
    residency.stage_on_device_boundary(lambda stage: module.to(torch.float32), None)
    assert transfers == []
    assert module.weight.device.type == "rbln"

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from sglang.multimodal_gen.runtime.utils import precision as upstream
from torch._subclasses.fake_tensor import FakeTensorMode

pytest.importorskip("torch_rbln")

from sglang_rbln.patches.utils import precision  # noqa: E402


def autocast_context(dtype, disable_autocast, *, enabled=None):
    return precision.resident_autocast(
        upstream.autocast_context, dtype, disable_autocast, enabled=enabled
    )


@pytest.fixture(autouse=True)
def policy(dummy_device):
    precision.rbln_autocast.register()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
def test_opens_upstream_autocast_on_rbln_too(dtype):
    with autocast_context(dtype, disable_autocast=False):
        assert torch.is_autocast_enabled(upstream.current_platform.device_type)
        assert torch.is_autocast_enabled("rbln")
        assert torch.get_autocast_dtype("rbln") == dtype
    assert not torch.is_autocast_enabled("rbln")


@pytest.mark.parametrize(
    ("dtype", "disable_autocast", "enabled"),
    [
        (torch.bfloat16, True, None),
        (torch.float32, False, None),
        (torch.bfloat16, False, False),
    ],
    ids=["disabled", "float32", "explicit-off"],
)
def test_stays_off_where_upstream_turns_it_off(dtype, disable_autocast, enabled):
    with autocast_context(dtype, disable_autocast, enabled=enabled):
        assert not torch.is_autocast_enabled("rbln")


@pytest.mark.parametrize("channels", [8, 128])
def test_a_bf16_conv_takes_float32_resident_inputs(channels):
    with FakeTensorMode():
        x = torch.empty(1, channels, 2, 4, 4, device="rbln:0")
        weight = torch.empty(
            channels, channels, 3, 3, 3, dtype=torch.bfloat16, device="rbln:0"
        )
        with autocast_context(torch.bfloat16, disable_autocast=False):
            assert F.conv3d(x, weight, padding=1).dtype == torch.bfloat16


@pytest.fixture
def mode(monkeypatch, request):
    """The platform as each execution mode resolves it: its device type, and whether
    regions keep their tensors on the NPUs."""
    from sglang_rbln import device

    resident_regions, on_device = request.param
    monkeypatch.setattr(device, "resident_regions", lambda: resident_regions)
    monkeypatch.setattr(
        type(upstream.current_platform),
        "device_type",
        property(lambda self: "rbln" if on_device else "cpu"),
    )
    return request.param


@pytest.mark.parametrize(
    "mode",
    [(False, True)],
    ids=["device"],
    indirect=True,
)
def test_device_mode_keeps_upstreams_one_rbln_autocast(mode):
    host = upstream.autocast_context(torch.bfloat16, False)
    hooked = precision.resident_autocast(
        upstream.autocast_context, torch.bfloat16, False
    )
    assert isinstance(hooked, torch.autocast) and hooked.device == "rbln"
    assert type(hooked) is type(host)


@pytest.mark.parametrize(
    "mode",
    [(True, False)],
    ids=["resident-regions"],
    indirect=True,
)
def test_resident_regions_open_rbln_autocast_beside_the_hosts(mode):
    with precision.resident_autocast(upstream.autocast_context, torch.bfloat16, False):
        assert torch.is_autocast_enabled("cpu")
        assert torch.is_autocast_enabled("rbln")


@pytest.mark.parametrize(
    "mode",
    [(False, False)],
    ids=["staged"],
    indirect=True,
)
def test_staged_mode_leaves_upstreams_autocast_alone(mode):
    hooked = precision.resident_autocast(
        upstream.autocast_context, torch.bfloat16, False
    )
    assert isinstance(hooked, torch.autocast) and hooked.device == "cpu"
    with hooked:
        assert not torch.is_autocast_enabled("rbln")


def test_tp_size_parsed_after_import_opens_both(monkeypatch):
    """`--tp-size` turns resident regions on after the plugin registered: the hook
    and the platform's device type read the mode at call time, so both follow."""
    from types import SimpleNamespace

    from sglang_rbln import platform as rbln_platform

    monkeypatch.setenv("SGLANG_RBLN_EXECUTION_MODE", "resident")
    monkeypatch.setattr(
        rbln_platform, "EXECUTION_MODE", rbln_platform.device.ExecutionMode.RESIDENT
    )
    monkeypatch.setenv(rbln_platform.NUM_DEVICES_ENV, "")
    args = SimpleNamespace(
        tp_size=4, num_gpus=1, is_arg_explicitly_set=lambda flag: flag == "tp_size"
    )
    assert upstream.current_platform.device_type == "rbln"  # before the arguments
    rbln_platform.take_tensor_parallel_degree(args)
    assert upstream.current_platform.device_type == "cpu"
    with precision.resident_autocast(upstream.autocast_context, torch.bfloat16, False):
        assert torch.is_autocast_enabled("cpu")
        assert torch.is_autocast_enabled("rbln")

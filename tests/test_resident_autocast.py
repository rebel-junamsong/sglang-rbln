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

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

torch_rbln = pytest.importorskip("torch_rbln")

from sglang_rbln.patches.torch_rbln import autocast  # noqa: E402

DEVICE = "rbln:0"


@pytest.fixture(name="fake")
def fake_fixture(dummy_device):
    autocast.register()
    with FakeTensorMode():
        yield lambda *shape, dtype=torch.float32: torch.empty(
            *shape, dtype=dtype, device=DEVICE
        )


def test_declares_the_dtypes_autocast_takes(fake):
    with torch.autocast("rbln", torch.bfloat16):
        assert torch.is_autocast_enabled("rbln")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
@pytest.mark.parametrize("channels", [8, 64])
def test_convs_and_linears_run_in_the_autocast_dtype(fake, dtype, channels):
    x = fake(1, channels, 4, 4, dtype=torch.float32)
    weight = fake(channels, channels, 3, 3, dtype=torch.bfloat16)
    bias = fake(channels, dtype=torch.bfloat16)
    with torch.autocast("rbln", dtype):
        assert F.conv2d(x, weight, bias, padding=1).dtype == dtype
        tokens = x.flatten(2).transpose(1, 2)
        assert F.linear(tokens, weight[:, :, 0, 0], bias).dtype == dtype


@pytest.mark.parametrize("features", [16, 128])
def test_norms_and_softmax_run_in_float32(fake, features):
    x = fake(2, 8, features, dtype=torch.bfloat16)
    weight = fake(features, dtype=torch.bfloat16)
    with torch.autocast("rbln", torch.bfloat16):
        assert F.layer_norm(x, (features,), weight, weight).dtype == torch.float32
        assert F.group_norm(x.transpose(1, 2), 4).dtype == torch.float32
        assert torch.softmax(x, -1).dtype == torch.float32
        assert torch.softmax(x, -1, dtype=torch.bfloat16).dtype == torch.bfloat16


def test_promote_ops_take_the_widest_input(fake):
    low, high = fake(4, dtype=torch.bfloat16), fake(4)
    with torch.autocast("rbln", torch.bfloat16):
        assert torch.addcmul(low, high, low).dtype == torch.float32


def test_other_ops_fall_through(fake):
    x = fake(4, 4)
    with torch.autocast("rbln", torch.bfloat16):
        assert (x.to(torch.float16) + 1).dtype == torch.float16
        assert torch.relu(x).dtype == torch.float32


def test_leaves_host_tensors_to_cpu_autocast(fake):
    x = torch.empty(1, 8, 4, 4)
    weight = torch.empty(8, 8, 3, 3, dtype=torch.bfloat16)
    with torch.autocast("rbln", torch.bfloat16), pytest.raises(RuntimeError):
        F.conv2d(x, weight)

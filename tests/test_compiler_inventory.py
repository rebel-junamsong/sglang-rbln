# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
import tvm
from rebel.core.custom_converter._pt_sdpa import extend_scaled_dot_product_attention
from tvm import relay

COMPILER_FIX = "rebellions-sw/rebel_compiler#14989"


def _infer_shape(expr):
    typed = relay.transform.InferType()(tvm.IRModule.from_expr(expr))
    return tuple(int(dim) for dim in typed["main"].body.checked_type.shape)


def _lower_grouped_query_sdpa():
    query = relay.var("query", shape=(1, 8, 16, 32), dtype="float32")
    key = relay.var("key", shape=(1, 2, 16, 32), dtype="float32")
    value = relay.var("value", shape=(1, 2, 16, 32), dtype="float32")
    inputs = [query, key, value, None, 0.0, True, None, True]
    input_types = ["float32"] * 3 + ["", "float", "bool", "", "bool"]
    output = extend_scaled_dot_product_attention(_infer_shape, inputs, input_types)
    return _infer_shape(output)


def test_installed_compiler_still_needs_the_head_expansion():
    try:
        _lower_grouped_query_sdpa()
    except tvm.error.InternalError as error:
        assert "BatchDot" in str(error)
        return
    pytest.fail(
        f"rebel-compiler now honours enable_gqa ({COMPILER_FIX}); "
        "delete attention.py and the platform's get_attn_backend_cls_str"
    )


class _RotaryAngles(torch.nn.Module):
    def __init__(self):
        super().__init__()
        inv_freq = 1.0 / (256 ** (torch.arange(0, 16, 2).float() / 16))
        self.register_buffer("inv_freq", inv_freq.to(torch.bfloat16))

    def forward(self, x):
        positions = torch.arange(x.shape[1], dtype=torch.float32)
        angles = torch.einsum("t,f->tf", positions, self.inv_freq.float())
        return x * angles.to(x.dtype)


@pytest.mark.npu
def test_installed_compiler_still_misreads_upcast_bfloat16_buffers():
    import rebel

    module = _RotaryAngles().eval()
    x = torch.ones(1, 6, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = module(x)
        compiled = torch.compile(
            module,
            backend="rbln",
            dynamic=False,
            options={"compile_context": rebel.CompileContext(use_weight_sharing=True)},
        )
        actual = compiled(x)
    assert not torch.allclose(actual.float(), expected.float(), atol=0.05), (
        "rebel-compiler now reads a weight-free bfloat16 buffer upcast to float32 "
        "correctly; delete float32_rotary_tables from patches/models/flux3_video_vae"
    )

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
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

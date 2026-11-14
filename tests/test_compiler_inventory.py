# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch
import tvm
from rebel.core.custom_converter._pt_sdpa import extend_scaled_dot_product_attention
from tvm import relay

COMPILER_FIX = "rebellions-sw/rebel_compiler#14989"


def _infer_shape(expr):
    typed = relay.transform.InferType()(tvm.IRModule.from_expr(expr))
    return tuple(int(dim) for dim in typed["main"].body.checked_type.shape)


def _lower_grouped_query_sdpa(query, key, value):
    tensors = {
        name: relay.var(name, shape=tensor.shape, dtype="float32")
        for name, tensor in (("query", query), ("key", key), ("value", value))
    }
    inputs = [*tensors.values(), None, 0.0, True, None, True]
    input_types = ["float32"] * 3 + ["", "float", "bool", "", "bool"]
    output = extend_scaled_dot_product_attention(_infer_shape, inputs, input_types)
    module = tvm.IRModule.from_expr(relay.Function(list(tensors.values()), output))
    evaluate = relay.create_executor(
        "debug", mod=module, device=tvm.cpu(), target="llvm"
    ).evaluate()
    return evaluate(query.numpy(), key.numpy(), value.numpy()).numpy()


@pytest.mark.parametrize("kv_heads", [1, 2, 4], ids=lambda n: f"kv{n}")
def test_installed_compiler_honours_enable_gqa(kv_heads):
    """The platform used to expand the kv heads itself; `COMPILER_FIX` removed
    that need. If this fails the compiler has lost `enable_gqa` again and the
    expanding SDPA backend has to come back -- see the commit that deleted it.
    """
    torch.manual_seed(0)
    query = torch.randn(2, 8, 16, 32)
    key = torch.randn(2, kv_heads, 16, 32)
    value = torch.randn(2, kv_heads, 16, 32)

    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, is_causal=True, enable_gqa=True
    )
    lowered = _lower_grouped_query_sdpa(query, key, value)

    assert lowered.shape == tuple(expected.shape), COMPILER_FIX
    np.testing.assert_allclose(lowered, expected.numpy(), rtol=1e-4, atol=1e-5)


EAGER_OP_ON_SECOND_SET = """
import torch, torch_rbln
weight = torch.ones(16, 8, dtype=torch.bfloat16).to("rbln:1")
print((weight * torch.tensor(2.0, dtype=torch.bfloat16)).device)
"""


@pytest.mark.xfail(
    strict=True,
    reason=(
        f"{COMPILER_FIX}: rebel.core.torch_compile._create_runtime_with_weights "
        "hands an eager op's logical device index to create_runtime, which expands "
        "only device 0 to all of a multi-NPU device's NPUs"
    ),
)
def test_an_eager_op_runs_on_a_second_multi_npu_logical_device():
    """In resident TP mode the text encoder, connectors and audio graph sit on the
    spare NPU set, logical device 1, and the eager ops between their regions run
    there: Gemma's embedding scale failed every request with "The number of
    devices provided (1) does not match the number of devices in the compiled
    model (4)". Logical device 0 (the DiT's set) is unaffected."""
    import subprocess
    import sys

    env = {
        **__import__("os").environ,
        "RBLN_DUMMY_DEVICE": "1",
        "RBLN_DEVICE_MAP": "[0,1,2,3],[4,5,6,7]",
        "TORCH_RBLN_DISABLE_FALLBACK": "all",
    }
    env.setdefault("RBLN_FORCE_NPU_NAME", "RBLN-CA22")
    run = subprocess.run(
        [sys.executable, "-c", EAGER_OP_ON_SECOND_SET],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr[-1500:]
    assert run.stdout.strip().splitlines()[-1] == "rbln:1"

# SPDX-License-Identifier: Apache-2.0
import logging

import pytest
import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from torch import nn

from sglang_rbln.compilation import backend as compilation
from sglang_rbln.layers.attention.backends import sdpa as attention

QUERY_HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128


def _qkv(query_len=64, kv_len=96, dtype=torch.bfloat16):
    torch.manual_seed(0)
    query = torch.randn(1, query_len, QUERY_HEADS, HEAD_DIM, dtype=dtype)
    key = torch.randn(1, kv_len, KV_HEADS, HEAD_DIM, dtype=dtype)
    value = torch.randn(1, kv_len, KV_HEADS, HEAD_DIM, dtype=dtype)
    return query, key, value


def _impl(impl_cls):
    return impl_cls(
        num_heads=QUERY_HEADS,
        head_size=HEAD_DIM,
        causal=False,
        softmax_scale=HEAD_DIM**-0.5,
        num_kv_heads=KV_HEADS,
    )


def test_expanded_heads_match_grouped_query_attention():
    query, key, value = _qkv()
    reference = _impl(SDPAImpl).forward(query, key, value, None)
    expanded = _impl(attention.RBLNSdpaImpl).forward(query, key, value, None)
    assert torch.equal(expanded, reference)


def test_expand_is_a_no_op_for_equal_heads():
    query, key, value = _qkv()
    same_key = key.repeat_interleave(QUERY_HEADS // KV_HEADS, dim=2)
    out_key, _ = attention.expand_kv_heads(query, same_key, same_key)
    assert out_key is same_key


def test_rejects_non_multiple_heads():
    query, key, value = _qkv()
    with pytest.raises(ValueError, match="multiple"):
        attention.expand_kv_heads(query, key[:, :, :3], value[:, :, :3])


def test_backend_exposes_the_expanding_impl():
    assert attention.RBLNSdpaBackend.get_impl_cls() is attention.RBLNSdpaImpl


class _Attention(nn.Module):
    def __init__(self, impl):
        super().__init__()
        self.impl = impl

    def forward(self, query, key, value):
        return self.impl.forward(query, key, value, None)


@pytest.mark.npu
def test_grouped_query_attention_compiles_on_the_npu(caplog, monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compilation.ensure_compile_backend()
    query, key, value = _qkv()
    reference = _impl(SDPAImpl).forward(query, key, value, None)
    module = _Attention(_impl(attention.RBLNSdpaImpl))
    compiled = torch.compile(
        module,
        backend=compilation.COMPILE_BACKEND_NAME,
        options=compilation.compile_options(module),
        fullgraph=True,
        dynamic=False,
    )
    with caplog.at_level(logging.WARNING, logger="rebel-compiler"):
        output = compiled(query, key, value)
    assert "Fallback to eager" not in caplog.text
    assert output.device.type == "cpu"
    assert torch.allclose(output.float(), reference.float(), atol=2e-2, rtol=2e-2)

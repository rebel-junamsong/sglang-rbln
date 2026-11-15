# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch._dynamo import list_backends
from torch._dynamo.testing import CompileCounter

from sglang_rbln import envs
from sglang_rbln.compilation import backend as compilation
from sglang_rbln.compilation import placement


class _TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(64, 256)
        self.down = nn.Linear(256, 64)

    def forward(self, hidden):
        return self.down(torch.nn.functional.gelu(self.up(hidden)))


def test_the_rbln_backend_is_registered_with_static_shapes():
    compilation.ensure_compile_backend()
    assert "rbln" in list_backends()
    assert torch._dynamo.config.automatic_dynamic_shapes is False


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16)

    def forward(self, hidden):
        return hidden + self.proj(hidden)


LTX2_DIT_BLOCKS = 48


def test_every_block_compiles_for_every_request_shape():
    compilation.ensure_compile_backend()
    torch._dynamo.reset()
    counter = CompileCounter()
    blocks = nn.ModuleList(_Block() for _ in range(LTX2_DIT_BLOCKS))
    for block in blocks:
        block.compile(backend=counter, dynamic=False)
    shapes = range(8, 40, 4)
    with torch.no_grad():
        for tokens in shapes:
            hidden = torch.randn(1, tokens, 16)
            for block in blocks:
                hidden = block(hidden)
    assert counter.frame_count == len(blocks) * len(shapes)


def test_describe_graph_lists_ops_and_inputs():
    graph = torch.fx.symbolic_trace(_TinyMlp())
    described = compilation.describe_graph(
        graph, [torch.zeros(2, 0, 8, dtype=torch.bfloat16)]
    )
    assert described == "3 ops, inputs [2x0x8:bfloat16]"


def test_contiguous_device_inputs_leave_host_tensors_alone():
    cpu_view = torch.zeros(4, 6)[:, ::2]
    plain = torch.zeros(3)
    passed = compilation.contiguous_device_inputs([cpu_view, plain, 7])
    assert passed[0] is cpu_view and passed[1] is plain and passed[2] == 7


def test_wraps_the_registered_backend_once(monkeypatch):
    from torch._dynamo.backends import registry

    original = getattr(
        registry._COMPILER_FNS["rbln"],
        "__wrapped_backend__",
        registry._COMPILER_FNS["rbln"],
    )
    monkeypatch.setitem(registry._COMPILER_FNS, "rbln", original)
    compilation.ensure_compile_backend()
    wrapped = registry._COMPILER_FNS["rbln"]
    assert wrapped.__wrapped_backend__ is original
    compilation.ensure_compile_backend()
    assert registry._COMPILER_FNS["rbln"] is wrapped


@pytest.mark.parametrize("flag", ["RBLN_POOL_IO_BUFFERS", "RBLN_RSD_GROUPING"])
@pytest.mark.parametrize(
    ("preset", "expected"), [(None, "1"), ("0", "0")], ids=["unset", "set-to-0"]
)
def test_compiler_flags_default_on_unless_told_otherwise(
    monkeypatch, flag, preset, expected
):
    if preset is None:
        monkeypatch.delenv(flag, raising=False)
    else:
        monkeypatch.setenv(flag, preset)
    compilation.ensure_compile_backend()
    assert os.environ[flag] == expected


class TestCompileOptions:
    @pytest.fixture(autouse=True)
    def no_plugin_env(self, monkeypatch):
        for name in envs.environment_variables:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("SGLANG_RBLN_DISABLE_COMPILE_CACHE", "1")

    @staticmethod
    def compile_switches(module=None) -> dict:
        options = compilation.compile_options(module or _TinyMlp())
        del options["compile_context"]
        return options

    def test_modules_on_one_device_set_share_a_context(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SGLANG_RBLN_DISABLE_COMPILE_CACHE")
        monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
        monkeypatch.setattr(compilation, "_device_set_contexts", {})
        first = compilation.compile_options(_TinyMlp())
        second = compilation.compile_options(_TinyMlp())
        assert first["compile_context"] is second["compile_context"]
        assert first["cache_dir"] == str(tmp_path)
        assert first["mode"] == ["strict"]

    def test_device_sets_keep_their_own_context(self, monkeypatch):
        monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
        monkeypatch.setattr(compilation, "_device_set_contexts", {})
        default = compilation.compile_options(_TinyMlp())["compile_context"]
        placed = _TinyMlp()
        placement.place(placed, (4, 5, 6, 7))
        assert compilation.compile_options(placed)["compile_context"] is not default

    def test_context_shares_weights_across_graphs(self):
        context = compilation.compile_options(_TinyMlp())["compile_context"]
        assert context.state.use_weight_sharing

    @pytest.mark.parametrize(
        ("env", "expected"),
        [
            ({}, {"mode": ["strict"]}),
            ({"SGLANG_RBLN_NUM_DEVICES": "4"}, {"mode": ["strict"], "num_devices": 4}),
            ({"SGLANG_RBLN_COMPILE_ONLY": "1"}, {"mode": ["compile_only", "strict"]}),
        ],
        ids=["always-strict", "num-devices", "compile-only"],
    )
    def test_compile_switches_follow_the_environment(self, monkeypatch, env, expected):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        assert self.compile_switches() == expected


@pytest.mark.npu
def test_compiled_module_runs_on_the_npu(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("SGLANG_RBLN_COMPILE_ONLY", raising=False)
    monkeypatch.delenv("SGLANG_RBLN_DISABLE_COMPILE_CACHE", raising=False)
    compilation.ensure_compile_backend()
    module = _TinyMlp().eval().to(torch.bfloat16)
    hidden = torch.randn(4, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        eager = module(hidden)
        compiled = torch.compile(
            module,
            backend=compilation.COMPILE_BACKEND_NAME,
            options=compilation.compile_options(module),
            fullgraph=True,
            dynamic=False,
        )
        first = compiled(hidden)
        second = compiled(hidden)
    assert first.device.type == "cpu"
    assert torch.equal(first, second)
    assert torch.allclose(first.float(), eager.float(), atol=2e-2, rtol=2e-2)
    assert list(tmp_path.glob("*.rbln")), "the compiled graph was not persisted"


def test_compile_only_stands_an_empty_tensor_in_for_a_device_view(
    monkeypatch, dummy_device
):
    # contiguous() is an eager copy a compile-only run has no use for: it reads only
    # the shape.
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_ONLY", "1")
    view = torch.zeros(4, 6).to("rbln:0")[:, ::2]

    def copies(*args, **kwargs):
        raise AssertionError("a compile-only run copies nothing")

    monkeypatch.setattr(torch.Tensor, "contiguous", copies)
    (stand_in,) = compilation.contiguous_device_inputs([view])
    assert stand_in.is_contiguous()
    assert (stand_in.shape, stand_in.dtype, stand_in.device) == (
        view.shape,
        view.dtype,
        view.device,
    )


class _GroupedQueryAttention(nn.Module):
    def forward(self, query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, enable_gqa=True
        )


@pytest.mark.npu
@pytest.mark.parametrize("kv_heads", [1, 2, 4], ids=lambda n: f"kv{n}")
def test_grouped_query_attention_matches_eager(monkeypatch, tmp_path, kv_heads):
    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("SGLANG_RBLN_COMPILE_ONLY", raising=False)
    compilation.ensure_compile_backend()
    torch.manual_seed(0)
    module = _GroupedQueryAttention()
    query = torch.randn(2, 8, 16, 32)
    key = torch.randn(2, kv_heads, 16, 32)
    value = torch.randn(2, kv_heads, 16, 32)
    with torch.no_grad():
        expected = module(query, key, value)
        torch._dynamo.reset()
        compiled = torch.compile(
            module,
            backend=compilation.COMPILE_BACKEND_NAME,
            options=compilation.compile_options(module),
            fullgraph=True,
            dynamic=False,
        )
        actual = compiled(query, key, value)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


EAGER_OP_ON_SECOND_SET = """
import torch, torch_rbln
weight = torch.ones(16, 8, dtype=torch.bfloat16).to("rbln:1")
print((weight * torch.tensor(2.0, dtype=torch.bfloat16)).device)
"""


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the locked rebel-compiler does not run eager ops on a second multi-NPU set yet"
    ),
)
def test_eager_ops_run_on_the_text_encoder_npu_set():
    """With the text encoder on its own NPU set, the eager ops between its graphs run
    on that set."""
    env = {
        **os.environ,
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

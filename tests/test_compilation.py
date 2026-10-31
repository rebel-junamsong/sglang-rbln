# SPDX-License-Identifier: Apache-2.0
import os
import threading

import pytest
import torch
from torch import nn
from torch._dynamo import list_backends
from torch._dynamo.testing import CompileCounter

from sglang_rbln import envs
from sglang_rbln.compilation import backend as compilation


class _TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(64, 256)
        self.down = nn.Linear(256, 64)

    def forward(self, hidden):
        return self.down(torch.nn.functional.gelu(self.up(hidden)))


def test_rebel_backend_is_available():
    compilation.ensure_compile_backend()
    assert "rbln" in list_backends()
    assert torch._dynamo.config.automatic_dynamic_shapes is False


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16)

    def forward(self, hidden):
        return hidden + self.proj(hidden)


def test_every_block_compiles_for_every_request_shape():
    compilation.ensure_compile_backend()
    torch._dynamo.reset()
    counter = CompileCounter()
    blocks = nn.ModuleList(_Block() for _ in range(48))
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


def test_runtime_report_totals_split_device_and_host():
    totals = compilation.RuntimeReportTotals()
    totals.add(
        [
            {
                "type": "timer",
                "seq": 1,
                "host_pad": 70,
                "main_0": 292,
                "total_device": 292,
                "total_host": 70,
                "total_ccl": 0,
            },
            {"type": "prep", "seq": 1, "prepare_input_us": 134, "prepare_output_us": 6},
            {
                "type": "timer",
                "seq": 2,
                "host_pad": 13,
                "main_0": 133,
                "total_device": 133,
                "total_host": 13,
                "total_ccl": 0,
            },
        ]
    )
    assert (totals.calls, totals.device_us, totals.host_us, totals.prepare_us) == (
        2,
        425,
        83,
        140,
    )
    assert totals.node_us == {"host_pad": 83, "main_0": 425}
    assert totals.top_nodes(1) == [("main_0", 425)]
    assert "device share 65.6%" in totals.summary()


def test_dump_graph_round_trips(tmp_path):
    graph = torch.fx.symbolic_trace(_TinyMlp())
    stem = compilation.dump_graph(graph, [torch.zeros(2, 64)], tmp_path)
    assert "gelu" in stem.with_suffix(".txt").read_text()
    inputs = torch.load(stem.with_suffix(".inputs.pt"))
    module = torch.load(stem.with_suffix(".module.pt"), weights_only=False)
    assert torch.equal(module(*inputs), graph(*inputs))
    assert torch.load(stem.with_suffix(".state.pt")).keys() == graph.state_dict().keys()


def test_dump_graph_drops_process_groups(tmp_path, monkeypatch):
    lock_type = type(threading.Lock())
    monkeypatch.setattr(compilation, "_dropped_on_dump", lambda: (lock_type,))
    graph = torch.fx.symbolic_trace(_TinyMlp())
    graph.process_group = threading.Lock()
    stem = compilation.dump_graph(graph, [torch.zeros(2, 64)], tmp_path)
    module = torch.load(stem.with_suffix(".module.pt"), weights_only=False)
    assert module.process_group is None
    hidden = torch.zeros(2, 64)
    assert torch.equal(module(hidden), graph(hidden))


def test_dump_graph_keeps_weights_of_unpicklable_graphs(tmp_path, caplog):
    graph = torch.fx.symbolic_trace(_TinyMlp())
    graph.process_group = threading.Lock()
    stem = compilation.dump_graph(graph, [torch.zeros(2, 64)], tmp_path)
    assert not stem.with_suffix(".module.pt").exists()
    assert torch.load(stem.with_suffix(".state.pt")).keys() == graph.state_dict().keys()
    assert "kept without its module" in caplog.text


class TestGraphCheck:
    @pytest.mark.parametrize(
        ("offset", "expected"),
        [(0.0, (0.0, 0.0)), (0.5, (0.5, 0.25))],
        ids=["exact", "shifted"],
    )
    def test_output_deviation(self, offset, expected):
        reference = (torch.tensor([2.0, -1.0]), 3)
        actual = (reference[0] + offset, 3)
        assert compilation.output_deviation(actual, reference) == expected

    def test_checks_only_the_first_run(self, caplog):
        graph = torch.fx.symbolic_trace(_TinyMlp())
        calls = []

        def compiled(x):
            calls.append(x)
            return graph(x) + 1.0

        run = compilation._checking_callable(compiled, graph, "tiny")
        x = torch.ones(2, 64)
        with caplog.at_level("INFO", logger=compilation.__name__):
            run(x)
            run(x)
        checks = [r for r in caplog.records if "graph check tiny" in r.message]
        assert len(calls) == 2 and len(checks) == 1
        assert "max abs 1" in checks[0].message


def test_contiguous_device_inputs_leave_host_tensors_alone():
    cpu_view = torch.zeros(4, 6)[:, ::2]
    plain = torch.zeros(3)
    passed = compilation.contiguous_device_inputs([cpu_view, plain, 7])
    assert passed[0] is cpu_view and passed[1] is plain and passed[2] == 7


def test_log_graphs_wraps_the_registered_backend(monkeypatch, caplog):
    from torch._dynamo.backends import registry

    monkeypatch.setenv("SGLANG_RBLN_LOG_GRAPHS", "1")
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
@pytest.mark.parametrize(("preset", "expected"), [(None, "1"), ("0", "0")])
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
    def clean_env(self, monkeypatch):
        for name in envs.environment_variables:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("SGLANG_RBLN_DISABLE_COMPILE_CACHE", "1")

    @staticmethod
    def switches(module=None) -> dict:
        options = compilation.compile_options(module or _TinyMlp())
        del options["compile_context"]
        return options

    def test_context_is_per_module(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SGLANG_RBLN_DISABLE_COMPILE_CACHE")
        monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
        module = _TinyMlp()
        first = compilation.compile_options(module)
        second = compilation.compile_options(module)
        assert first["compile_context"] is second["compile_context"]
        assert (
            first["compile_context"]
            is not compilation.compile_options(_TinyMlp())["compile_context"]
        )
        assert first["cache_dir"] == str(tmp_path)
        assert first["mode"] == ["strict"]

    def test_context_shares_weights_across_graphs(self):
        context = compilation.compile_options(_TinyMlp())["compile_context"]
        assert context.state.use_weight_sharing

    @pytest.mark.parametrize(
        ("env", "expected"),
        [
            ({}, {"mode": ["strict"]}),
            ({"SGLANG_RBLN_COMPILE_STRICT": "0"}, {}),
            (
                {"SGLANG_RBLN_STATIC_OUTPUT": "1"},
                {"mode": ["strict"], "use_static_output": True},
            ),
            ({"SGLANG_RBLN_NUM_DEVICES": "4"}, {"mode": ["strict"], "num_devices": 4}),
            (
                {"SGLANG_RBLN_COMPILE_ONLY": "1", "SGLANG_RBLN_COMPILE_STRICT": "0"},
                {"mode": ["compile_only"]},
            ),
            ({"SGLANG_RBLN_COMPILE_ONLY": "1"}, {"mode": ["compile_only", "strict"]}),
        ],
        ids=[
            "strict-by-default",
            "fallback-allowed",
            "static-output",
            "num-devices",
            "compile-only",
            "both",
        ],
    )
    def test_switches(self, monkeypatch, env, expected):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        assert self.switches() == expected


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

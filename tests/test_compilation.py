# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch._dynamo import list_backends

from sglang_rbln import device
from sglang_rbln.compilation import backend as compilation
from sglang_rbln.platform import RBLNPlatform

ON_NPU_HOST = bool(device.device_nodes())


class _TinyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(64, 256)
        self.down = nn.Linear(256, 64)

    def forward(self, hidden):
        return self.down(torch.nn.functional.gelu(self.up(hidden)))


def test_rebel_backend_is_available():
    assert compilation.ensure_compile_backend() == "rbln"
    assert "rbln" in list_backends()
    assert torch._dynamo.config.cache_size_limit >= compilation.DYNAMO_CACHE_SIZE_LIMIT
    assert torch._dynamo.config.automatic_dynamic_shapes is False


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
    assert totals.host_ops == {"host_pad"}
    assert totals.top_nodes(1) == [("main_0", 425)]
    assert "device share 65.6%" in totals.summary()


def test_dump_graph_writes_readable_inputs_and_export(tmp_path):
    graph = torch.fx.symbolic_trace(_TinyMlp())
    stem = compilation.dump_graph(graph, [torch.zeros(2, 64)], tmp_path)
    assert "gelu" in stem.with_suffix(".txt").read_text()
    assert torch.load(stem.with_suffix(".inputs.pt"))[0].shape == (2, 64)


def test_log_graphs_wraps_the_registered_backend(monkeypatch, caplog):
    from torch._dynamo.backends import registry

    monkeypatch.setenv(compilation.LOG_GRAPHS_ENV, "1")
    original = registry._COMPILER_FNS["rbln"]
    monkeypatch.setitem(registry._COMPILER_FNS, "rbln", original)
    compilation.ensure_compile_backend()
    wrapped = registry._COMPILER_FNS["rbln"]
    assert wrapped.__wrapped_backend__ is original
    compilation.ensure_compile_backend()
    assert registry._COMPILER_FNS["rbln"] is wrapped


def test_platform_compiles_through_rbln():
    platform = RBLNPlatform()
    assert platform.get_compile_backend("max-autotune") == "rbln"
    assert platform.simple_compile_backend == "rbln"


class TestCompileOptions:
    def test_context_is_per_module(self, monkeypatch, tmp_path):
        monkeypatch.setenv(compilation.CACHE_DIR_ENV, str(tmp_path))
        for flag in (
            compilation.COMPILE_ONLY_ENV,
            compilation.STRICT_ENV,
            compilation.DISABLE_CACHE_ENV,
        ):
            monkeypatch.delenv(flag, raising=False)
        module = _TinyMlp()
        first = compilation.compile_options(module)
        second = compilation.compile_options(module)
        assert first["compile_context"] is second["compile_context"]
        assert (
            first["compile_context"]
            is not compilation.compile_options(_TinyMlp())["compile_context"]
        )
        assert first["cache_dir"] == str(tmp_path)
        assert first["mega_cache_only"] is True
        assert "mode" not in first

    def test_static_output_switch(self, monkeypatch):
        monkeypatch.setenv(compilation.STATIC_OUTPUT_ENV, "1")
        monkeypatch.setenv(compilation.DISABLE_CACHE_ENV, "1")
        assert compilation.compile_options() == {"use_static_output": True}

    @pytest.mark.parametrize(
        ("flags", "modes"),
        [
            ((compilation.COMPILE_ONLY_ENV,), ["compile_only"]),
            ((compilation.STRICT_ENV,), ["strict"]),
            (
                (compilation.COMPILE_ONLY_ENV, compilation.STRICT_ENV),
                ["compile_only", "strict"],
            ),
        ],
        ids=["compile-only", "strict", "both"],
    )
    def test_mode_and_cache_switches(self, monkeypatch, flags, modes):
        for flag in flags:
            monkeypatch.setenv(flag, "1")
        monkeypatch.setenv(compilation.DISABLE_CACHE_ENV, "1")
        assert compilation.compile_options() == {"mode": modes}


class TestServerArgDefaults:
    @pytest.fixture
    def server_args(self):
        return SimpleNamespace(
            enable_torch_compile=False,
            regional_compile=False,
            component_precisions={},
            _explicit_arg_names=set(),
        )

    def test_compiles_per_block_through_the_platform(self, server_args):
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert server_args.enable_torch_compile is True
        assert server_args.regional_compile is True

    @pytest.mark.parametrize(
        ("on_host", "expected"),
        [(True, {"vocoder": "fp32"}), (False, {})],
        ids=["host", "device"],
    )
    def test_host_mode_runs_the_vocoder_in_fp32(
        self, server_args, monkeypatch, on_host, expected
    ):
        monkeypatch.setattr(RBLNPlatform, "is_cpu", lambda self: on_host)
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert server_args.component_precisions == expected

    def test_keeps_an_explicit_vocoder_precision(self, server_args, monkeypatch):
        monkeypatch.setattr(RBLNPlatform, "is_cpu", lambda self: True)
        server_args.component_precisions = {"vocoder": "bf16"}
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert server_args.component_precisions == {"vocoder": "bf16"}

    @pytest.mark.parametrize("flag", ["enable_torch_compile", "regional_compile"])
    def test_keeps_an_explicit_choice(self, server_args, flag):
        server_args._explicit_arg_names = {flag}
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert getattr(server_args, flag) is False


@pytest.mark.xfail(not ON_NPU_HOST, reason="no RBLN device nodes on host")
def test_compiled_module_runs_on_the_npu(monkeypatch, tmp_path):
    monkeypatch.setenv(compilation.CACHE_DIR_ENV, str(tmp_path))
    monkeypatch.delenv(compilation.COMPILE_ONLY_ENV, raising=False)
    monkeypatch.delenv(compilation.DISABLE_CACHE_ENV, raising=False)
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

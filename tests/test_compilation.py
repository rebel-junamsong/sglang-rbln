# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch._dynamo import list_backends

from sglang_rbln import compilation, device
from sglang_rbln.platform import RblnPlatform

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


def test_platform_compiles_through_rbln():
    platform = RblnPlatform()
    assert platform.get_compile_backend("max-autotune") == "rbln"
    assert platform.simple_compile_backend == "rbln"


class TestCompileOptions:
    def test_context_is_per_module(self, monkeypatch, tmp_path):
        monkeypatch.setenv(compilation.CACHE_DIR_ENV, str(tmp_path))
        monkeypatch.delenv(compilation.COMPILE_ONLY_ENV, raising=False)
        monkeypatch.delenv(compilation.DISABLE_CACHE_ENV, raising=False)
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

    def test_compile_only_and_cache_switches(self, monkeypatch):
        monkeypatch.setenv(compilation.COMPILE_ONLY_ENV, "1")
        monkeypatch.setenv(compilation.DISABLE_CACHE_ENV, "1")
        options = compilation.compile_options()
        assert options == {"mode": ["compile_only"]}


class TestServerArgDefaults:
    @pytest.fixture
    def server_args(self):
        return SimpleNamespace(
            enable_torch_compile=False,
            regional_compile=False,
            _explicit_arg_names=set(),
        )

    def test_compiles_per_block_through_the_platform(self, server_args):
        RblnPlatform().apply_server_args_defaults(server_args)
        assert server_args.enable_torch_compile is True
        assert server_args.regional_compile is True

    @pytest.mark.parametrize("flag", ["enable_torch_compile", "regional_compile"])
    def test_keeps_an_explicit_choice(self, server_args, flag):
        server_args._explicit_arg_names = {flag}
        RblnPlatform().apply_server_args_defaults(server_args)
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

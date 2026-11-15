# SPDX-License-Identifier: Apache-2.0
import importlib
import importlib.util
import sys
from types import ModuleType

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.patches import cuda_imports
from sglang_rbln.patches.layers import layernorm
from sglang_rbln.patches.layers.attention import video_sparse_attn
from sglang_rbln.patches.models import ltx_2


class TestTritonNormExports:
    def test_binds_only_missing_names(self):
        exports_module = ModuleType("fake_exports")
        exports_module.rms_norm_fn = real_kernel = object()
        assert layernorm.register(exports_module) == ["norm_infer"]
        assert exports_module.rms_norm_fn is real_kernel
        with pytest.raises(NotImplementedError):
            exports_module.norm_infer()

    def test_is_idempotent(self):
        exports_module = ModuleType("fake_exports")
        assert layernorm.register(exports_module) == list(layernorm.TRITON_NORM_EXPORTS)
        assert layernorm.register(exports_module) == []


class TestCudaImportGuard:
    @pytest.fixture(autouse=True)
    def clean_meta_path(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "meta_path",
            [
                finder
                for finder in sys.meta_path
                if not isinstance(finder, cuda_imports._BlockedImportFinder)
            ],
        )
        monkeypatch.delitem(sys.modules, "vsa", raising=False)

    def test_blocks_the_cuda_only_module(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(cuda_imports, "_imports_cleanly", lambda module: False)
        assert video_sparse_attn.register()
        with pytest.raises(ImportError, match="blocked on rbln"):
            importlib.import_module("vsa")
        assert importlib.util.find_spec("vsa") is not None
        assert "vsa" not in sys.modules
        assert video_sparse_attn.register()
        finders = [
            f for f in sys.meta_path if isinstance(f, cuda_imports._BlockedImportFinder)
        ]
        assert len(finders) == 1 and finders[0].modules == {"vsa"}

    def test_leaves_cuda_hosts_alone(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert not video_sparse_attn.register()
        assert not any(
            isinstance(f, cuda_imports._BlockedImportFinder) for f in sys.meta_path
        )

    def test_leaves_importable_modules_alone(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(video_sparse_attn, "BLOCKED_MODULE", "json")
        assert not video_sparse_attn.register()

    def test_does_not_shadow_an_imported_module(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setitem(sys.modules, "vsa", ModuleType("vsa"))
        assert not video_sparse_attn.register()


class TestLtx2RopeHook:
    def test_returns_contiguous_tables(self):
        table = torch.arange(2 * 4 * 3 * 8, dtype=torch.bfloat16).view(2, 4, 3, 8)
        cos, sin = ltx_2.contiguous_rotary_tables(
            (table.swapaxes(1, 2), table.swapaxes(1, 2))
        )
        assert cos.is_contiguous() and sin.is_contiguous()
        assert torch.equal(cos, table.swapaxes(1, 2))

    @pytest.mark.parametrize("target", ltx_2.HOOK_TARGETS)
    def test_registers_an_around_hook(self, target):
        assert target in ltx_2.register()
        assert any(
            hook is ltx_2.cached_rotary_tables and kind is HookType.AROUND
            for kind, hook, _ in HookRegistry._hooks[target]
        )

    def test_replaces_the_triton_ada_values_kernel(self):
        assert ltx_2.FUSED_ADA_VALUES in ltx_2.register()
        assert any(
            hook is ltx_2.without_triton_ada_values and kind is HookType.REPLACE
            for kind, hook, _ in HookRegistry._hooks[ltx_2.FUSED_ADA_VALUES]
        )
        table = torch.zeros(9, 256, dtype=torch.bfloat16)
        assert ltx_2.without_triton_ada_values(table, 1, table.view(1, 1, -1)) is None

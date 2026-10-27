# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.patches.layers import layernorm
from sglang_rbln.patches.layers.attention import video_sparse_attn
from sglang_rbln.patches.models import ltx_2


class TestTritonNormExports:
    def test_binds_only_missing_names(self):
        exports_module = ModuleType("fake_exports")
        exports_module.rms_norm_fn = real_kernel = object()
        assert layernorm.bind_missing_exports(exports_module) == ["norm_infer"]
        assert exports_module.rms_norm_fn is real_kernel
        with pytest.raises(NotImplementedError):
            exports_module.norm_infer()

    def test_is_idempotent(self):
        exports_module = ModuleType("fake_exports")
        assert layernorm.bind_missing_exports(exports_module) == list(
            layernorm.TRITON_NORM_EXPORTS
        )
        assert layernorm.bind_missing_exports(exports_module) == []


class TestVsaImportBlock:
    @pytest.fixture(autouse=True)
    def clean_meta_path(self, monkeypatch):
        monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
        monkeypatch.delitem(sys.modules, "vsa", raising=False)

    def test_blocks_vsa_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert video_sparse_attn.block_without_cuda()
        with pytest.raises(ImportError, match="blocked on rbln"):
            import vsa  # noqa: F401
        assert video_sparse_attn.block_without_cuda()
        assert (
            sum(
                isinstance(f, video_sparse_attn._BlockedImportFinder)
                for f in sys.meta_path
            )
            == 1
        )

    def test_leaves_cuda_hosts_alone(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert not video_sparse_attn.block_without_cuda()
        assert not any(
            isinstance(f, video_sparse_attn._BlockedImportFinder) for f in sys.meta_path
        )

    def test_does_not_shadow_an_imported_vsa(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setitem(sys.modules, "vsa", ModuleType("vsa"))
        assert not video_sparse_attn.block_without_cuda()


class TestLtx2RopeHook:
    def test_returns_contiguous_tables(self):
        table = torch.arange(2 * 4 * 3 * 8, dtype=torch.bfloat16).view(2, 4, 3, 8)
        cos, sin = ltx_2.contiguous_rotary_tables(
            (table.swapaxes(1, 2), table.swapaxes(1, 2))
        )
        assert cos.is_contiguous() and sin.is_contiguous()
        assert torch.equal(cos, table.swapaxes(1, 2))

    def test_registers_an_after_hook(self):
        assert ltx_2.register() == ltx_2.HOOK_TARGET
        assert any(
            hook is ltx_2.contiguous_rotary_tables and kind is HookType.AFTER
            for kind, hook, _ in HookRegistry._hooks[ltx_2.HOOK_TARGET]
        )

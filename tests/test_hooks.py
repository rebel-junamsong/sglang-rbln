# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.hooks import ltx2_rope, triton_norm, vsa_import


class TestTritonNormExports:
    def test_binds_only_missing_names(self):
        exports_module = ModuleType("fake_exports")
        exports_module.rms_norm_fn = real_kernel = object()
        assert triton_norm.bind_missing_exports(exports_module) == ["norm_infer"]
        assert exports_module.rms_norm_fn is real_kernel
        with pytest.raises(NotImplementedError):
            exports_module.norm_infer()

    def test_is_idempotent(self):
        exports_module = ModuleType("fake_exports")
        assert triton_norm.bind_missing_exports(exports_module) == list(
            triton_norm.TRITON_NORM_EXPORTS
        )
        assert triton_norm.bind_missing_exports(exports_module) == []


class TestVsaImportBlock:
    @pytest.fixture(autouse=True)
    def clean_meta_path(self, monkeypatch):
        monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
        monkeypatch.delitem(sys.modules, "vsa", raising=False)

    def test_blocks_vsa_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert vsa_import.block_without_cuda()
        with pytest.raises(ImportError, match="blocked on rbln"):
            import vsa  # noqa: F401
        assert vsa_import.block_without_cuda()
        assert (
            sum(isinstance(f, vsa_import._BlockedImportFinder) for f in sys.meta_path)
            == 1
        )

    def test_leaves_cuda_hosts_alone(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert not vsa_import.block_without_cuda()
        assert not any(
            isinstance(f, vsa_import._BlockedImportFinder) for f in sys.meta_path
        )

    def test_does_not_shadow_an_imported_vsa(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setitem(sys.modules, "vsa", ModuleType("vsa"))
        assert not vsa_import.block_without_cuda()


class TestLtx2RopeHook:
    def test_returns_contiguous_tables(self):
        table = torch.arange(2 * 4 * 3 * 8, dtype=torch.bfloat16).view(2, 4, 3, 8)
        cos, sin = ltx2_rope.contiguous_rotary_tables(
            (table.swapaxes(1, 2), table.swapaxes(1, 2))
        )
        assert cos.is_contiguous() and sin.is_contiguous()
        assert torch.equal(cos, table.swapaxes(1, 2))

    def test_registers_an_after_hook(self):
        assert ltx2_rope.register() == ltx2_rope.HOOK_TARGET
        assert any(
            hook is ltx2_rope.contiguous_rotary_tables and kind is HookType.AFTER
            for kind, hook, _ in HookRegistry._hooks[ltx2_rope.HOOK_TARGET]
        )

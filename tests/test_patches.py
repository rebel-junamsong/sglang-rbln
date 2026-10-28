# SPDX-License-Identifier: Apache-2.0
import importlib
import itertools
import sys
from types import ModuleType

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.patches import cuda_imports
from sglang_rbln.patches.layers import layernorm
from sglang_rbln.patches.layers.attention import video_sparse_attn
from sglang_rbln.patches.models import ltx_2
from sglang_rbln.patches.srt.layers.quantization import fp4_utils
from sglang_rbln.patches.torch_rbln import ops_utils


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


class TestCudaImportGuard:
    @pytest.fixture(autouse=True)
    def clean_meta_path(self, monkeypatch):
        monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
        for name in ("vsa", "flashinfer"):
            monkeypatch.delitem(sys.modules, name, raising=False)

    @pytest.mark.parametrize(
        ("patch", "blocked"),
        [(video_sparse_attn, "vsa"), (fp4_utils, "flashinfer")],
        ids=["vsa", "flashinfer"],
    )
    def test_blocks_the_cuda_only_module(self, monkeypatch, patch, blocked):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(cuda_imports, "_imports_cleanly", lambda module: False)
        assert patch.register()
        with pytest.raises(ImportError, match="blocked on rbln"):
            importlib.import_module(blocked)
        assert patch.register()
        finders = [
            f for f in sys.meta_path if isinstance(f, cuda_imports._BlockedImportFinder)
        ]
        assert len(finders) == 1 and blocked in finders[0].modules

    def test_one_finder_serves_both_modules(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(cuda_imports, "_imports_cleanly", lambda module: False)
        video_sparse_attn.register()
        fp4_utils.register()
        finders = [
            f for f in sys.meta_path if isinstance(f, cuda_imports._BlockedImportFinder)
        ]
        assert len(finders) == 1 and finders[0].modules == {"vsa", "flashinfer"}

    def test_leaves_cuda_hosts_alone(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert not video_sparse_attn.register()
        assert not any(
            isinstance(f, cuda_imports._BlockedImportFinder) for f in sys.meta_path
        )

    def test_leaves_importable_modules_alone(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(fp4_utils, "BLOCKED_MODULE", "json")
        assert not fp4_utils.register()

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

    def test_registers_an_after_hook(self):
        assert ltx_2.register() == ltx_2.HOOK_TARGET
        assert any(
            hook is ltx_2.contiguous_rotary_tables and kind is HookType.AFTER
            for kind, hook, _ in HookRegistry._hooks[ltx_2.HOOK_TARGET]
        )


class TestTorchRblnPromotion:
    REAL_DTYPES = (
        torch.bool,
        torch.int32,
        torch.int64,
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
    )

    @pytest.mark.parametrize("shapes", [((), ()), ((), (2,)), ((2,), ()), ((2,), (2,))])
    def test_matches_torch_result_type(self, shapes):
        for left, right in itertools.product(self.REAL_DTYPES, repeat=2):
            tensors = [
                torch.ones(shapes[0], dtype=left),
                torch.ones(shapes[1], dtype=right),
            ]
            assert ops_utils.result_dtype(tensors) == torch.result_type(*tensors)

    def test_zero_dim_float32_scale_keeps_a_bfloat16_output(self):
        scale = torch.tensor(0.5)
        sample = torch.ones(4, dtype=torch.bfloat16)
        assert ops_utils.is_type_promotion_allowed((scale, sample), sample)
        assert not ops_utils.is_type_promotion_allowed(
            (scale.expand(4), sample), sample
        )

    def test_rejects_inputs_without_tensors(self):
        with pytest.raises(TypeError, match="No tensor inputs"):
            ops_utils.is_type_promotion_allowed((1.0,), torch.ones(1))

    def test_binds_into_both_torch_rbln_modules(self, monkeypatch):
        fakes = {name: ModuleType(name) for name in ops_utils.PATCHED_MODULES}
        for name, module in fakes.items():
            module.is_type_promotion_allowed = object()
            monkeypatch.setitem(sys.modules, name, module)
        assert ops_utils.register()
        assert all(
            module.is_type_promotion_allowed is ops_utils.is_type_promotion_allowed
            for module in fakes.values()
        )

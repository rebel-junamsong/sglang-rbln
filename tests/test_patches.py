# SPDX-License-Identifier: Apache-2.0
import importlib
import itertools
import sys
from types import ModuleType

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

from sglang_rbln.patches import cuda_imports
from sglang_rbln.patches.layers import layernorm
from sglang_rbln.patches.layers.attention import video_sparse_attn
from sglang_rbln.patches.models import cosmos3video, ltx_2
from sglang_rbln.patches.srt.layers.quantization import fp4_utils
from sglang_rbln.patches.torch_rbln import ops_utils
from sglang_rbln.platform import RUNS_ON_DEVICE


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


@pytest.mark.usefixtures("rbln_platform")
class TestQkNormRopePrebuild:
    @pytest.fixture(autouse=True)
    def fresh_rope_dict(self, monkeypatch):
        from sglang.multimodal_gen.runtime.layers import layernorm as upstream

        monkeypatch.setattr(upstream, "_QK_NORM_ROPE_DICT", {})
        return upstream._QK_NORM_ROPE_DICT

    @staticmethod
    def qk_norm_rope_graphs() -> tuple[int, int]:
        from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
        from sglang.multimodal_gen.runtime.models.dits import cosmos3video as dit

        tokens, head_dim = 8, 128
        q_norm, k_norm = RMSNorm(head_dim), RMSNorm(head_dim)

        def qk_norm_rope(qkv, cos_sin_cache, positions):
            return dit._apply_qwen3_qk_norm_rope(
                qkv[:, :, :4],
                qkv[:, :, 4:6],
                q_norm,
                k_norm,
                head_dim,
                cos_sin_cache,
                positions,
            )

        torch._dynamo.reset()
        explained = torch._dynamo.explain(qk_norm_rope)(
            torch.randn(1, tokens, 8, head_dim),
            torch.randn(tokens, head_dim),
            torch.arange(tokens),
        )
        return explained.graph_count, explained.graph_break_count

    def test_builds_the_rotary_module_once(self, fresh_rope_dict):
        rotary_emb = layernorm.prebuild_qk_norm_rope(128)
        assert fresh_rope_dict == {(128, True): rotary_emb}
        assert layernorm.prebuild_qk_norm_rope(128) is rotary_emb
        assert rotary_emb.is_neox_style and not rotary_emb.use_precomputed_cache

    def test_lazy_rotary_module_splits_the_graph(self):
        assert self.qk_norm_rope_graphs()[1] > 0

    def test_prebuilt_rotary_module_traces_as_one_graph(self):
        layernorm.prebuild_qk_norm_rope(128)
        assert self.qk_norm_rope_graphs() == (1, 0)


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


class _TimestepEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 4)

    def forward(self, t):
        return nn.functional.silu(self.linear(t[:, None]))


class TestCosmos3TimestepEmbedderHook:
    forward = staticmethod(_TimestepEmbedder.forward)

    def test_host_weights_run_the_original(self):
        embedder, t = _TimestepEmbedder(), torch.tensor([0.5, 0.25])
        out = cosmos3video.embed_timesteps_on_host(self.forward, embedder, t)
        assert torch.equal(out, embedder(t))
        assert not hasattr(embedder, cosmos3video.HOST_STATE)

    @pytest.mark.xfail(not RUNS_ON_DEVICE, reason="execution mode is host")
    def test_device_weights_embed_in_float32_on_the_host(self):
        embedder, t = _TimestepEmbedder(), torch.tensor([0.5, 0.25])
        expected = embedder(t)
        embedder.to("rbln:0")
        out = cosmos3video.embed_timesteps_on_host(self.forward, embedder, t)
        assert out.device.type == "rbln"
        assert torch.equal(out.cpu(), expected)


class TestRunOnHost:
    @staticmethod
    def norm() -> nn.Module:
        norm = nn.RMSNorm(8, eps=1e-6)
        nn.init.uniform_(norm.weight, 0.5, 1.5)
        return norm

    def test_host_weights_stay_untouched(self):
        norm = self.norm()
        assert not cosmos3video.run_on_host(norm)
        assert "forward" not in vars(norm)

    @pytest.mark.xfail(not RUNS_ON_DEVICE, reason="execution mode is host")
    def test_device_weights_run_in_float32_on_the_host(self):
        norm, hidden = self.norm(), torch.randn(2, 3, 8)
        expected = norm(hidden)
        norm.to("rbln:0")
        assert cosmos3video.run_on_host(norm)
        out = norm(hidden.to("rbln:0"))
        assert out.device.type == "rbln"
        assert torch.equal(out.cpu(), expected)
        assert next(norm.parameters()).device.type == "rbln"

    @pytest.mark.xfail(not RUNS_ON_DEVICE, reason="execution mode is host")
    def test_nested_outputs_return_to_the_device(self):
        layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        layers.forward = lambda hidden: [(layer(hidden), hidden) for layer in layers]
        hidden = torch.randn(2, 8)
        expected = layers(hidden)
        layers.to("rbln:0")
        assert cosmos3video.run_on_host(layers)
        out = layers(hidden.to("rbln:0"))
        assert all(t.device.type == "rbln" for pair in out for t in pair)
        assert torch.equal(out[1][0].cpu(), expected[1][0])

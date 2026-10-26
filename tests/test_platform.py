# SPDX-License-Identifier: Apache-2.0
from importlib.metadata import entry_points
from pkgutil import resolve_name

import pytest
import torch
from sglang.multimodal_gen.runtime import platforms
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.platforms import PlatformEnum

import sglang_rbln
from sglang_rbln import device
from sglang_rbln.platform import RblnPlatform

try:
    import torch_rbln  # noqa: F401

    HAS_TORCH_RBLN = True
except ImportError:
    HAS_TORCH_RBLN = False

ON_NPU_HOST = bool(device.device_nodes())


def test_entry_points_are_registered():
    groups = {
        "sglang.multimodal_gen.platforms": ("rbln", "sglang_rbln:activate"),
        "sglang.multimodal_gen.plugins": ("rbln_hooks", "sglang_rbln.hooks:register"),
    }
    for group, (name, value) in groups.items():
        (ep,) = [ep for ep in entry_points(group=group) if ep.name == name]
        assert ep.value == value
    assert sglang_rbln.ENTRY_POINT_NAME == "rbln"


class TestActivate:
    @pytest.fixture(autouse=True)
    def no_forced_name(self, monkeypatch):
        monkeypatch.delenv(device.FORCE_NPU_NAME_ENV, raising=False)

    def test_returns_none_without_device_nodes(self, monkeypatch):
        monkeypatch.setattr(device, "device_nodes", list)
        assert sglang_rbln.activate() is None

    def test_returns_none_without_the_rebel_distribution(self, monkeypatch):
        monkeypatch.setattr(device.importlib.util, "find_spec", lambda name: None)
        assert sglang_rbln.activate() is None

    def test_forced_npu_name_activates_off_host(self, monkeypatch):
        monkeypatch.setattr(device, "device_nodes", list)
        monkeypatch.setenv(device.FORCE_NPU_NAME_ENV, "RBLN-CA25")
        assert sglang_rbln.activate() == sglang_rbln.PLATFORM_CLASS_PATH

    @pytest.mark.xfail(not ON_NPU_HOST, reason="no RBLN device nodes on host")
    def test_activates_on_an_npu_host(self):
        assert sglang_rbln.activate() == sglang_rbln.PLATFORM_CLASS_PATH

    def test_activate_does_not_import_rebel(self, monkeypatch):
        # activate() runs in every spawned process; importing rebel costs seconds.
        import builtins

        real_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            assert name != "rebel", "activate() imported rebel"
            return real_import(name, *args, **kwargs)

        monkeypatch.setenv(device.FORCE_NPU_NAME_ENV, "RBLN-CA25")
        monkeypatch.setattr(builtins, "__import__", guarded)
        assert sglang_rbln.activate() == sglang_rbln.PLATFORM_CLASS_PATH


class TestSglangResolvesThePlugin:
    """Go through SGLang's own resolver, not our code, so the OOT contract
    (entry-point name, PlatformEnum.OOT, device_name/device_type, dispatch_key)
    is checked by the consumer."""

    def test_override_selects_rbln(self, fresh_platform, monkeypatch):
        monkeypatch.setenv(device.FORCE_NPU_NAME_ENV, "RBLN-CA25")
        monkeypatch.setenv("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "rbln")
        assert (
            platforms.resolve_current_platform_cls_qualname()
            == sglang_rbln.PLATFORM_CLASS_PATH
        )
        platform = platforms.current_platform
        assert isinstance(platform, RblnPlatform)
        assert platform.is_out_of_tree()
        assert platforms.get_selected_platform_dist() == "sglang-rbln"

    def test_platform_passes_oot_validation(self, fresh_platform):
        cls = platforms._load_platform_class(
            sglang_rbln.PLATFORM_CLASS_PATH, external=True
        )
        platform = cls()
        assert platform._enum is PlatformEnum.OOT
        assert platform.get_torch_library_dispatch_key() == "PrivateUse1"
        assert platform.get_dispatch_key_name() == "rbln"


class _Op(CustomOp):
    def forward_native(self, x):
        return ("native", x)


class TestCustomOpDispatch:
    def test_unregistered_op_falls_back_to_native(self, monkeypatch):
        monkeypatch.setattr(platforms, "_current_platform", RblnPlatform())
        assert _Op()(3) == ("native", 3)

    def test_registered_rbln_forward_wins(self, monkeypatch):
        monkeypatch.setattr(platforms, "_current_platform", RblnPlatform())
        CustomOp.register_oot_forward(
            _Op, fn=lambda self, x: ("rbln", x), platform_key="rbln"
        )
        try:
            assert _Op()(3) == ("rbln", 3)
        finally:
            CustomOp._oot_forward_registry.pop("rbln", None)


class TestDeviceQueries:
    def test_tensor_on_device_is_by_device_type(self):
        assert not RblnPlatform().tensor_on_device(torch.zeros(1))

    def test_attention_backend_is_importable(self):
        cls = resolve_name(RblnPlatform().get_attn_backend_cls_str())
        assert cls.__name__ == "SDPABackend"

    @pytest.mark.xfail(not HAS_TORCH_RBLN, reason="torch-rbln not importable")
    def test_local_torch_device_is_rbln(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "1")
        dev = RblnPlatform.get_local_torch_device()
        assert (dev.type, dev.index) == ("rbln", 1)

    @pytest.mark.xfail(not ON_NPU_HOST, reason="no RBLN device nodes on host")
    def test_memory_queries_answer_from_rebel(self, monkeypatch):
        monkeypatch.delenv(device.FORCE_NPU_NAME_ENV, raising=False)
        total = RblnPlatform.get_device_total_memory(0)
        free_gib = RblnPlatform.get_available_gpu_memory(0)
        assert total > 0
        assert 0 < free_gib <= total / (1 << 30)
        assert RblnPlatform.get_device_name(0).startswith("RBLN")

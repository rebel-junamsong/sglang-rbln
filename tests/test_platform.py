# SPDX-License-Identifier: Apache-2.0
import builtins
from importlib.metadata import entry_points
from pkgutil import resolve_name

import pytest
import torch
from sglang.multimodal_gen.runtime import platforms
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.platforms import PlatformEnum

import sglang_rbln
from sglang_rbln import device
from sglang_rbln.platform import RUNS_ON_DEVICE, RblnPlatform

ON_NPU_HOST = bool(device.device_nodes())


def test_entry_points_are_registered():
    expected = {
        "sglang.multimodal_gen.platforms": ("rbln", "sglang_rbln:activate"),
        "sglang.multimodal_gen.plugins": ("rbln_hooks", "sglang_rbln.hooks:register"),
    }
    for group, (entry_name, entry_value) in expected.items():
        (entry,) = [ep for ep in entry_points(group=group) if ep.name == entry_name]
        assert entry.value == entry_value
    assert sglang_rbln.ENTRY_POINT_NAME == "rbln"


class TestActivate:
    @pytest.fixture(autouse=True)
    def no_forced_name(self, monkeypatch):
        monkeypatch.delenv(device.FORCE_NPU_NAME_ENV, raising=False)

    def test_returns_none_without_device_nodes(self, monkeypatch):
        monkeypatch.setattr(device, "device_nodes", list)
        assert sglang_rbln.activate() is None

    def test_returns_none_without_the_rebel_distribution(self, monkeypatch):
        monkeypatch.setattr(device, "_distribution_installed", lambda name: False)
        assert sglang_rbln.activate() is None

    def test_forced_npu_name_activates_off_host(self, monkeypatch):
        monkeypatch.setattr(device, "device_nodes", list)
        monkeypatch.setenv(device.FORCE_NPU_NAME_ENV, "RBLN-CA25")
        assert sglang_rbln.activate() == sglang_rbln.PLATFORM_CLASS_PATH

    @pytest.mark.xfail(not ON_NPU_HOST, reason="no RBLN device nodes on host")
    def test_activates_on_an_npu_host(self):
        assert sglang_rbln.activate() == sglang_rbln.PLATFORM_CLASS_PATH

    def test_activate_does_not_import_rebel(self, monkeypatch):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            assert name != "rebel", "activate() imported rebel"
            return real_import(name, *args, **kwargs)

        monkeypatch.setenv(device.FORCE_NPU_NAME_ENV, "RBLN-CA25")
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        assert sglang_rbln.activate() == sglang_rbln.PLATFORM_CLASS_PATH


class TestExecutionMode:
    @pytest.mark.parametrize("requested", ["host", "device"])
    def test_explicit_mode_wins(self, monkeypatch, requested):
        monkeypatch.setenv(device.EXECUTION_MODE_ENV, requested)
        assert device.execution_mode() == requested

    def test_auto_follows_torch_rbln(self, monkeypatch):
        monkeypatch.delenv(device.EXECUTION_MODE_ENV, raising=False)
        monkeypatch.setattr(device, "torch_rbln_available", lambda: False)
        assert device.execution_mode() == "host"
        monkeypatch.setattr(device, "torch_rbln_available", lambda: True)
        assert device.execution_mode() == "device"

    def test_unknown_mode_is_rejected(self, monkeypatch):
        monkeypatch.setenv(device.EXECUTION_MODE_ENV, "npu")
        with pytest.raises(ValueError, match=device.EXECUTION_MODE_ENV):
            device.execution_mode()

    def test_platform_identity_matches_mode(self):
        platform = RblnPlatform()
        assert platform.device_name == "rbln"
        assert platform.is_out_of_tree()
        assert platform.get_dispatch_key_name() == "rbln"
        if RUNS_ON_DEVICE:
            assert platform.execution_mode == "device"
            assert (platform.device_type, platform.dispatch_key) == (
                "rbln",
                "PrivateUse1",
            )
            assert not platform.is_cpu()
        else:
            assert platform.execution_mode == "host"
            assert (platform.device_type, platform.dispatch_key) == ("cpu", "CPU")
            assert platform.is_cpu()
            assert platform.get_local_torch_device() == torch.device("cpu")
            assert platform.tensor_on_device(torch.zeros(1))


class TestSglangResolvesThePlugin:
    def test_override_selects_rbln(self, fresh_platform, monkeypatch):
        monkeypatch.setenv(device.FORCE_NPU_NAME_ENV, "RBLN-CA25")
        monkeypatch.setenv("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "rbln")
        assert (
            platforms.resolve_current_platform_cls_qualname()
            == sglang_rbln.PLATFORM_CLASS_PATH
        )
        platform = platforms.current_platform
        assert isinstance(platform, RblnPlatform)
        assert platforms.get_selected_platform_dist() == "sglang-rbln"

    def test_platform_passes_oot_validation(self, fresh_platform):
        platform_cls = platforms._load_platform_class(
            sglang_rbln.PLATFORM_CLASS_PATH, external=True
        )
        platform = platform_cls()
        assert platform._enum is PlatformEnum.OOT
        assert platform.get_torch_library_dispatch_key() == platform.dispatch_key


class _NativeOnlyOp(CustomOp):
    def forward_native(self, x):
        return ("native", x)


class TestCustomOpDispatch:
    def test_unregistered_op_falls_back_to_native(self, monkeypatch):
        monkeypatch.setattr(platforms, "_current_platform", RblnPlatform())
        assert _NativeOnlyOp()(3) == ("native", 3)

    def test_registered_rbln_forward_wins(self, monkeypatch):
        monkeypatch.setattr(platforms, "_current_platform", RblnPlatform())
        CustomOp.register_oot_forward(
            _NativeOnlyOp, fn=lambda self, x: ("rbln", x), platform_key="rbln"
        )
        try:
            assert _NativeOnlyOp()(3) == ("rbln", 3)
        finally:
            CustomOp._oot_forward_registry.pop("rbln", None)


class TestDeviceQueries:
    def test_attention_backend_is_importable(self):
        backend_cls = resolve_name(RblnPlatform().get_attn_backend_cls_str())
        assert backend_cls.__name__ == "SDPABackend"

    def test_memory_queries_are_positive(self):
        total_bytes = RblnPlatform.get_device_total_memory(0)
        free_gib = RblnPlatform.get_available_gpu_memory(0)
        assert total_bytes > 0
        assert 0 < free_gib <= total_bytes / (1 << 30)

    @pytest.mark.xfail(not ON_NPU_HOST, reason="no RBLN device nodes on host")
    def test_device_name_comes_from_rebel(self, monkeypatch):
        monkeypatch.delenv(device.FORCE_NPU_NAME_ENV, raising=False)
        assert RblnPlatform.get_device_name(0).startswith("RBLN")

    @pytest.mark.xfail(not RUNS_ON_DEVICE, reason="execution mode is host")
    def test_local_torch_device_is_rbln(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "1")
        local_device = RblnPlatform.get_local_torch_device()
        assert (local_device.type, local_device.index) == ("rbln", 1)

# SPDX-License-Identifier: Apache-2.0
import builtins
import os
import sys
from importlib.metadata import entry_points
from pkgutil import resolve_name
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime import platforms
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.platforms import PlatformEnum

import sglang_rbln
from sglang_rbln import device
from sglang_rbln import platform as platform_module
from sglang_rbln.platform import OFFLOAD_FLAGS, RUNS_ON_DEVICE, RBLNPlatform
from tests.conftest import NPU_FAMILIES

ON_NPU_HOST = bool(device.device_nodes())


def test_entry_points_are_registered():
    expected = {
        "sglang.multimodal_gen.platforms": ("rbln", "sglang_rbln:activate"),
        "sglang.multimodal_gen.plugins": (
            "rbln_patches",
            "sglang_rbln.patches:register",
        ),
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

    @pytest.mark.parametrize("npu", NPU_FAMILIES)
    def test_forced_npu_name_activates_off_host(self, monkeypatch, npu):
        monkeypatch.setattr(device, "device_nodes", list)
        monkeypatch.setenv(device.FORCE_NPU_NAME_ENV, npu)
        assert sglang_rbln.activate() == sglang_rbln.PLATFORM_CLASS_PATH

    @pytest.mark.xfail(not ON_NPU_HOST, reason="no RBLN device nodes on host")
    def test_activates_on_an_npu_host(self):
        assert sglang_rbln.activate() == sglang_rbln.PLATFORM_CLASS_PATH

    def test_activate_does_not_import_rebel(self, monkeypatch):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            assert name != "rebel", "activate() imported rebel"
            return real_import(name, *args, **kwargs)

        monkeypatch.setenv(device.FORCE_NPU_NAME_ENV, NPU_FAMILIES[0])
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        assert sglang_rbln.activate() == sglang_rbln.PLATFORM_CLASS_PATH


class TestExecutionMode:
    @pytest.mark.parametrize("requested", ["host", "device"])
    def test_explicit_mode_wins(self, monkeypatch, requested):
        monkeypatch.setenv(device.EXECUTION_MODE_ENV, requested)
        assert device.execution_mode() == requested

    @pytest.mark.parametrize(
        ("torch_rbln_installed", "expected"),
        [(False, device.ExecutionMode.HOST), (True, device.ExecutionMode.DEVICE)],
        ids=["without-torch-rbln", "with-torch-rbln"],
    )
    def test_auto_follows_torch_rbln(self, monkeypatch, torch_rbln_installed, expected):
        monkeypatch.delenv(device.EXECUTION_MODE_ENV, raising=False)
        monkeypatch.setattr(
            device, "torch_rbln_available", lambda: torch_rbln_installed
        )
        assert device.execution_mode() is expected

    def test_unknown_mode_is_rejected(self, monkeypatch):
        monkeypatch.setenv(device.EXECUTION_MODE_ENV, "npu")
        with pytest.raises(ValueError, match=device.EXECUTION_MODE_ENV):
            device.execution_mode()

    def test_platform_identity_matches_mode(self):
        platform = RBLNPlatform()
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
    def test_override_selects_rbln(self, fresh_platform, rbln_override):
        assert (
            platforms.resolve_current_platform_cls_qualname()
            == sglang_rbln.PLATFORM_CLASS_PATH
        )
        platform = platforms.current_platform
        assert isinstance(platform, RBLNPlatform)
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
        monkeypatch.setattr(platforms, "_current_platform", RBLNPlatform())
        assert _NativeOnlyOp()(3) == ("native", 3)

    def test_registered_rbln_forward_wins(self, monkeypatch):
        monkeypatch.setattr(platforms, "_current_platform", RBLNPlatform())
        CustomOp.register_oot_forward(
            _NativeOnlyOp, fn=lambda self, x: ("rbln", x), platform_key="rbln"
        )
        try:
            assert _NativeOnlyOp()(3) == ("rbln", 3)
        finally:
            CustomOp._oot_forward_registry.pop("rbln", None)


class TestDeviceQueries:
    def test_attention_backend_is_importable(self):
        backend_cls = resolve_name(RBLNPlatform().get_attn_backend_cls_str())
        assert backend_cls.__name__ == "RBLNSdpaBackend"

    def test_memory_queries_are_positive(self):
        total_bytes = RBLNPlatform.get_device_total_memory(0)
        free_gib = RBLNPlatform.get_available_gpu_memory(0)
        assert total_bytes > 0
        assert 0 < free_gib <= total_bytes / (1 << 30)

    @pytest.mark.xfail(not ON_NPU_HOST, reason="no RBLN device nodes on host")
    def test_device_name_comes_from_rebel(self, monkeypatch):
        monkeypatch.delenv(device.FORCE_NPU_NAME_ENV, raising=False)
        assert RBLNPlatform.get_device_name(0).startswith("RBLN")

    @pytest.mark.xfail(not RUNS_ON_DEVICE, reason="execution mode is host")
    def test_local_torch_device_is_rbln(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "1")
        local_device = RBLNPlatform.get_local_torch_device()
        assert (local_device.type, local_device.index) == ("rbln", 1)


def test_platform_compiles_through_rbln():
    platform = RBLNPlatform()
    assert platform.get_compile_backend("max-autotune") == "rbln"
    assert platform.simple_compile_backend == "rbln"


class _ServerArgs(SimpleNamespace):
    def is_arg_explicitly_set(self, arg_name: str) -> bool:
        return arg_name in self.explicit_args


class TestServerArgDefaults:
    @pytest.fixture
    def server_args(self) -> _ServerArgs:
        return _ServerArgs(
            enable_torch_compile=False,
            regional_compile=False,
            component_precisions={},
            explicit_args=set(),
            **dict.fromkeys(OFFLOAD_FLAGS),
        )

    @pytest.fixture
    def on_host(self, monkeypatch: pytest.MonkeyPatch, request) -> bool:
        monkeypatch.setattr(RBLNPlatform, "is_cpu", lambda self: request.param)
        return request.param

    def test_compiles_per_block_through_the_platform(self, server_args):
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert server_args.enable_torch_compile is True
        assert server_args.regional_compile is True

    @pytest.mark.parametrize("flag", ["enable_torch_compile", "regional_compile"])
    def test_keeps_an_explicit_choice(self, server_args, flag):
        server_args.explicit_args = {flag}
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert getattr(server_args, flag) is False

    @pytest.mark.parametrize(
        ("on_host", "expected"),
        [(True, {"vocoder": "fp32"}), (False, {})],
        ids=["host", "device"],
        indirect=["on_host"],
    )
    def test_host_mode_runs_the_vocoder_in_fp32(self, server_args, on_host, expected):
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert server_args.component_precisions == expected

    @pytest.mark.parametrize("on_host", [True], indirect=True)
    def test_keeps_an_explicit_vocoder_precision(self, server_args, on_host):
        server_args.component_precisions = {"vocoder": "bf16"}
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert server_args.component_precisions == {"vocoder": "bf16"}

    @pytest.mark.parametrize("on_host", [False], indirect=True)
    def test_device_mode_keeps_components_resident(self, server_args, on_host):
        server_args.explicit_args = {"vae_cpu_offload"}
        server_args.vae_cpu_offload = True
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert [getattr(server_args, flag) for flag in OFFLOAD_FLAGS] == [
            False,
            False,
            False,
            True,
        ]


@pytest.mark.parametrize(
    ("preset", "expected"),
    [(None, platform_module.TORCH_RBLN_FALLBACK_DEFAULT), ("all", "all")],
    ids=["default", "explicit"],
)
def test_device_mode_forbids_silent_compile_fallback(monkeypatch, preset, expected):
    monkeypatch.setattr(platform_module, "RUNS_ON_DEVICE", True)
    monkeypatch.setitem(sys.modules, "torch_rbln", ModuleType("torch_rbln"))
    monkeypatch.delenv(platform_module.TORCH_RBLN_FALLBACK_ENV, raising=False)
    if preset is not None:
        monkeypatch.setenv(platform_module.TORCH_RBLN_FALLBACK_ENV, preset)
    RBLNPlatform().init_backend()
    assert os.environ[platform_module.TORCH_RBLN_FALLBACK_ENV] == expected

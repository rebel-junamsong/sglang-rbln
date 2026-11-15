# SPDX-License-Identifier: Apache-2.0
import builtins
import json
import os
import subprocess
import sys
from importlib.metadata import entry_points
from pkgutil import resolve_name
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime import platforms
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.platforms import PlatformEnum

import sglang_rbln
from sglang_rbln import device
from sglang_rbln import platform as platform_module
from sglang_rbln.platform import RBLNPlatform

NPU_FAMILIES = ("RBLN-CA22", "RBLN-CA25", "RBLN-CR13")


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


class TestActivate:
    @pytest.fixture(autouse=True)
    def no_forced_name(self, monkeypatch):
        monkeypatch.delenv("RBLN_FORCE_NPU_NAME", raising=False)

    def test_returns_none_without_device_nodes(self, monkeypatch):
        monkeypatch.setattr(device, "device_nodes", list)
        assert sglang_rbln.activate() is None

    def test_returns_none_without_the_rebel_distribution(self, monkeypatch):
        monkeypatch.setattr(device, "_module_available", lambda name: False)
        assert sglang_rbln.activate() is None

    @pytest.mark.parametrize("npu", NPU_FAMILIES)
    def test_forced_npu_name_activates_off_host(self, monkeypatch, npu):
        monkeypatch.setattr(device, "device_nodes", list)
        monkeypatch.setenv("RBLN_FORCE_NPU_NAME", npu)
        assert sglang_rbln.activate() == sglang_rbln.PLATFORM_CLASS_PATH

    @pytest.mark.npu
    def test_activates_on_an_npu_host(self):
        assert sglang_rbln.activate() == sglang_rbln.PLATFORM_CLASS_PATH

    def test_activate_does_not_import_rebel(self, monkeypatch):
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            assert name != "rebel", "activate() imported rebel"
            return real_import(name, *args, **kwargs)

        monkeypatch.setenv("RBLN_FORCE_NPU_NAME", NPU_FAMILIES[0])
        monkeypatch.setattr(builtins, "__import__", guarded_import)
        assert sglang_rbln.activate() == sglang_rbln.PLATFORM_CLASS_PATH


def test_upstream_sees_the_host():
    """Upstream's tensors are cpu ones; the compiled regions take them onto their
    NPUs, so every platform answer is the host's."""
    platform = RBLNPlatform()
    assert platform.device_name == "rbln"
    assert platform.is_out_of_tree()
    assert platform.get_dispatch_key_name() == "rbln"
    assert (platform.device_type, platform.dispatch_key) == ("cpu", "CPU")
    assert platform.is_cpu()
    assert platform.get_local_torch_device() == torch.device("cpu")
    assert platform.get_device(1) == torch.device("cpu")
    assert platform.tensor_on_device(torch.zeros(1))
    assert platform.is_float64_supported()
    communicator = resolve_name(RBLNPlatform.get_device_communicator_cls())
    assert communicator.__name__ == "CpuCommunicator"


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
    def test_attention_backend_is_upstreams(self):
        """The plugin uses upstream's SDPA backend. `Platform`'s method is a stub
        returning "", so the override stays and defers to upstream.
        """
        backend_cls = resolve_name(
            RBLNPlatform().get_attn_backend_cls_str(None, 128, torch.bfloat16)
        )
        assert backend_cls.__name__ == "SDPABackend"
        assert backend_cls.__module__.startswith("sglang.")

    @pytest.mark.npu  # reads the memory of a real NPU
    def test_memory_queries_are_positive(self):
        total_bytes = RBLNPlatform.get_device_total_memory(0)
        free_gib = RBLNPlatform.get_available_gpu_memory(0)
        assert total_bytes > 0
        assert 0 < free_gib <= total_bytes / (1 << 30)

    @pytest.mark.npu
    def test_device_name_comes_from_rebel(self, monkeypatch):
        monkeypatch.delenv("RBLN_FORCE_NPU_NAME", raising=False)
        assert RBLNPlatform.get_device_name(0).startswith("RBLN")


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
            offload_during_compile=True,
            dit_layerwise_offload=False,
            layerwise_offload_components=None,
            component_precisions={},
            explicit_args=set(),
            disable_autocast=False,
        )

    def test_compiles_through_the_platform(self, server_args):
        """The LTX DiT compiles whole, as upstream's default does; an explicit
        `--regional-compile` makes it block regions instead."""
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert server_args.enable_torch_compile is True
        assert server_args.regional_compile is False

    def test_keeps_an_explicit_choice(self, server_args):
        server_args.explicit_args = {"enable_torch_compile"}
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert server_args.enable_torch_compile is False

    def test_the_compile_warmup_keeps_the_dit_resident(self, server_args):
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert server_args.offload_during_compile is False

    @pytest.mark.parametrize(
        ("flag", "value"),
        [
            ("offload_during_compile", True),
            ("dit_layerwise_offload", True),
            ("layerwise_offload_components", ["transformer"]),
        ],
    )
    def test_rejects_layerwise_offload(self, server_args, flag, value):
        # it would move the resident weights to the host, and nothing moves them back
        setattr(server_args, flag, value)
        server_args.explicit_args = {flag}
        with pytest.raises(ValueError, match=f"--{flag.replace('_', '-')}"):
            RBLNPlatform().apply_server_args_defaults(server_args)

    @pytest.mark.parametrize(
        "flag", ["offload_during_compile", "dit_layerwise_offload"]
    )
    def test_accepts_layerwise_offload_turned_off(self, server_args, flag):
        setattr(server_args, flag, False)
        server_args.explicit_args = {flag}
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert getattr(server_args, flag) is False

    def test_runs_the_vocoder_in_fp32(self, server_args):
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert server_args.component_precisions == {"vocoder": "fp32"}

    def test_keeps_an_explicit_vocoder_precision(self, server_args):
        server_args.component_precisions = {"vocoder": "bf16"}
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert server_args.component_precisions == {"vocoder": "bf16"}

    def test_autocast_stays_as_upstream_sets_it(self, server_args):
        RBLNPlatform().apply_server_args_defaults(server_args)
        assert server_args.disable_autocast is False


@pytest.fixture
def device_groups(monkeypatch) -> list[None]:
    from sglang_rbln.compilation import residency

    configured = []
    monkeypatch.setattr(
        residency, "configure_device_group", lambda: configured.append(None)
    )
    return configured


def test_init_backend_lays_out_the_npu_sets(device_groups):
    RBLNPlatform().init_backend()
    assert device_groups == [None]


@pytest.mark.parametrize(
    ("preset", "expected"),
    [(None, platform_module.TORCH_RBLN_DISABLE_FALLBACK_DEFAULT), ("all", "all")],
    ids=["default", "explicit"],
)
def test_forbids_silent_compile_fallback(monkeypatch, device_groups, preset, expected):
    monkeypatch.delenv(platform_module.TORCH_RBLN_DISABLE_FALLBACK_ENV, raising=False)
    if preset is not None:
        monkeypatch.setenv(platform_module.TORCH_RBLN_DISABLE_FALLBACK_ENV, preset)
    RBLNPlatform().init_backend()
    assert os.environ[platform_module.TORCH_RBLN_DISABLE_FALLBACK_ENV] == expected


@pytest.mark.parametrize(
    ("compile_only", "preset", "expected"),
    [("1", None, "1"), ("0", None, None), ("1", "0", "0")],
    ids=["compile-only", "run", "explicit"],
)
def test_a_compile_only_run_takes_the_dummy_device(
    monkeypatch, device_groups, compile_only, preset, expected
):
    """It compiles on device tensors as a device run does, with no NPU to open."""
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_ONLY", compile_only)
    monkeypatch.delenv(platform_module.DUMMY_DEVICE_ENV, raising=False)
    if preset is not None:
        monkeypatch.setenv(platform_module.DUMMY_DEVICE_ENV, preset)
    RBLNPlatform().init_backend()
    assert os.environ.get(platform_module.DUMMY_DEVICE_ENV) == expected


def test_amp_query_traces_without_a_graph_break():
    torch._dynamo.reset()
    explained = torch._dynamo.explain(
        lambda x: x + 1 if RBLNPlatform.is_amp_supported() else x
    )(torch.ones(1))
    assert explained.graph_break_count == 0


class TestTensorParallelDegree:
    """`sglang serve --tp-size N` splits each model over N NPUs, as upstream's flag
    does on GPUs, with one worker driving them."""

    @pytest.fixture
    def server_args(self, monkeypatch) -> _ServerArgs:
        # set, not deleted, so the test's own writes are undone with it
        monkeypatch.setenv(platform_module.NUM_DEVICES_ENV, "")
        return _ServerArgs(tp_size=None, num_gpus=1, explicit_args=set())

    @pytest.mark.parametrize(
        ("explicit", "expected"),
        [
            ({"tp_size": 4}, "4"),
            ({"num_gpus": 4}, "4"),
            ({"tp_size": 4, "num_gpus": 8}, "4"),
            ({"tp_size": 1}, None),
            ({}, None),
        ],
        ids=["tp-size", "num-gpus", "tp-size-first", "one", "unset"],
    )
    def test_the_degree_becomes_the_npus_per_model(
        self, monkeypatch, server_args, explicit, expected
    ):
        for flag, value in explicit.items():
            setattr(server_args, flag, value)
        server_args.explicit_args = set(explicit)
        platform_module.set_num_devices_from_tensor_parallel_degree(server_args)
        assert (os.environ[platform_module.NUM_DEVICES_ENV] or None) == expected
        if expected is not None:
            assert (server_args.tp_size, server_args.num_gpus) == (1, 1)

    def test_an_agreeing_environment_is_kept(self, monkeypatch, server_args):
        monkeypatch.setenv(platform_module.NUM_DEVICES_ENV, "4")
        server_args.tp_size, server_args.explicit_args = 4, {"tp_size"}
        platform_module.set_num_devices_from_tensor_parallel_degree(server_args)
        assert os.environ[platform_module.NUM_DEVICES_ENV] == "4"

    def test_a_contradicting_environment_fails(self, monkeypatch, server_args):
        monkeypatch.setenv(platform_module.NUM_DEVICES_ENV, "2")
        server_args.tp_size, server_args.explicit_args = 4, {"tp_size"}
        with pytest.raises(ValueError, match="contradicts"):
            platform_module.set_num_devices_from_tensor_parallel_degree(server_args)


CPU_REFERENCE_PROBE = """
import json, sys
from sglang.multimodal_gen.runtime import platforms
from sglang.multimodal_gen.runtime.platforms import plugins
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
plugins.load_plugins(); plugins.apply_plugin_hooks()
print(json.dumps({
    "platform": type(platforms.current_platform).__name__,
    "hooks": sorted(HookRegistry._hooks),
    "plugin_platform": "sglang_rbln.platform" in sys.modules,
}))
"""


def test_a_cpu_reference_run_leaves_the_plugin_out():
    """The README's CPU reference: upstream's own platform, with no plugin hook."""
    env = {**os.environ, "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "cpu"}
    completed = subprocess.run(
        [sys.executable, "-c", CPU_REFERENCE_PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {
        "platform": "CpuPlatform",
        "hooks": [],
        "plugin_platform": False,
    }

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable

import pytest
import torch
from sglang.multimodal_gen.runtime import platforms

from sglang_rbln import device, envs
from sglang_rbln.patches.layers import layernorm
from sglang_rbln.patches.layers.attention import video_sparse_attn
from sglang_rbln.patches.srt.layers.quantization import fp4_utils


def pytest_configure(config: pytest.Config) -> None:
    # The import-time shims the plugin binds before any model module is imported;
    # test modules import models at collection, before the plugin registers.
    layernorm.register()
    video_sparse_attn.register()
    fp4_utils.register()


RBLN_DEVICE = "rbln"


def _names_the_device(value) -> bool:
    if isinstance(value, torch.device):
        return value.type == RBLN_DEVICE
    return isinstance(value, str) and value.split(":")[0] == RBLN_DEVICE


def on_a_dummy_device() -> bool:
    """True when the process cannot reach real hardware whatever it names."""
    return os.environ.get("RBLN_DUMMY_DEVICE", "").strip() not in ("", "0")


class ForbidDevice(torch.overrides.TorchFunctionMode):
    """Raise on any torch call that names the RBLN device.

    Checking the call rather than the process is what makes this per-test: one rbln
    tensor opens every device node and leaves them open, so anything that reads
    /proc/self/fd blames whichever test happens to run next and a single offender
    poisons the rest of the run.
    """

    MOVES = frozenset({"to", "to_empty", "_to_copy", "new_empty", "new_zeros"})

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # a device argument only: "rbln" also names the compile backend, which is a
        # host-side choice and not a tensor reaching the hardware
        candidates = [kwargs.get("device")]
        if getattr(func, "__name__", "") in self.MOVES:
            candidates += list(args[1:])
        named = [value for value in candidates if _names_the_device(value)]
        if named:
            raise AssertionError(
                f"{getattr(func, '__name__', func)} names the {named[0]} device in a "
                "test without @pytest.mark.npu. Mark it, or keep its tensors on the "
                "host: the gate runs -m 'not npu' on a host whose NPUs belong to "
                "other sessions, and RBLN_VISIBLE_DEVICES= does not stop a tensor "
                "reaching one."
            )
        return func(*args, **kwargs)


RBLN_NODE = "/dev/rbln"


def open_device_nodes() -> set[str]:
    """The RBLN device nodes this process has open."""
    nodes = set()
    for descriptor in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:  # the descriptor closed while we looked
            continue
        if target.startswith(RBLN_NODE):
            nodes.add(target)
    return nodes


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """Hold every test that is not marked `npu` to the host.

    Two ways to reach the device, so two checks. A tensor placed on it is a torch call
    naming it, which `ForbidDevice` refuses as it happens. A compiled graph run through
    the rbln backend reaches it without any such call, and shows up as the process
    opening the device nodes -- which only the test that opens them first can be blamed
    for, since they stay open for the rest of the process. So this names one opener per
    run: mark it, run again, and the next one surfaces.
    """
    if item.get_closest_marker("npu") or on_a_dummy_device():
        yield
        return
    before = open_device_nodes()
    with ForbidDevice():
        outcome = yield
    opened = open_device_nodes() - before
    if opened and outcome.excinfo is None:
        pytest.fail(
            f"opened {len(opened)} RBLN device node(s) "
            f"({', '.join(sorted(opened)[:3])}, ...) without @pytest.mark.npu: its "
            "compiled graphs ran on the device. Mark it, or keep the run on the host.",
            pytrace=False,
        )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    unmet = {
        "npu": (not device.device_nodes(), "no RBLN device nodes on host"),
        "device_mode": (
            device.execution_mode() is not device.ExecutionMode.DEVICE,
            "execution mode is host",
        ),
    }
    for item in items:
        for marker, (missing, reason) in unmet.items():
            if missing and item.get_closest_marker(marker):
                item.add_marker(pytest.mark.xfail(reason=reason, strict=True))


@pytest.fixture
def dummy_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """A device that exists without being hardware, for one test.

    Compiling and placement need a device to name; they do not need the NPUs, which on
    this host belong to other sessions. Never set this for the whole run: a dummy
    device refuses to execute a compiled graph, and its eager outputs are garbage, so
    only placement and structure may be asserted under it.
    """
    monkeypatch.setenv("RBLN_DUMMY_DEVICE", "1")
    monkeypatch.setenv("RBLN_FORCE_NPU_NAME", envs.RBLN_FORCE_NPU_NAME or "RBLN-CA22")


@pytest.fixture(scope="session")
def npu_name() -> str:
    if forced := envs.RBLN_FORCE_NPU_NAME:
        return forced
    if device.device_nodes():
        return device.npu_name()
    raise pytest.UsageError(
        "no RBLN NPU on this host; set RBLN_FORCE_NPU_NAME to the NPU "
        "the compiler should target (as printed by rbln-smi)"
    )


@pytest.fixture
def fresh_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    for attr in ("_current_platform", "_current_platform_selection"):
        monkeypatch.setattr(platforms, attr, None)
    monkeypatch.setattr(platforms, "_backend_init_done", False)


@pytest.fixture
def rbln_override(monkeypatch: pytest.MonkeyPatch, npu_name: str) -> dict[str, str]:
    env = {
        "RBLN_FORCE_NPU_NAME": npu_name,
        "SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "rbln",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return env


@pytest.fixture
def rbln_platform(fresh_platform: None, rbln_override: dict[str, str]):
    platform = platforms.current_platform
    assert platform.device_name == "rbln"
    return platform


@pytest.fixture
def run_probe(rbln_override: dict[str, str]) -> Callable[[str, str], dict]:
    def run(code: str, execution_mode: str) -> dict:
        probe_env = {
            **os.environ,
            **rbln_override,
            "SGLANG_RBLN_EXECUTION_MODE": execution_mode,
            "PYTHONWARNINGS": "ignore",
        }
        completed = subprocess.run(
            [sys.executable, "-c", code],
            env=probe_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        return json.loads(completed.stdout.strip().splitlines()[-1])

    return run


def build_tiny_wan_vae(load_encoder: bool):
    from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
    from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan

    config = WanVAEConfig(
        load_encoder=load_encoder, use_parallel_encode=False, use_parallel_decode=False
    )
    arch = config.arch_config
    arch.base_dim = arch.decoder_base_dim = 16
    arch.z_dim, arch.is_residual, arch.patch_size = 4, True, 2
    arch.in_channels = arch.out_channels = 12
    arch.latents_mean, arch.latents_std = arch.latents_mean[:4], arch.latents_std[:4]
    arch.__post_init__()
    return AutoencoderKLWan(config).eval()


@pytest.fixture
def tiny_wan_vae():
    return build_tiny_wan_vae(load_encoder=False)


@pytest.fixture
def tiny_wan_codec():
    return build_tiny_wan_vae(load_encoder=True)

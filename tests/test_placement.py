# SPDX-License-Identifier: Apache-2.0
import pytest
from torch import nn

from sglang_rbln import envs
from sglang_rbln.compilation import placement
from sglang_rbln.compilation.backend import compile_options


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in envs.environment_variables:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SGLANG_RBLN_DISABLE_COMPILE_CACHE", "1")
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    monkeypatch.setenv(
        "SGLANG_RBLN_COMPONENT_DEVICES", "text_encoder=4,5,6,7;vocoder=8"
    )


def _component() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.Linear(4, 4)))


def _load(name: str, *args, **kwargs):
    module = object() if name == "tokenizer" else _component()
    return module, 1.0


def _placement(module: nn.Module) -> tuple[tuple[int, ...] | None, int]:
    options = compile_options(module)
    devices = options.get("device")
    return devices and tuple(devices), options["num_devices"]


@pytest.mark.parametrize(
    ("component_name", "expected"),
    [
        ("text_encoder", ((4, 5, 6, 7), 4)),
        ("vocoder", ((8,), 1)),
        ("transformer", (None, 4)),
    ],
)
def test_component_and_its_submodules_compile_for_their_devices(
    component_name, expected
):
    component, memory_usage = placement.place_component(_load, component_name, "path")
    assert memory_usage == 1.0
    assert {_placement(module) for module in component.modules()} == {expected}


def test_components_that_are_not_modules_pass_through():
    tokenizer, _ = placement.place_component(_load, "tokenizer", "path")
    assert not isinstance(tokenizer, nn.Module)


def test_registers_on_the_component_loader(monkeypatch):
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    calls = []
    monkeypatch.setattr(
        HookRegistry,
        "register",
        classmethod(lambda cls, *args: calls.append(args)),
    )
    target = placement.register()
    assert calls == [(target, placement.place_component, HookType.AROUND)]
    assert target.endswith("PipelineComponentLoader.load_component")

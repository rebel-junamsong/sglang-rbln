# SPDX-License-Identifier: Apache-2.0
import pytest
from torch import nn

from sglang_rbln import device, envs
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


GiB = 1 << 30


@pytest.fixture
def npus(monkeypatch):
    """Visible NPUs with `free` GiB each; the encoder weighs 8 GiB."""
    monkeypatch.delenv("SGLANG_RBLN_COMPONENT_DEVICES")
    monkeypatch.setattr(placement, "weight_bytes", lambda module: 8 * GiB)

    def visible(count: int, free: float = 15.0) -> None:
        monkeypatch.setenv("RBLN_VISIBLE_DEVICES", ",".join(map(str, range(count))))
        info = device.MemoryInfo(16 * GiB, 0, int(free * GiB))
        monkeypatch.setattr(device, "npu_memory_info", lambda index: info)

    return visible


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (8, ((4, 5, 6, 7), 4)),
        (12, ((4, 5, 6, 7), 4)),
        (6, (None, 4)),
        (4, (None, 4)),
    ],
    ids=["eight", "twelve", "six", "dit-only"],
)
@pytest.mark.parametrize("component_name", ["text_encoder", "connectors", "vocoder"])
def test_text_encoder_takes_the_npus_beyond_the_dit(
    npus, count, expected, component_name
):
    npus(count)
    encoder, _ = placement.place_component(_load, component_name, "path")
    assert {_placement(module) for module in encoder.modules()} == {expected}
    assert placement.component_devices(component_name) == expected[0]


def test_text_encoder_stays_with_the_dit_when_spare_npus_cannot_hold_it(npus):
    npus(8, free=3.0)  # 2 GiB of weights per NPU plus the reserve
    encoder, _ = placement.place_component(_load, "text_encoder", "path")
    assert {_placement(module) for module in encoder.modules()} == {(None, 4)}


def test_a_compile_only_run_places_on_the_spare_npus_without_asking_them(
    npus, monkeypatch
):
    """It has no NPU to ask for free memory: the dummy device reports none."""
    npus(8)
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_ONLY", "1")
    monkeypatch.setattr(device, "npu_memory_info", lambda index: None)
    encoder, _ = placement.place_component(_load, "vocoder", "path")
    assert {_placement(module) for module in encoder.modules()} == {((4, 5, 6, 7), 4)}


def test_explicit_placement_wins_over_spare_npus(npus, monkeypatch):
    npus(8)
    monkeypatch.setenv("SGLANG_RBLN_COMPONENT_DEVICES", "text_encoder=6,7")
    encoder, _ = placement.place_component(_load, "text_encoder", "path")
    assert {_placement(module) for module in encoder.modules()} == {((6, 7), 2)}


def test_only_text_encoders_move_to_spare_npus(npus):
    npus(8)
    vae, _ = placement.place_component(_load, "vae", "path")
    assert {_placement(module) for module in vae.modules()} == {(None, 4)}

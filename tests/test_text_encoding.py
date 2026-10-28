# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from torch import nn

from sglang_rbln.patches.stages import text_encoding

pytestmark = pytest.mark.usefixtures("rbln_platform")


class _Layer(nn.Linear):
    pass


def _encoder(*layer_names: str) -> nn.Module:
    encoder = nn.Module()
    encoder.language_model = nn.Module()
    encoder.language_model.layers = nn.ModuleList([_Layer(4, 4), _Layer(4, 4)])
    encoder.language_model.embed_tokens = nn.Embedding(8, 4)
    encoder.blocks = nn.ModuleList([_Layer(4, 4)])
    encoder.layer_names = list(layer_names)
    return encoder


@pytest.fixture
def compiled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs))
    )
    return calls


def forward(stage, batch, server_args):
    return "encoded"


def test_registers_both_text_stages():
    targets = text_encoding.register()
    assert targets == (text_encoding.HOOK_TARGET, text_encoding.CONNECTOR_HOOK_TARGET)
    assert [
        hook for target in targets for _, hook, _ in HookRegistry._hooks[target]
    ] == [text_encoding.compile_text_encoders, text_encoding.compile_connectors]


@pytest.mark.parametrize(
    ("layer_names", "expected"),
    [
        (("language_model.layers",), 2),
        (("language_model.layers", "blocks"), 3),
        ((), 0),
    ],
)
def test_compiles_each_declared_layer_once(compiled, layer_names, expected):
    stage = SimpleNamespace(text_encoders=[_encoder(*layer_names), None])
    server_args = SimpleNamespace(
        enable_torch_compile=True, text_encoder_cpu_offload=False
    )
    outputs = [
        text_encoding.compile_text_encoders(forward, stage, None, server_args)
        for _ in range(2)
    ]
    assert outputs == ["encoded", "encoded"]
    assert [type(module) for module, _ in compiled] == [_Layer] * expected
    assert all(kwargs["backend"] == "rbln" for _, kwargs in compiled)
    assert all(kwargs["dynamic"] is False for _, kwargs in compiled)


def test_layers_of_one_module_share_its_compile_context(compiled):
    stage = SimpleNamespace(connectors=_encoder("language_model.layers", "blocks"))
    server_args = SimpleNamespace(
        enable_torch_compile=True, text_encoder_cpu_offload=False
    )
    text_encoding.compile_connectors(forward, stage, None, server_args)
    contexts = {id(kwargs["options"]["compile_context"]) for _, kwargs in compiled}
    assert len(compiled) == 3
    assert len(contexts) == 1


def test_respects_disabled_compile(compiled):
    stage = SimpleNamespace(
        text_encoders=[_encoder("language_model.layers")],
        connectors=_encoder("blocks"),
    )
    server_args = SimpleNamespace(
        enable_torch_compile=False, text_encoder_cpu_offload=False
    )
    text_encoding.compile_text_encoders(forward, stage, None, server_args)
    text_encoding.compile_connectors(forward, stage, None, server_args)
    assert compiled == []


def test_text_encoder_offloaded_to_cpu_stays_eager(compiled):
    stage = SimpleNamespace(
        text_encoders=[_encoder("language_model.layers")],
        connectors=_encoder("blocks"),
    )
    server_args = SimpleNamespace(
        enable_torch_compile=True, text_encoder_cpu_offload=True
    )
    text_encoding.compile_text_encoders(forward, stage, None, server_args)
    text_encoding.compile_connectors(forward, stage, None, server_args)
    assert [module for module, _ in compiled] == list(stage.connectors.blocks)

# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
from sglang.multimodal_gen.runtime.utils import torch_compile
from torch import nn

from sglang_rbln.compilation.regions import BlockGroup
from sglang_rbln.patches.stages import text_encoding

pytestmark = pytest.mark.usefixtures("rbln_platform")


class _Layer(nn.Linear):
    pass


def _text_encoder(*layer_names: str) -> nn.Module:
    encoder = nn.Module()
    encoder.language_model = nn.Module()
    encoder.language_model.layers = nn.ModuleList([_Layer(4, 4), _Layer(4, 4)])
    encoder.language_model.embed_tokens = nn.Embedding(8, 4)
    encoder.blocks = nn.ModuleList([_Layer(4, 4)])
    encoder.video_aggregate_embed = _Layer(4, 4)
    encoder.text_proj_in = None
    encoder.layer_names = list(layer_names)
    return encoder


@pytest.fixture
def compiled_modules(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs))
    )
    return calls


def upstream_forward(stage, batch, server_args):
    return "encoded"


def test_registers_both_text_stages_and_the_encoder_call():
    targets = text_encoding.register()
    assert targets == (
        text_encoding.HOOK_TARGET,
        text_encoding.CONNECTOR_HOOK_TARGET,
        text_encoding.FORWARD_ENCODER_TARGET,
    )
    assert [
        hook for target in targets for _, hook, _ in HookRegistry._hooks[target]
    ] == [
        text_encoding.compile_text_encoders,
        text_encoding.compile_connectors,
        text_encoding.encode_from_host_inputs,
    ]


@pytest.mark.parametrize(
    ("layer_names", "per_region", "expected"),
    [
        (("language_model.layers",), None, [BlockGroup]),
        (("language_model.layers", "blocks"), None, [BlockGroup, _Layer]),
        (("language_model.layers", "blocks"), 1, [_Layer] * 3),
        ((), None, []),
    ],
)
def test_compiles_each_declared_layer_list_once_as_one_region(
    monkeypatch, compiled_modules, layer_names, per_region, expected
):
    if per_region:
        monkeypatch.setenv("SGLANG_RBLN_BLOCKS_PER_REGION", str(per_region))
    else:
        monkeypatch.delenv("SGLANG_RBLN_BLOCKS_PER_REGION", raising=False)
    stage = SimpleNamespace(text_encoders=[_text_encoder(*layer_names), None])
    server_args = SimpleNamespace(
        enable_torch_compile=True, text_encoder_cpu_offload=False
    )
    outputs = [
        text_encoding.compile_text_encoders(upstream_forward, stage, None, server_args)
        for _ in range(2)
    ]
    assert outputs == ["encoded", "encoded"]
    assert [type(module) for module, _ in compiled_modules] == expected
    assert all(kwargs["backend"] == "rbln" for _, kwargs in compiled_modules)
    assert all(kwargs["dynamic"] is False for _, kwargs in compiled_modules)


def test_layers_of_one_module_share_its_compile_context(compiled_modules):
    stage = SimpleNamespace(
        text_encoders=[_text_encoder("language_model.layers", "blocks")]
    )
    server_args = SimpleNamespace(
        enable_torch_compile=True, text_encoder_cpu_offload=False
    )
    text_encoding.compile_text_encoders(upstream_forward, stage, None, server_args)
    contexts = {
        id(kwargs["options"]["compile_context"]) for _, kwargs in compiled_modules
    }
    (group, _), *rest = compiled_modules
    encoder = stage.text_encoders[0]
    assert [block.weight for block in group.blocks] == [
        layer.weight for layer in encoder.language_model.layers
    ]
    assert [module for module, _ in rest] == [*encoder.blocks]
    assert len(contexts) == 1


def test_the_connectors_compile_only_their_projections(compiled_modules):
    # their transformers compile whole when they run (ltx_2_text_features)
    stage = SimpleNamespace(connectors=_text_encoder("language_model.layers", "blocks"))
    server_args = SimpleNamespace(enable_torch_compile=True)
    for _ in range(2):
        text_encoding.compile_connectors(upstream_forward, stage, None, server_args)
    assert [module for module, _ in compiled_modules] == [
        stage.connectors.video_aggregate_embed
    ]


def test_nothing_compiles_without_torch_compile(compiled_modules):
    stage = SimpleNamespace(
        text_encoders=[_text_encoder("language_model.layers")],
        connectors=_text_encoder("blocks"),
    )
    server_args = SimpleNamespace(
        enable_torch_compile=False, text_encoder_cpu_offload=False
    )
    text_encoding.compile_text_encoders(upstream_forward, stage, None, server_args)
    text_encoding.compile_connectors(upstream_forward, stage, None, server_args)
    assert compiled_modules == []


def test_text_encoder_offloaded_to_cpu_stays_eager(compiled_modules):
    stage = SimpleNamespace(
        text_encoders=[_text_encoder("language_model.layers")],
        connectors=_text_encoder("blocks"),
    )
    server_args = SimpleNamespace(
        enable_torch_compile=True, text_encoder_cpu_offload=True
    )
    text_encoding.compile_text_encoders(upstream_forward, stage, None, server_args)
    text_encoding.compile_connectors(upstream_forward, stage, None, server_args)
    assert [module for module, _ in compiled_modules] == [
        stage.connectors.video_aggregate_embed
    ]


def test_projections_trace_into_a_graph_that_owns_their_weights(monkeypatch):
    graph_inputs = []

    def backend(graph, example_inputs):
        graph_inputs.extend(tuple(value.shape) for value in example_inputs)
        return graph

    monkeypatch.setattr(
        torch_compile,
        "build_torch_compile_kwargs",
        lambda **kwargs: {"backend": backend},
    )
    connectors = nn.Module()
    connectors.video_aggregate_embed = nn.Linear(8, 4)
    projection = connectors.video_aggregate_embed
    hidden_states = torch.randn(2, 8)
    expected = projection(hidden_states)
    assert text_encoding.compile_projections(connectors, ["video_aggregate_embed"]) == 1
    torch.testing.assert_close(projection(hidden_states), expected)
    assert isinstance(projection, nn.Linear)
    assert graph_inputs == [(2, 8)]


PLAIN_WORDS = 124  # the tokenizer's 128-word vocabulary less its 4 special tokens


def _tiny_gemma():
    from transformers.models.gemma4_unified import (
        configuration_gemma4_unified,
        modeling_gemma4_unified,
    )

    config = configuration_gemma4_unified.Gemma4UnifiedTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        num_global_key_value_heads=1,
        vocab_size=128,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=16,
        attention_k_eq_v=True,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=128,
    )
    config._attn_implementation = "sdpa"
    model = modeling_gemma4_unified.Gemma4UnifiedTextModel(config).eval()
    model.layer_names = ["layers"]
    return model


@pytest.mark.parametrize("declared", [True, False], ids=["layer_names", "no_split"])
def test_capture_hooks_stay_out_of_compiled_layers(monkeypatch, declared):
    from torch._dynamo.utils import counters

    graphs = []

    def backend(graph, example_inputs):
        graphs.append(graph)
        return graph

    monkeypatch.setattr(
        torch_compile,
        "build_torch_compile_kwargs",
        lambda **kwargs: {"backend": backend},
    )
    torch.manual_seed(0)
    model = _tiny_gemma()
    if not declared:
        del model.layer_names
    input_ids = torch.randint(0, 128, (1, 32))
    with torch.no_grad():
        expected = model(input_ids=input_ids, output_hidden_states=True).hidden_states
        torch._dynamo.reset()
        counters.clear()
        assert text_encoding.compile_regions(model) == 1
        runs = [
            model(input_ids=input_ids, output_hidden_states=True).hidden_states
            for _ in range(2)
        ]
    assert sum(counters["graph_break"].values()) == 0
    assert len(graphs) == 1
    for actual in runs:
        assert len(actual) == len(expected) == len(model.layers) + 1
        for output, reference in zip(actual, expected, strict=True):
            assert torch.equal(output, reference)


def _word_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = ["<pad>", "<bos>", "<eos>", "<unk>", *(f"w{i}" for i in range(PLAIN_WORDS))]
    backend = Tokenizer(
        models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


def _prompt_of(words: int) -> str:
    return " ".join(f"w{i % PLAIN_WORDS}" for i in range(words))


@pytest.mark.parametrize("caption_proj_before_connector", [False, True])
def test_encoder_regions_match_eager_at_one_shape_for_any_prompt_length(
    monkeypatch, caption_proj_before_connector
):
    from sglang.multimodal_gen.configs.pipeline_configs import ltx_2

    graphs = []

    def backend(graph, example_inputs):
        graphs.append(tuple(tuple(value.shape) for value in example_inputs))
        return graph

    monkeypatch.setattr(
        torch_compile,
        "build_torch_compile_kwargs",
        lambda **kwargs: {"backend": backend},
    )
    config = ltx_2.LTX2PipelineConfig()
    monkeypatch.setattr(
        config.dit_config.arch_config,
        "caption_proj_before_connector",
        caption_proj_before_connector,
    )
    tokenizer = _word_tokenizer()
    torch.manual_seed(0)
    model = _tiny_gemma().to(torch.bfloat16)

    def encode(prompt: str, max_length: int) -> torch.Tensor:
        text_inputs = config.tokenize_prompt(
            [prompt], tokenizer, {"max_length": max_length}
        )
        outputs = model(
            input_ids=text_inputs["input_ids"],
            attention_mask=config.get_text_encoder_attention_mask(text_inputs, 0),
            output_hidden_states=True,
        )
        return ltx_2._gemma_postprocess_func(outputs, text_inputs, config)

    requests = [(_prompt_of(words), 32) for words in (0, 1, 7, 31, 40)]
    requests.append((_prompt_of(11), 48))
    with torch.no_grad():
        expected = [encode(*request) for request in requests]
        torch._dynamo.reset()
        text_encoding.compile_regions(model)
        actual = [encode(*request) for request in requests]
    for output, reference in zip(actual, expected, strict=True):
        assert torch.equal(output, reference)
    assert len(graphs) == 3
    assert {shape for graph in graphs for shape in graph} >= {(1, 32, 64), (1, 48, 64)}


class EagerDeviceOps(torch.overrides.TorchFunctionMode):
    """The torch calls made eagerly on rbln tensors, by name."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if any(
            isinstance(leaf, torch.Tensor) and leaf.device.type == "rbln"
            for leaf in (*args, *kwargs.values())
        ):
            self.names.append(getattr(func, "__name__", str(func)))
        return func(*args, **kwargs)


class _PreludeEncoder(nn.Module):
    """An encoder's eager prelude: the token embedding lookup in a host table and
    the attention-mask arithmetic, before any region."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(1024, 64)
        self.seen: list[tuple[str, str]] = []

    def forward(self, input_ids, attention_mask, output_hidden_states=True):
        self.seen.append((input_ids.device.type, attention_mask.device.type))
        embeds = self.embed_tokens(input_ids)
        additive = (1 - attention_mask[:, None, None, :].to(embeds.dtype)) * -1e4
        return embeds, additive


def test_the_encoder_prelude_runs_on_host_inputs(dummy_device):
    """Upstream moves the token ids and mask to the stage's output device, the
    connectors' NPUs; the encoder still gets them on the host, so its eager prelude
    runs there instead of op by op on the device."""
    from sglang_rbln.compilation import residency

    def forward_encoder(stage, text_encoder, forward_kwargs):
        return text_encoder(**forward_kwargs)

    encoder = _PreludeEncoder()
    forward_kwargs = {
        "input_ids": torch.randint(0, 1024, (1, 128)).to("rbln:0"),
        "attention_mask": torch.ones(1, 128, dtype=torch.long).to("rbln:0"),
        "output_hidden_states": True,
    }
    eager = EagerDeviceOps()
    with residency.DeviceBoundary(), eager:
        embeds, additive = text_encoding.encode_from_host_inputs(
            forward_encoder, None, encoder, forward_kwargs
        )
    assert encoder.seen == [("cpu", "cpu")]
    assert (embeds.device.type, additive.device.type) == ("cpu", "cpu")
    assert eager.names == ["cpu", "cpu"]  # only taking the two inputs down

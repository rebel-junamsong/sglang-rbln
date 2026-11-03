# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
import transformers
from fastapi import HTTPException

from sglang_rbln.patches.entrypoints import prompt_enhancement


@pytest.fixture(scope="module")
def tiny_enhancer() -> transformers.PreTrainedModel:
    """Gemma-4 text: sliding and full layers, KV-shared tail, logit softcap."""
    config = transformers.Gemma4TextConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        global_head_dim=32,
        sliding_window=8,
        num_kv_shared_layers=2,
        layer_types=["sliding_attention", "full_attention"] * 3,
        hidden_size_per_layer_input=16,
        vocab_size_per_layer_input=512,
        final_logit_softcapping=30.0,
        max_position_embeddings=512,
    )
    torch.manual_seed(0)
    model = transformers.Gemma4ForCausalLM(config).eval()
    model.generation_config.eos_token_id = 511
    model.generation_config.pad_token_id = 0
    return model


@pytest.mark.parametrize("no_repeat_ngram", [0, 3])
@pytest.mark.parametrize("prompt_tokens", [20, 128, 130])
def test_static_cache_decode_matches_generate(
    tiny_enhancer, prompt_tokens, no_repeat_ngram
):
    # past the 8-token sliding window, around the 128-token prefill bucket edge
    input_ids = torch.randint(
        1, 500, (1, prompt_tokens), generator=torch.Generator().manual_seed(1)
    )
    expected = tiny_enhancer.generate(
        input_ids,
        do_sample=False,
        max_new_tokens=40,
        no_repeat_ngram_size=no_repeat_ngram,
    )[0, prompt_tokens:].tolist()

    tokens = prompt_enhancement.greedy_decode(
        tiny_enhancer, input_ids, max_new_tokens=40, no_repeat_ngram=no_repeat_ngram
    )

    assert tokens == expected


def test_kv_shared_layers_get_no_cache_of_their_own(tiny_enhancer):
    cache = prompt_enhancement.window_cache(tiny_enhancer, window=256)
    assert len(cache.layers) == 4
    assert {layer.max_cache_len for layer in cache.layers} == {256}


def test_the_whole_cache_is_allocated_outside_any_graph(tiny_enhancer):
    # the graphs take the cache as an input they write back, not as something the
    # prefill graph creates
    cache = prompt_enhancement.window_cache(tiny_enhancer, window=256)
    assert all(layer.is_initialized for layer in cache.layers)
    assert {tuple(layer.keys.shape) for layer in cache.layers} == {(1, 1, 256, 32)}
    assert {layer.keys.dtype for layer in cache.layers} == {torch.float32}


@pytest.fixture
def on_device(monkeypatch):
    """Judge the host tensors of a test the way the plugin judges device tensors."""
    monkeypatch.setattr(prompt_enhancement, "on_device", lambda tensor: True)


def rows(layer, count: int) -> None:
    for step in range(count):
        row = torch.full((1, 1, 1, 4), float(step + 1))
        layer.update(row, row * 10)


def test_a_single_row_write_lands_where_index_copy_puts_it(on_device):
    masked = prompt_enhancement.MaskedWriteLayer(max_cache_len=8)
    plain = prompt_enhancement.StaticLayer(max_cache_len=8)

    rows(masked, 3)
    with pytest.MonkeyPatch.context() as off:  # the plain layer keeps index_copy_
        off.setattr(prompt_enhancement, "on_device", lambda tensor: False)
        rows(plain, 3)

    torch.testing.assert_close(masked.keys, plain.keys, rtol=0, atol=0)
    torch.testing.assert_close(masked.values, plain.values, rtol=0, atol=0)
    assert int(masked.cumulative_length) == int(plain.cumulative_length) == 3


def test_a_single_row_write_uses_no_index_copy(on_device, monkeypatch):
    layer = prompt_enhancement.MaskedWriteLayer(max_cache_len=8)
    rows(layer, 1)  # allocate before the ban, where lazy_initialization may index
    monkeypatch.setattr(
        torch.Tensor,
        "index_copy_",
        lambda *args, **kwargs: pytest.fail("the decode step wrote with index_copy_"),
    )

    rows(layer, 1)

    assert layer.keys[0, 0, 1, 0] == 1  # the second row holds the second write


def test_a_prefill_keeps_the_index_copy_path(on_device, monkeypatch):
    layer = prompt_enhancement.MaskedWriteLayer(max_cache_len=8)
    block = torch.ones((1, 1, 4, 4))
    layer.update(block, block)  # allocate before the ban
    monkeypatch.setattr(
        torch,
        "where",
        lambda *args, **kwargs: pytest.fail("a multi-row write took the masked path"),
    )

    layer.update(block * 2, block * 2)

    assert int(layer.cumulative_length) == 8


def test_the_whole_graph_hands_the_model_an_integer_mask():
    seen = {}

    class Decoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask, **kwargs):
            seen["dtype"] = attention_mask.dtype
            return attention_mask.sum()

    step = prompt_enhancement.WholeGraph(Decoder())
    mask = torch.ones((1, 4), dtype=prompt_enhancement.WholeGraph.MASK_DTYPE)

    assert (
        step(input_ids=torch.zeros((1, 4), dtype=torch.long), attention_mask=mask) == 4
    )
    assert seen["dtype"] == torch.long
    assert prompt_enhancement.mask_dtype(step) is torch.bfloat16
    assert prompt_enhancement.mask_dtype(Decoder()) is torch.long


def test_whole_graph_decode_matches_generate(tiny_enhancer):
    # the boundary mask is bf16, so the step must cast it back before the model reads it
    input_ids = torch.randint(
        1, 500, (1, 20), generator=torch.Generator().manual_seed(1)
    )
    expected = tiny_enhancer.generate(input_ids, do_sample=False, max_new_tokens=16)[
        0, 20:
    ].tolist()

    tokens = prompt_enhancement.greedy_decode(
        tiny_enhancer,
        input_ids,
        max_new_tokens=16,
        step=prompt_enhancement.WholeGraph(tiny_enhancer.get_decoder()),
    )

    assert tokens == expected


@pytest.fixture
def served(monkeypatch, tmp_path):
    """A server whose model directory ships a prompt_enhancer/ component."""
    (tmp_path / "prompt_enhancer").mkdir()
    (tmp_path / "prompt_enhancer" / "config.json").write_text("{}")
    server_args = SimpleNamespace(model_path=str(tmp_path), enable_torch_compile=True)
    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
        lambda: server_args,
    )
    calls = []
    monkeypatch.setattr(
        prompt_enhancement,
        "enhance",
        lambda *args: calls.append(args) or f"rewritten: {args[2]}",
    )
    monkeypatch.setenv("SGLANG_RBLN_PROMPT_ENHANCER", "1")
    return SimpleNamespace(path=str(tmp_path), calls=calls)


def request(external=None):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(prompt_enhancer=external))
    )


async def upstream(request, prompt, *, enabled, task, image_paths=None):
    return f"upstream: {prompt}" if enabled else prompt


def run(req, **kwargs):
    kwargs = {"enabled": True, "task": "video"} | kwargs
    return asyncio.run(
        prompt_enhancement.enhance_with_model_enhancer(upstream, req, "a fox", **kwargs)
    )


def test_rewrites_with_the_models_enhancer(served):
    assert run(request()) == "rewritten: a fox"
    assert served.calls == [(served.path, True, "a fox", "video")]


@pytest.mark.parametrize(
    ("case", "kwargs"),
    [
        ("not requested", {"enabled": False}),
        ("external endpoint configured", {}),
        ("opt-in off", {}),
        ("no prompt_enhancer component", {}),
    ],
)
def test_leaves_the_request_to_upstream(served, monkeypatch, tmp_path, case, kwargs):
    req = request()
    if case == "external endpoint configured":
        req = request(external=object())
    elif case == "opt-in off":
        monkeypatch.delenv("SGLANG_RBLN_PROMPT_ENHANCER")
    elif case == "no prompt_enhancer component":
        (tmp_path / "prompt_enhancer" / "config.json").unlink()
    expected = "a fox" if case == "not requested" else "upstream: a fox"
    assert run(req, **kwargs) == expected
    assert served.calls == []


def test_refuses_image_prompts(served):
    with pytest.raises(HTTPException) as error:
        run(request(), image_paths=["frame.png"])
    assert error.value.status_code == 400
    assert served.calls == []


def test_user_turn_defaults_to_upstreams_format(monkeypatch):
    monkeypatch.delenv("SGLANG_RBLN_PROMPT_ENHANCER_USER_TEMPLATE", raising=False)
    enhancer = object.__new__(prompt_enhancement.ModelPromptEnhancer)
    assert json.loads(enhancer.user_turn("a fox", "video")) == {
        "task": "video",
        "prompt": "a fox",
    }
    monkeypatch.setenv(
        "SGLANG_RBLN_PROMPT_ENHANCER_USER_TEMPLATE", "user prompt: {prompt}"
    )
    assert enhancer.user_turn("a fox", "video") == "user prompt: a fox"

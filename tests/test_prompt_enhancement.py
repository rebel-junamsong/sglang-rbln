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

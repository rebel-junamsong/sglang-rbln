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


def test_the_cache_write_compiles_to_one_device_function(tmp_path, dummy_device):
    """The step is one graph only while the cache write has a device form.

    `index_copy_` at a tensor position had none, for a decode row or a prefill chunk,
    so the compiler left it and the ops around it on the host and split the step into
    one graph per layer. rebel_compiler 4aea7ee411 lowers it; this is the contract the
    plugin relies on, and it fails here rather than as a slow step on a device.
    """
    import rebel

    window, heads, dim = 256, 1, 64
    layer = prompt_enhancement.StaticLayer(max_cache_len=window)
    layer.lazy_initialization(
        *(torch.zeros((1, heads, 1, dim), dtype=torch.bfloat16, device="rbln"),) * 2
    )

    class Write(torch.nn.Module):
        def forward(self, rows):
            keys, _ = layer.update(rows, rows)
            return keys.sum()

    for written in (1, 128):
        layer.cumulative_length.fill_(0)
        cache_dir = tmp_path / f"rows{written}"
        compiled = torch.compile(
            Write(),
            backend="rbln",
            dynamic=False,
            options={"mode": ["strict", "compile_only"], "cache_dir": str(cache_dir)},
        )
        with torch.inference_mode():
            compiled(
                torch.ones(
                    (1, heads, written, dim), dtype=torch.bfloat16, device="rbln"
                )
            )
        stats = [
            stat
            for artifact in sorted(cache_dir.glob("*.rbln"))
            for stat in rebel.RBLNCompiledModel.inspect(str(artifact))["subgraph"][
                "stats"
            ]
        ]
        assert [stat["device_func_count"] for stat in stats] == [1], written
        host_ops = {
            op
            for stat in stats
            for ops in stat["host_op_reason"].values()
            for op in ops
            if not op.startswith("args_")
        }
        assert host_ops == set(), (written, host_ops)


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


def recorded_shapes(model, prompt_tokens: int, monkeypatch) -> set:
    """The shapes the step is called with for a prompt of `prompt_tokens`."""
    shapes = set()
    decoder = model.get_decoder()

    def record(*args, **kwargs):
        shapes.add(
            (tuple(kwargs["input_ids"].shape), tuple(kwargs["attention_mask"].shape))
        )
        return decoder(*args, **kwargs)

    input_ids = torch.randint(
        1, 500, (1, prompt_tokens), generator=torch.Generator().manual_seed(2)
    )
    prompt_enhancement.greedy_decode(model, input_ids, max_new_tokens=8, step=record)
    return shapes


def test_the_graphs_do_not_follow_the_prompt_length(tiny_enhancer, monkeypatch):
    # a server that compiled a graph per prompt length would pay the compile again for
    # every new length it sees
    monkeypatch.setenv("SGLANG_RBLN_PROMPT_ENHANCER_MAX_SEQ_LEN", "512")

    short = recorded_shapes(tiny_enhancer, 20, monkeypatch)
    long = recorded_shapes(tiny_enhancer, 300, monkeypatch)

    assert short == long == {((1, 128), (1, 512)), ((1, 1), (1, 512))}


def test_a_prompt_past_the_configured_length_still_runs(tiny_enhancer, monkeypatch):
    monkeypatch.setenv("SGLANG_RBLN_PROMPT_ENHANCER_MAX_SEQ_LEN", "128")

    shapes = recorded_shapes(tiny_enhancer, 200, monkeypatch)

    assert shapes == {((1, 128), (1, 256)), ((1, 1), (1, 256))}


def test_the_configured_length_is_what_the_cache_is_compiled_for(monkeypatch):
    monkeypatch.setenv("SGLANG_RBLN_PROMPT_ENHANCER_MAX_SEQ_LEN", "1024")
    assert prompt_enhancement.cache_window(900) == 1024
    assert prompt_enhancement.cache_window(1024) == 1024
    assert (
        prompt_enhancement.cache_window(1025) == 1152
    )  # its own graphs, chunk-aligned


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

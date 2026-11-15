# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import json
import threading
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
def test_greedy_decode_matches_generate(tiny_enhancer, prompt_tokens, no_repeat_ngram):
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


class Bf16Head(torch.nn.Linear):
    """An output head that computes its logits in bf16."""

    def forward(self, hidden_states):
        return super().forward(hidden_states.to(torch.bfloat16))


@pytest.mark.parametrize("prompt_tokens", [5, 20])
def test_tied_logits_pick_the_token_generate_picks(tiny_enhancer, prompt_tokens):
    model = copy.deepcopy(tiny_enhancer)
    direction = torch.nn.functional.normalize(
        torch.randn(
            model.config.hidden_size, generator=torch.Generator().manual_seed(3)
        ),
        dim=0,
    )
    gains = torch.logspace(1, 4, model.config.vocab_size // 2)
    head = Bf16Head(model.config.hidden_size, model.config.vocab_size, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.cat([gains, -gains])[:, None] * direction)
    model.lm_head = head.to(torch.bfloat16)
    input_ids = torch.randint(
        1, 500, (1, prompt_tokens), generator=torch.Generator().manual_seed(2)
    )
    expected = model.generate(input_ids, do_sample=False, max_new_tokens=4)[
        0, prompt_tokens:
    ].tolist()

    tokens = prompt_enhancement.greedy_decode(model, input_ids, max_new_tokens=4)

    assert tokens == expected


@pytest.mark.parametrize("prompt_tokens", [20, 128, 130])
def test_step_logits_match_one_shot_forward(tiny_enhancer, prompt_tokens):
    softcap = tiny_enhancer.config.get_text_config().final_logit_softcapping
    head = tiny_enhancer.get_output_embeddings()
    steps = []
    record = head.register_forward_hook(
        lambda module, args, out: steps.append(out[0, -1].float())
    )
    input_ids = torch.randint(
        1, 500, (1, prompt_tokens), generator=torch.Generator().manual_seed(3)
    )
    try:
        tokens = prompt_enhancement.greedy_decode(
            tiny_enhancer, input_ids, max_new_tokens=8
        )
    finally:
        record.remove()
    sequence = torch.cat([input_ids[0], torch.tensor(tokens)])
    with torch.inference_mode():
        for step, logits in enumerate(steps[: len(tokens)]):
            prefix = sequence[: prompt_tokens + step][None]
            expected = tiny_enhancer(input_ids=prefix).logits[0, -1].float()
            capped = torch.tanh(logits / softcap) * softcap
            torch.testing.assert_close(capped, expected, rtol=0, atol=1e-4)


def test_kv_shared_layers_get_no_cache_of_their_own(tiny_enhancer):
    cache = prompt_enhancement.window_cache(tiny_enhancer, window=256)
    assert len(cache.layers) == 4  # 6 layers, the last 2 sharing earlier ones' KV
    assert {layer.max_cache_len for layer in cache.layers} == {256}


def test_the_whole_cache_is_allocated_outside_any_graph(tiny_enhancer):
    # the graphs take the cache as an input they write back, not as something the
    # prefill graph creates
    cache = prompt_enhancement.window_cache(tiny_enhancer, window=256)
    assert all(layer.is_initialized for layer in cache.layers)
    assert {tuple(layer.keys.shape) for layer in cache.layers} == {(1, 1, 256, 32)}
    assert {layer.keys.dtype for layer in cache.layers} == {torch.float32}


def test_the_cache_write_compiles_into_the_step_graph(tmp_path, dummy_device):
    """The cache write compiles into the step's graph, for a decode row and for a
    prefill chunk. The plugin relies on this, and the test fails here rather than as
    a slow step on a device.
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


def step_input_shapes(model, prompt_tokens: int) -> set:
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
    monkeypatch.setattr(prompt_enhancement, "CACHE_WINDOW", 512)

    short = step_input_shapes(tiny_enhancer, 20)
    long = step_input_shapes(tiny_enhancer, 300)

    assert short == long == {((1, 128), (1, 512)), ((1, 1), (1, 512))}


def test_a_prompt_past_the_cache_window_still_runs(tiny_enhancer, monkeypatch):
    monkeypatch.setattr(prompt_enhancement, "CACHE_WINDOW", 128)

    shapes = step_input_shapes(tiny_enhancer, 200)

    assert shapes == {((1, 128), (1, 256)), ((1, 1), (1, 256))}


def test_the_cache_window_grows_by_whole_chunks_past_its_default(monkeypatch):
    monkeypatch.setattr(prompt_enhancement, "CACHE_WINDOW", 1024)
    assert prompt_enhancement.cache_window(900) == 1024
    assert prompt_enhancement.cache_window(1024) == 1024
    assert prompt_enhancement.cache_window(1025) == 1152


def bare_enhancer(**attributes):
    """A ModelPromptEnhancer without its checkpoint, for the settings around it."""
    enhancer = object.__new__(prompt_enhancement.ModelPromptEnhancer)
    for name, value in attributes.items():
        setattr(enhancer, name, value)
    return enhancer


def test_the_system_prompt_comes_from_a_file_when_one_is_named(monkeypatch, tmp_path):
    monkeypatch.delenv("SGLANG_RBLN_PROMPT_ENHANCER_SYSTEM_PROMPT", raising=False)
    upstream_default = prompt_enhancement.ModelPromptEnhancer._system_prompt()
    recipe = tmp_path / "system.txt"
    recipe.write_text("Rewrite it the way this model was trained.\n")

    monkeypatch.setenv("SGLANG_RBLN_PROMPT_ENHANCER_SYSTEM_PROMPT", str(recipe))

    assert prompt_enhancement.ModelPromptEnhancer._system_prompt() == recipe.read_text()
    assert upstream_default and upstream_default != recipe.read_text()


def test_decoding_follows_ltx_cores_recipe(monkeypatch):
    seen = []
    monkeypatch.setattr(
        prompt_enhancement,
        "greedy_decode",
        lambda model, ids, **kwargs: seen.append(kwargs) or [7],
    )
    enhancer = bare_enhancer(
        model=None,
        step=None,
        system_prompt="",
        lock=threading.Lock(),
        tokenizer=SimpleNamespace(
            apply_chat_template=lambda *a, **k: {
                "input_ids": torch.zeros((1, 3), dtype=torch.long)
            },
            decode=lambda tokens, **k: "a rewrite",
        ),
    )

    enhancer.enhance("a fox", "video")

    assert seen == [{"max_new_tokens": 600, "no_repeat_ngram": 5, "step": None}]


def test_the_decode_step_compiles_as_one_graph(tmp_path, dummy_device, tiny_enhancer):
    """It checks the compiled step only: work outside the traced region, or a graph
    break in ``greedy_decode``, is not covered here.
    """
    import rebel

    window = 2 * prompt_enhancement.PREFILL_CHUNK
    step = copy.deepcopy(tiny_enhancer.get_decoder())
    step.compile(
        backend="rbln",
        dynamic=False,
        options={"mode": ["strict", "compile_only"], "cache_dir": str(tmp_path)},
    )
    cache = prompt_enhancement.window_cache(tiny_enhancer, window)
    for layer in cache.layers:
        layer.cumulative_length.fill_(1)

    with torch.inference_mode():
        step(
            input_ids=torch.ones((1, 1), dtype=torch.long),
            attention_mask=torch.ones((1, window), dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([1]),
            position_ids=torch.tensor([[1]]),
        )

    stats = [
        stat
        for artifact in sorted(tmp_path.glob("*.rbln"))
        for stat in rebel.RBLNCompiledModel.inspect(str(artifact))["subgraph"]["stats"]
    ]
    assert [stat["device_func_count"] for stat in stats] == [1], stats
    host_ops = {
        op
        for stat in stats
        for ops in stat["host_op_reason"].values()
        for op in ops
        if not op.startswith("args_")
    }
    assert host_ops == set(), host_ops


def test_the_cache_is_built_on_the_device_the_step_will_run_on(
    dummy_device, monkeypatch, tiny_enhancer
):
    """A resident step moves itself on its first call, after the cache is built.

    Reading the device from the model's parameters then gives cpu, and the cache that
    the graphs write back is left on the host.
    """
    from sglang_rbln.compilation import placement, residency

    step = torch.nn.Linear(1, 1)
    assert prompt_enhancement.cache_device(step, tiny_enhancer).type == "cpu"

    placement.place(step, (4, 5, 6, 7))
    monkeypatch.setattr(residency, "logical_device_of", lambda module: 0)
    device = prompt_enhancement.cache_device(step, tiny_enhancer)
    assert (device.type, device.index) == ("rbln", 0)
    cache = prompt_enhancement.window_cache(tiny_enhancer, 128, device)
    assert {layer.keys.device.type for layer in cache.layers} == {"rbln"}


def test_the_compiled_step_runs_where_the_enhancer_is_placed(
    monkeypatch, tiny_enhancer
):
    from sglang_rbln.compilation import placement

    model = copy.deepcopy(tiny_enhancer)
    monkeypatch.setenv("SGLANG_RBLN_COMPONENT_DEVICES", "prompt_enhancer=4,5,6,7")
    monkeypatch.setattr(
        "sglang_rbln.compilation.backend.static_compile_kwargs", lambda module: {}
    )
    monkeypatch.setattr(torch.nn.Module, "compile", lambda self, **k: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "sglang_rbln.patches.stages.text_encoding",
        SimpleNamespace(traceable=lambda m: m),
    )
    enhancer = bare_enhancer(model=model, step=model.get_decoder())

    enhancer._compile()

    assert enhancer.step is model.get_decoder()
    assert placement.devices_of(enhancer.step) == (4, 5, 6, 7)


@pytest.fixture
def enhancer_server(monkeypatch, tmp_path):
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


def http_request(external=None):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(prompt_enhancer=external))
    )


async def upstream_enhance(request, prompt, *, enabled, task, image_paths=None):
    return f"upstream: {prompt}" if enabled else prompt


def enhance_prompt(req, **kwargs):
    kwargs = {"enabled": True, "task": "video"} | kwargs
    return asyncio.run(
        prompt_enhancement.enhance_with_model_enhancer(
            upstream_enhance, req, "a fox", **kwargs
        )
    )


def test_rewrites_with_the_models_enhancer(enhancer_server):
    assert enhance_prompt(http_request()) == "rewritten: a fox"
    assert enhancer_server.calls == [(enhancer_server.path, True, "a fox", "video")]


@pytest.mark.parametrize(
    ("case", "kwargs"),
    [
        ("not requested", {"enabled": False}),
        ("external endpoint configured", {}),
        ("opt-in off", {}),
        ("no prompt_enhancer component", {}),
    ],
    ids=["not-requested", "external-endpoint", "opt-in-off", "no-component"],
)
def test_leaves_the_request_to_upstream(
    enhancer_server, monkeypatch, tmp_path, case, kwargs
):
    req = http_request()
    if case == "external endpoint configured":
        req = http_request(external=object())
    elif case == "opt-in off":
        monkeypatch.delenv("SGLANG_RBLN_PROMPT_ENHANCER")
    elif case == "no prompt_enhancer component":
        (tmp_path / "prompt_enhancer" / "config.json").unlink()
    expected = "a fox" if case == "not requested" else "upstream: a fox"
    assert enhance_prompt(req, **kwargs) == expected
    assert enhancer_server.calls == []


def test_refuses_image_prompts(enhancer_server):
    with pytest.raises(HTTPException) as error:
        enhance_prompt(http_request(), image_paths=["frame.png"])
    assert error.value.status_code == 400
    assert enhancer_server.calls == []


def test_the_user_turn_matches_the_system_prompt_in_use(monkeypatch, tmp_path):
    # upstream's system prompt reads a JSON turn; ltx-core's a plain one
    monkeypatch.delenv("SGLANG_RBLN_PROMPT_ENHANCER_SYSTEM_PROMPT", raising=False)
    enhancer = object.__new__(prompt_enhancement.ModelPromptEnhancer)
    assert json.loads(enhancer.user_turn("a fox", "video")) == {
        "task": "video",
        "prompt": "a fox",
    }
    monkeypatch.setenv("SGLANG_RBLN_PROMPT_ENHANCER_SYSTEM_PROMPT", str(tmp_path))
    assert enhancer.user_turn("a fox", "video") == "user prompt: a fox"

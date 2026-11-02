# SPDX-License-Identifier: Apache-2.0
"""Serve ``enhance_prompt`` with the model's own prompt enhancer on the NPU.

Upstream rewrites a prompt only through an external chat endpoint
(``--prompt-enhancer-config``). Checkpoints that ship their rewrite model as a
``prompt_enhancer/`` component (LTX-2.5: a Gemma-4 instruct model) are left without
one. With ``SGLANG_RBLN_PROMPT_ENHANCER=1`` and no external endpoint configured, the
request's ``enhance_prompt`` runs that component here instead: its decoder layers
compile as one region on ``SGLANG_RBLN_COMPONENT_DEVICES["prompt_enhancer"]`` and it
decodes greedily with a static KV cache, so the prefill and the decode step are one
graph each.

The chat turn follows upstream's enhancer by default (its system prompt, the
``{"task", "prompt"}`` JSON user turn, at most 512 new tokens). A model's own recipe is
set with ``SGLANG_RBLN_PROMPT_ENHANCER_SYSTEM_PROMPT`` (a file),
``SGLANG_RBLN_PROMPT_ENHANCER_USER_TEMPLATE`` (``{prompt}`` / ``{task}``),
``SGLANG_RBLN_PROMPT_ENHANCER_MAX_NEW_TOKENS`` and
``SGLANG_RBLN_PROMPT_ENHANCER_NO_REPEAT_NGRAM``.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import threading
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.entrypoints.openai import (
    prompt_enhancement as upstream,
)
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import envs

logger = logging.getLogger(__name__)

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.entrypoints.openai.prompt_enhancement"
    ".maybe_enhance_prompt"
)
COMPONENT = "prompt_enhancer"
WINDOW_BUCKET = 128


def window_cache(model, window: int):
    """A plain static KV layer of the whole window for every layer that owns its KV.

    Sliding-window layers get the whole window as well: the attention mask, built from
    ``config.sliding_window``, hides what lies outside their window, which is what
    sliding attention means. ``StaticSlidingWindowLayer`` instead branches on a Python
    int that grows every step, so a compiled decode step would recompile at every token.
    """
    from transformers.cache_utils import Cache, StaticLayer

    text = model.config.get_text_config()
    owners = text.num_hidden_layers - (getattr(text, "num_kv_shared_layers", 0) or 0)
    return Cache(layers=[StaticLayer(max_cache_len=window) for _ in range(owners)])


def _banned_by_ngram(history: list[int], size: int) -> list[int]:
    if size <= 0 or len(history) < size:
        return []
    prefix = tuple(history[len(history) - size + 1 :])
    return [
        history[i + size - 1]
        for i in range(len(history) - size + 1)
        if tuple(history[i : i + size - 1]) == prefix
    ]


@torch.inference_mode()
def greedy_decode(
    model, input_ids: torch.Tensor, *, max_new_tokens: int, no_repeat_ngram: int = 0
) -> list[int]:
    """Greedy decoding with a static KV cache: one prefill shape, one decode shape.

    The prompt is right-padded to a multiple of ``WINDOW_BUCKET`` for the prefill; the
    pad tail is masked and overwritten by the decode steps.
    """
    decoder = model.get_decoder()
    head = model.get_output_embeddings()
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    prompt = input_ids.shape[1]
    prefill = -(-prompt // WINDOW_BUCKET) * WINDOW_BUCKET
    window = -(-(prompt + max_new_tokens) // WINDOW_BUCKET) * WINDOW_BUCKET
    cache = window_cache(model, window)
    ids = torch.zeros((1, prefill), dtype=torch.long)
    ids[0, :prompt] = input_ids[0]
    mask = torch.zeros((1, window), dtype=torch.long)
    mask[0, :prompt] = 1
    hidden = decoder(
        input_ids=ids,
        attention_mask=mask[:, :prefill],
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.arange(prefill),
        position_ids=torch.arange(prefill)[None],
    ).last_hidden_state[:, prompt - 1 : prompt]
    for layer in cache.layers:
        layer.cumulative_length.fill_(prompt)
    history, tokens = input_ids[0].tolist(), []
    for position in range(prompt, prompt + max_new_tokens):
        logits = head(hidden).float()[0, -1]
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        if banned := _banned_by_ngram(history, no_repeat_ngram):
            logits[banned] = float("-inf")
        token = int(logits.argmax())
        if token in eos:
            break
        tokens.append(token)
        history.append(token)
        mask[0, position] = 1
        hidden = decoder(
            input_ids=torch.tensor([[token]]),
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([position]),
            position_ids=torch.tensor([[position]]),
        ).last_hidden_state
    return tokens


class ModelPromptEnhancer:
    """The checkpoint's ``prompt_enhancer/`` component, loaded once per process."""

    def __init__(self, model_path: str, compile_model: bool) -> None:
        import transformers

        root = Path(model_path)
        weights = root / COMPONENT
        config = transformers.AutoConfig.from_pretrained(weights)
        architecture = getattr(transformers, config.architectures[0])
        self.model = architecture.from_pretrained(weights, dtype=torch.bfloat16).eval()
        tokenizer_dir = root / "processor" if (root / "processor").is_dir() else weights
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)
        self.system_prompt = self._system_prompt()
        self.lock = threading.Lock()
        if compile_model:
            self._compile()

    @staticmethod
    def _system_prompt() -> str:
        if path := envs.SGLANG_RBLN_PROMPT_ENHANCER_SYSTEM_PROMPT:
            return Path(path).read_text()
        return upstream.PromptEnhancerConfig.model_fields["system_prompt"].default

    def _compile(self) -> None:
        from sglang_rbln.compilation import placement
        from sglang_rbln.compilation.backend import static_compile_kwargs
        from sglang_rbln.patches.stages.text_encoding import compile_regions, traceable

        if devices := envs.SGLANG_RBLN_COMPONENT_DEVICES.get(COMPONENT):
            placement.place(self.model, devices)
        graphs = compile_regions(self.model.get_decoder())
        head = traceable(self.model.get_output_embeddings())
        head.compile(**static_compile_kwargs(head))
        logger.info("Compiling the prompt enhancer as %d regions and its head", graphs)

    def user_turn(self, prompt: str, task: str) -> str:
        if template := envs.SGLANG_RBLN_PROMPT_ENHANCER_USER_TEMPLATE:
            return template.format(prompt=prompt, task=task)
        return json.dumps({"task": task, "prompt": prompt}, ensure_ascii=False)

    def enhance(self, prompt: str, task: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_turn(prompt, task)},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )["input_ids"]
        with self.lock:
            tokens = greedy_decode(
                self.model,
                input_ids,
                max_new_tokens=envs.SGLANG_RBLN_PROMPT_ENHANCER_MAX_NEW_TOKENS,
                no_repeat_ngram=envs.SGLANG_RBLN_PROMPT_ENHANCER_NO_REPEAT_NGRAM,
            )
        rewritten = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if not rewritten:
            raise ValueError("the prompt enhancer returned an empty rewrite")
        return rewritten


_load_lock = threading.Lock()


@functools.cache
def _load(model_path: str, compile_model: bool) -> ModelPromptEnhancer:
    return ModelPromptEnhancer(model_path, compile_model)


def model_enhancer(model_path: str, compile_model: bool) -> ModelPromptEnhancer:
    with _load_lock:  # concurrent first requests load the model once
        return _load(model_path, compile_model)


def enhance(model_path: str, compile_model: bool, prompt: str, task: str) -> str:
    return model_enhancer(model_path, compile_model).enhance(prompt, task)


def ships_enhancer(model_path: str) -> bool:
    return (Path(model_path) / COMPONENT / "config.json").is_file()


async def enhance_with_model_enhancer(
    maybe_enhance_prompt, request, prompt: str, *, enabled: bool, task, image_paths=None
):
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    server_args = get_global_server_args()
    if (
        not enabled
        or not envs.SGLANG_RBLN_PROMPT_ENHANCER
        or request.app.state.prompt_enhancer is not None
        or not ships_enhancer(server_args.model_path)
    ):
        return await maybe_enhance_prompt(
            request, prompt, enabled=enabled, task=task, image_paths=image_paths
        )
    if image_paths:
        from fastapi import HTTPException

        raise HTTPException(
            400, "The model's own prompt enhancer rewrites text prompts only"
        )
    # loading (first request) and decoding both stay off the event loop
    return await asyncio.to_thread(
        enhance, server_args.model_path, server_args.enable_torch_compile, prompt, task
    )


def register() -> str:
    HookRegistry.register(HOOK_TARGET, enhance_with_model_enhancer, HookType.AROUND)
    return HOOK_TARGET

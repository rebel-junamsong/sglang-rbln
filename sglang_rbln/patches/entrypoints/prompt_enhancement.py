# SPDX-License-Identifier: Apache-2.0
"""Serve ``enhance_prompt`` with the model's own prompt enhancer on the NPU.

Upstream rewrites only through an external chat endpoint, so a checkpoint that ships
its rewrite model as a ``prompt_enhancer/`` component (LTX-2.5: a Gemma-4 instruct
model) gets no rewrite at all. With ``SGLANG_RBLN_PROMPT_ENHANCER=1`` and no endpoint
configured, the request runs that component here, compiled on
``SGLANG_RBLN_COMPONENT_DEVICES["prompt_enhancer"]`` and decoding greedily against a
static KV cache. ``..._WHOLE_GRAPH=1`` compiles the decoder's own forward from the
token ids rather than its layers region by region, putting embeddings, masks and
rotary on the device too.

The chat turn follows upstream's enhancer unless ``..._SYSTEM_PROMPT`` (a file),
``..._USER_TEMPLATE``, ``..._MAX_NEW_TOKENS`` or ``..._NO_REPEAT_NGRAM`` say otherwise;
the README lists them in full.
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
from transformers.cache_utils import Cache, StaticLayer

from sglang_rbln import envs

logger = logging.getLogger(__name__)

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.entrypoints.openai.prompt_enhancement"
    ".maybe_enhance_prompt"
)
COMPONENT = "prompt_enhancer"
PREFILL_CHUNK = 128


class WholeGraph(torch.nn.Module):
    """The decoder's own forward, behind a boundary dtype the runtime can bind.

    An attention mask's physical form is a one-byte mask, which pairs with bf16 and
    dlf16 and with no integer dtype, so an integer mask fails the bind
    (``IsDtypeConversionAvailable``). It crosses as bf16 and is cast back inside the
    graph; the model keeps its own dtypes.
    """

    MASK_DTYPE = torch.bfloat16

    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, input_ids, attention_mask, **kwargs):
        return self.decoder(
            input_ids=input_ids, attention_mask=attention_mask.to(torch.long), **kwargs
        )


def mask_dtype(step) -> torch.dtype:
    """The dtype the mask crosses the boundary as: the step says, or long by default.

    Read as an attribute rather than by type, so anything standing in for the step --
    a wrapper that times or traces it -- answers for the module it wraps.
    """
    return getattr(step, "MASK_DTYPE", torch.long)


def cache_device(step, model) -> torch.device:
    """Where the KV cache must live: the device the compiled step will run on.

    A resident step moves its module to its placed NPUs on the first call, which has
    not happened when the cache is built, so the model's parameters still read cpu.
    Allocating from them leaves the cache on the host and the first device run fails.
    """
    from sglang_rbln import device
    from sglang_rbln.compilation import placement, residency

    if placement.devices_of(step) is None or not device.resident_regions():
        return next(model.parameters()).device
    return torch.device("rbln", residency.logical_device_of(step))


def window_cache(model, window: int, device: torch.device | None = None):
    """A static KV layer of the whole window for every layer that owns its KV.

    Sliding-window layers get the whole window too; their mask hides what lies outside
    it. ``StaticSlidingWindowLayer`` would branch on a Python int that grows each step,
    recompiling the decode graph at every token.

    Every cache tensor is allocated here, outside any graph, so the graphs take the
    cache as an input they write back rather than creating it in the prefill graph.
    """
    text = model.config.get_text_config()
    owners = text.num_hidden_layers - (getattr(text, "num_kv_shared_layers", 0) or 0)
    cache = Cache(layers=[StaticLayer(max_cache_len=window) for _ in range(owners)])
    param = next(model.parameters())
    device = param.device if device is None else device
    for index, layer in enumerate(cache.layers):
        full = text.layer_types[index] == "full_attention"
        head_dim = text.head_dim
        if full:
            head_dim = getattr(text, "global_head_dim", None) or head_dim
        row = torch.empty(
            (1, text.num_key_value_heads, 1, head_dim),
            dtype=param.dtype,
            device=device,
        )
        layer.lazy_initialization(row, row)
    return cache


def cache_window(needed: int) -> int:
    """The static cache length, one configured size for every request.

    The cache length and the prefill chunk are the shapes the prefill and the decode
    graph are compiled for, so holding them still is what keeps a server from compiling
    a new pair of graphs for every prompt length it sees. A prompt that does not fit the
    configured length still runs, on graphs of its own.
    """
    limit = envs.SGLANG_RBLN_PROMPT_ENHANCER_MAX_SEQ_LEN
    if needed <= limit:
        return limit
    logger.warning(
        "The prompt and its rewrite need %d tokens, past the %d of "
        "SGLANG_RBLN_PROMPT_ENHANCER_MAX_SEQ_LEN; this request compiles graphs of its "
        "own. Raise that setting to keep every request on one pair of graphs.",
        needed,
        limit,
    )
    return -(-needed // PREFILL_CHUNK) * PREFILL_CHUNK


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
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    no_repeat_ngram: int = 0,
    step: torch.nn.Module | None = None,
) -> list[int]:
    """Greedy decoding with a static KV cache: one prefill shape, one decode shape.

    Both shapes belong to the configuration, not the request -- fixed ``PREFILL_CHUNK``
    chunks against a cache of ``SGLANG_RBLN_PROMPT_ENHANCER_MAX_SEQ_LEN`` -- so prompts
    of every length run the same two graphs. ``step`` is the module they run, the
    decoder unless a ``WholeGraph`` was compiled.

    Everything the step takes is built on the model's device and the mask grows on a
    host copy, so a device run has no eager op of its own between steps.
    """
    step = model.get_decoder() if step is None else step
    head = model.get_output_embeddings()
    softcap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    prompt = input_ids.shape[1]
    window = cache_window(prompt + max_new_tokens)
    padded = -(-prompt // PREFILL_CHUNK) * PREFILL_CHUNK
    device = cache_device(step, model)
    cache = window_cache(model, window, device)
    ids = torch.zeros((1, padded), dtype=torch.long)
    ids[0, :prompt] = input_ids[0]
    host_mask = torch.zeros((1, window), dtype=mask_dtype(step))
    host_mask[0, :prompt] = 1
    mask = host_mask.to(device)
    for start in range(0, padded, PREFILL_CHUNK):
        positions = torch.arange(start, start + PREFILL_CHUNK, device=device)
        hidden = step(
            # a chunk of its own, not a slice: a slice reaches the graph as its base,
            # whose length would follow the prompt again
            input_ids=ids[:, start : start + PREFILL_CHUNK].clone().to(device),
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=positions,
            position_ids=positions[None],
        ).last_hidden_state
    hidden = hidden[:, prompt - 1 - (padded - PREFILL_CHUNK)][:, None]
    for layer in cache.layers:
        layer.cumulative_length.fill_(prompt)
    history, tokens = input_ids[0].tolist(), []
    for position in range(prompt, prompt + max_new_tokens):
        logits = head(hidden).float().cpu()[0, -1]
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap
        if banned := _banned_by_ngram(history, no_repeat_ngram):
            logits[banned] = float("-inf")
        token = int(logits.argmax())
        if token in eos:
            break
        tokens.append(token)
        history.append(token)
        host_mask[0, position] = 1
        mask.copy_(host_mask)  # the device mask grows by a copy, not an eager write
        hidden = step(
            input_ids=torch.tensor([[token]], device=device),
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([position], device=device),
            position_ids=torch.tensor([[position]], device=device),
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
        self.step = self.model.get_decoder()
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
        decoder = self.model.get_decoder()
        if envs.SGLANG_RBLN_PROMPT_ENHANCER_WHOLE_GRAPH:
            self.step = WholeGraph(decoder)
            # a new module, so it carries no placement of its own and would resolve to
            # logical device 0 -- the DiT's set -- rather than the enhancer's
            if devices := placement.devices_of(decoder):
                placement.place(self.step, devices)
            self.step.compile(**static_compile_kwargs(decoder))
            graphs = 1
        else:
            graphs = compile_regions(decoder)
        head = traceable(self.model.get_output_embeddings())
        head.compile(**static_compile_kwargs(head))
        logger.info("Compiling the prompt enhancer as %d graphs and its head", graphs)

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
                step=self.step,
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

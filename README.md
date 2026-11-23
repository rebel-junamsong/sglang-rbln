# SGLang RBLN

<div align="center">

<picture>
  <source srcset="assets/rebellions_logo_dark.png" media="(prefers-color-scheme: dark)">
  <source srcset="assets/rebellions_logo_light.png" media="(prefers-color-scheme: light)">
  <img src="assets/rebellions_logo_light.png" alt="Rebellions" width="60%">
</picture>

[![License](https://img.shields.io/github/license/rebel-junamsong/sglang-rbln)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://docs.rbln.ai/)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

</div>

`sglang-rbln` runs [SGLang Diffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen)
on RBLN NPUs. It is an out-of-tree platform plugin: SGLang's own pipelines are
used unchanged, and the transformer, text encoder and VAE are compiled with
[rebel-compiler](https://docs.rbln.ai/) through `torch.compile`.

## Install

```bash
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.rbln.ai/simple \
  sglang-rbln
uv pip install flashinfer-python==0.7.1
```

Requires Python 3.11–3.14, an RBLN NPU with its driver, and an account on the
[RBLN package index](https://docs.rbln.ai/latest/getting_started/installation_guide.html).
sglang 0.5.21 pins `flashinfer-python==0.6.18`, which asserts a CUDA device at
import; the second command replaces it with 0.7.1, which does not. flashinfer
is never used on RBLN.
`torch-rbln` (built for `torch==2.13.0`) comes with it and keeps tensors on the
NPUs between compiled graphs.

To reproduce a published result, build the environment from the committed
`uv.lock`. It pins the exact `rebel-compiler` and `torch-rbln` builds the result
was measured with (the RBLN index needs credentials in `~/.netrc`;
`.python-version` selects CPython 3.12, the only ABI the pinned compiler ships):

```bash
uv sync --frozen --extra test
```

The plugin activates itself when an NPU is present, or when
`RBLN_FORCE_NPU_NAME` names the target NPU (for example `RBLN-CA25`) on a host
without one. Set `SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rbln` when several platform
plugins are installed.

## Quick start

The upstream command; `--tp-size` is how many NPUs each model is split across.

```bash
sglang serve --model-path Lightricks/LTX-2.5-Diffusers --tp-size 4
```

On eight NPUs, the DiT and the video decoder run tensor-parallel on NPUs 0–3.
The Gemma text encoder does not fit next to them, so it runs on NPUs 4–7, with
the modules that read its output: the text connector, the audio decoder and
vocoder, the latent upsampler and the duration head. Each set is one device to
`torch-rbln`, and tensors cross between the two sets on the NPUs. With fewer
than eight NPUs, pass `--text-encoder-cpu-offload` to keep the text encoder on
the CPU. To pick the NPUs, set `RBLN_VISIBLE_DEVICES` (for example
`0,1,2,3,4,5,6,7`); by default every NPU on the host is used.

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")
video = client.videos.create(
    model="Lightricks/LTX-2.5-Diffusers",
    prompt="A red fox walking through a snowy forest at dawn.",
)
```

## How it runs

Each model runs as compiled graphs on its NPUs. A graph's outputs stay there as
`torch-rbln` tensors for the next graph, and the eager ops between graphs run on
the NPUs through `torch-rbln`; a tensor comes back to the host only where upstream
reads it there (the decoded video, a sampled token). For a CPU reference run,
leave the plugin out: upstream's own CPU platform runs the same pipeline.

## Compiled graphs

The plugin compiles a model in the largest units the NPU can hold: a block list
(DiT blocks, text-encoder layers) is one region graph. When a region does not
fit in the NPU's free memory, it is halved and retried, down to single blocks,
and the split is logged.

`SGLANG_RBLN_BLOCKS_PER_REGION` caps the region size. Each request
shape compiles once per process and is cached on disk under
`SGLANG_RBLN_CACHE_DIR`. Later shapes of the same module share its weights on
the NPU. A compile error fails the request: nothing falls back to the host.

## Configuration

| Variable | Effect |
|---|---|
| `SGLANG_RBLN_CACHE_DIR` | compiled-graph cache (default `~/.cache/sglang_rbln`) |
| `SGLANG_RBLN_DISABLE_COMPILE_CACHE=1` | compile without the cache |
| `SGLANG_RBLN_COMPONENT_DEVICES=text_encoder=4,5,6,7` | run the named pipeline components (`;`-separated, e.g. `text_encoder=4,5,6,7;vae=8`) on their own NPUs, given as indices into `RBLN_VISIBLE_DEVICES`; each is tensor-parallel over its list, in place of `--tp-size`, so a component that does not fit next to the DiT gets NPUs of its own. Without an entry, a text encoder runs on the next `--tp-size` visible NPUs after the DiT's when there are that many and each can hold its share of the weights |
| `SGLANG_RBLN_COMPILE_ONLY=1` | compile the graphs and stop before running them |
| `SGLANG_RBLN_LOG_TIMELINE=1` | log, per pipeline stage, its wall time, its compiled graph runs and the time inside them, and the host time between them, the largest stretches first, each named by the graph that ran after it |
| `SGLANG_RBLN_PROFILE_DIR=<dir>` | write an RBLN Profiler trace (`*.pb`, Perfetto) of one request to `<dir>`; start the server with `RBLN_PROFILER=1` |
| `SGLANG_RBLN_PROFILE_REQUEST=3` | which request (counting from 1, warmups excluded) `SGLANG_RBLN_PROFILE_DIR` captures |
| `SGLANG_RBLN_LTX2_DETAILING_LORA=<file>` | experimental: run stage 2 of `LTX2TwoStagePipeline` as ltx-core does, with this detailing IC-LoRA (e.g. `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors`) at strength 0.5 and the stage-1 latent as in-context reference tokens. The LoRA stays dynamic in both stages (strength 0 in stage 1), so both share one copy of the DiT weights on the NPUs |
| `SGLANG_RBLN_BLOCKS_PER_REGION=7` | cap the blocks compiled as one region graph (see [Compiled graphs](#compiled-graphs); 1 compiles each block alone). Set it only when one graph does not fit or does not compile |
| `SGLANG_RBLN_PROMPT_ENHANCER=1` | serve a request's `enhance_prompt` with the checkpoint's own `prompt_enhancer/` component (LTX-2.5: a Gemma-4 instruct model) when no `--prompt-enhancer-config` endpoint is set: its decoder's whole forward compiles as one graph on `SGLANG_RBLN_COMPONENT_DEVICES` `prompt_enhancer=…`, so embeddings, masks and rotary run on the device (LTX-2.5's Gemma-4 enhancer on one CA22, 855 prompt tokens: 28.7 ms a token against 34 ms on the host), and it decodes with ltx-core's recipe (greedy, no repeated 5-gram, at most 600 new tokens) against a static 2048-token KV cache, so every request that fits runs one prefill graph and one decode graph. Text prompts only |
| `SGLANG_RBLN_PROMPT_ENHANCER_SYSTEM_PROMPT=<file>` | system prompt for `SGLANG_RBLN_PROMPT_ENHANCER`. Without it the enhancer uses upstream's system prompt and JSON user turn. Point it at ltx-core's `gemma4_t2v_system_prompt.txt` (from the LTX-2 repository, not shipped here) for the official LTX-2.5 rewrite; the user turn is then that recipe's `user prompt: {prompt}` |

`--enable-torch-compile` is on by default. The DiT compiles as one graph;
`--regional-compile` compiles it as block regions instead.

## Supported models

| Model | Pipeline | Endpoint |
|---|---|---|
| `Lightricks/LTX-2.5-Diffusers` | `LTX2Pipeline` | `/v1/videos` |

## Limits

- The audio decoder and vocoder run as one compiled graph: 0.34 s against
  2.2-3.2 s on the CPU.
- The first request of each shape pays the compile, which takes minutes for a
  large video DiT. Restarts reuse the on-disk cache.
- Layerwise offload is refused: `--dit-layerwise-offload`,
  `--layerwise-offload-components` and `--offload-during-compile true` would move a
  model's weights off its NPUs. The component offload flags (`--dit-cpu-offload` and
  the like) leave every component on its NPUs; `--text-encoder-cpu-offload` runs the
  text encoder on the CPU instead.
- The unit tests and the end-to-end serve tests run on a CPU host with
  `RBLN_FORCE_NPU_NAME` set (the serve tests without an NPU). Tests marked
  `npu` need an NPU host, so CI runs lint only.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Upstream workarounds are tracked by
`tests/test_patch_inventory.py`, which fails once the upstream change lands.

## License

[Apache License 2.0](LICENSE)

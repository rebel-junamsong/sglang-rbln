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
Install `sglang-rbln[device]` instead to add `torch-rbln` (built for
`torch==2.13.0`) for `device` execution mode.

To reproduce a published result, build the environment from the committed
`uv.lock`. It pins the exact `rebel-compiler` and `torch-rbln` builds the result
was measured with (the RBLN index needs credentials in `~/.netrc`;
`.python-version` selects CPython 3.12, the only ABI the pinned compiler ships):

```bash
uv sync --frozen --extra device --extra test
```

The plugin activates itself when an NPU is present, or when
`RBLN_FORCE_NPU_NAME` names the target NPU (for example `RBLN-CA25`) on a host
without one. Set `SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rbln` when several platform
plugins are installed.

## Quick start

The examples serve each model over four NPUs with compiler tensor parallelism.

### LTX-2.5

```bash
RBLN_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SGLANG_RBLN_NUM_DEVICES=4 \
sglang serve \
  --model-path Lightricks/LTX-2.5-Diffusers \
  --pipeline-class-name LTX2Pipeline
```

The DiT and the video decoder run tensor-parallel on NPUs 0–3. The Gemma text
encoder does not fit next to the DiT, so it runs tensor-parallel on the next
four visible NPUs, here 4–7. With fewer than eight NPUs visible, pass
`--text-encoder-cpu-offload` to keep it on the CPU.

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:30000/v1")
video = client.videos.create(
    model="Lightricks/LTX-2.5-Diffusers",
    prompt="A red fox walking through a snowy forest at dawn.",
)
```

### Cosmos3

```bash
SGLANG_RBLN_NUM_DEVICES=4 sglang serve --model-path nvidia/Cosmos3-Edge
```

```bash
curl -sS -X POST http://localhost:30000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A small warehouse robot moves a blue box across a clean floor.",
       "size": "640x640", "num_inference_steps": 20, "guidance_scale": 4.0}'
```

Cosmos3 guardrails need `cosmos-guardrail`; `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1`
turns them off. The [SGLang Diffusion cookbook](https://docs.sglang.io/cookbook/diffusion/)
covers the full request options for both models.

## Execution modes

Compiled graphs run on the NPU in both NPU modes; `SGLANG_RBLN_EXECUTION_MODE`
picks where tensors live between them.

| mode | where the graphs run | tensors between graphs | needs |
|---|---|---|---|
| CPU eager: `staged` with `--disable-torch-compile` | CPU | CPU tensors | — |
| `staged`: NPU, host-staged I/O | NPU | CPU tensors: each graph call uploads its inputs and downloads its outputs | `rebel-compiler` |
| `resident`: NPU, device-resident | NPU | `torch-rbln` tensors that stay on the NPU for the next graph; eager ops between graphs run through `torch-rbln`. With `SGLANG_RBLN_NUM_DEVICES` > 1, compiled regions keep them on the tensor-parallel NPUs | `torch-rbln` |

`auto` (default) is `resident` when `torch-rbln` is installed, otherwise `staged`.
The former names `host` (= `staged`) and `device` (= `resident`) are still
accepted, with a deprecation warning.

## Compiled graphs

The plugin compiles a model in the largest units the NPU can hold:

- A FLUX 3 Action denoise step is one graph, and so are its text context and
  conditioning streams.
- Every other block list (DiT blocks, text-encoder layers, and with
  `SGLANG_RBLN_COMPILE_VAE=1` the resnets of a Wan VAE up-block) is one region graph. When a region does not fit in the NPU's
  free memory, it is halved and retried, down to single blocks, and the split
  is logged.

`SGLANG_RBLN_BLOCKS_PER_REGION` caps the region size instead. Each request
shape compiles once per process and is cached on disk under
`SGLANG_RBLN_CACHE_DIR`. Later shapes of the same module share its weights on
the NPU. A compile error fails the request unless `SGLANG_RBLN_COMPILE_STRICT=0`.

## Configuration

| Variable | Effect |
|---|---|
| `SGLANG_RBLN_EXECUTION_MODE` | `auto` (default), `host`, or `device`: where activations live between compiled graphs |
| `SGLANG_RBLN_CACHE_DIR` | compiled-graph cache (default `~/.cache/sglang_rbln`) |
| `SGLANG_RBLN_DISABLE_COMPILE_CACHE=1` | compile without the cache |
| `SGLANG_RBLN_STATIC_OUTPUT=1` | reuse one device output buffer per graph output instead of allocating per run |
| `SGLANG_RBLN_CHECK_GRAPHS=1` | rerun each compiled graph once on the CPU and log the deviation of the NPU output (precision triage) |
| `SGLANG_RBLN_NUM_DEVICES=4` | compile every graph for tensor parallelism over that many NPUs (rebel `num_devices`) |
| `SGLANG_RBLN_COMPONENT_DEVICES=text_encoder=4,5,6,7` | run the named pipeline components (`;`-separated, e.g. `text_encoder=4,5,6,7;vae=8`) on their own NPUs, given as indices into `RBLN_VISIBLE_DEVICES`; each is tensor-parallel over its list, in place of `SGLANG_RBLN_NUM_DEVICES`, so a component that does not fit next to the DiT gets NPUs of its own. Without an entry, a text encoder runs on the next `SGLANG_RBLN_NUM_DEVICES` visible NPUs after the DiT's when there are that many and each can hold its share of the weights |
| `SGLANG_RBLN_COMPILE_VAE=1` | also compile the video VAE for the NPU: FLUX 3 Action with one graph per encoder and decoder on the last tensor-parallel NPU, and Cosmos3 with one graph per Wan VAE block, each compiled for a single NPU and spread over the tensor-parallel NPUs (needs rebel-compiler with rebellions-sw/rebel_compiler#14989); it otherwise runs on the host |
| `SGLANG_RBLN_COMPILE_ONLY=1` | compile the graphs and stop before running them |
| `SGLANG_RBLN_COMPILE_STRICT=0` | let a graph that fails to compile run on the host instead of failing the request (strict is on by default, so a compile failure is never silently timed as NPU work) |
| `SGLANG_RBLN_LOG_GRAPHS=1` | log every graph handed to the compiler |
| `SGLANG_RBLN_DUMP_GRAPHS=<dir>` | write each graph (readable FX and example inputs) to `<dir>` |
| `SGLANG_RBLN_RUNTIME_REPORTS=1` | log the runtime's device / host-op / buffer-prep time split every 100 graph runs |
| `SGLANG_RBLN_LOG_CROSSINGS=1` | log, per pipeline stage, every host↔NPU transfer of a tensor: downloads by the op that read it on the host, and host uploads |
| `SGLANG_RBLN_LOG_TIMELINE=1` | log, per pipeline stage, its wall time, its compiled graph runs and the time inside them, and the host time between them, the largest stretches first, each named by the graph that ran after it |
| `SGLANG_RBLN_PROFILE_DIR=<dir>` | write an RBLN Profiler trace (`*.pb`, Perfetto) of one request to `<dir>`; start the server with `RBLN_PROFILER=1` |
| `SGLANG_RBLN_PROFILE_REQUEST=3` | which request (counting from 1, warmups excluded) `SGLANG_RBLN_PROFILE_DIR` captures |
| `SGLANG_RBLN_COMPILE_VAE_ENCODER=0` | keep the LTX-2 video VAE encoder that I2V conditioning images go through on the CPU. On by default, across the VAE's devices like the decoder; it falls back to the CPU on its own if it does not fit them |
| `SGLANG_RBLN_DURATION_HEAD_ON_NPU=0` | keep the LTX-2.5 duration head (`auto_duration` requests) on the CPU. On by default, on one NPU: its own if `SGLANG_RBLN_COMPONENT_DEVICES` places it, else the text encoder's last, else the last tensor-parallel one |
| `SGLANG_RBLN_UPSAMPLER_ON_NPU=1` | compile the LTX-2 latent upsampler (`LTX2TwoStagePipeline`) on the text encoder's NPUs (or its own if `SGLANG_RBLN_COMPONENT_DEVICES` places it, else the DiT's): the DiT's set has no room for it beside both DiT shapes and the VAE decoder. In resident mode its latents cross between the two sets on the device |
| `SGLANG_RBLN_LTX2_DETAILING_LORA=<file>` | experimental: run stage 2 of `LTX2TwoStagePipeline` as the official DFR does, with this detailing IC-LoRA (e.g. `ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors`) at strength 0.5 and the stage-1 latent as in-context reference tokens. The LoRA stays dynamic in both stages (strength 0 in stage 1), so both share one copy of the DiT weights on the NPUs |
| `SGLANG_RBLN_COMPILE_DIFFUSION_DECODER=1` | compile the LTX-2.5 diffusion decoder (`use_diffusion_decoder` requests), one region per stage `ModuleList`, on the DiT's device set or the one `SGLANG_RBLN_COMPONENT_DEVICES` names for `diffusion_decoder`: its ~16 GiB of fixed graph memory does not fit one NPU, so the compiler shards it across that set and shares its compile context. Off by default, which runs the decoder eagerly as upstream does |
| `SGLANG_RBLN_VOCODER_ON_NPU=1` | compile the LTX-2 vocoder for one NPU, the last tensor-parallel one. Off by default: the NPU's convolutions accumulate with an upward gain that compounds through the vocoder's ~100 convs, so its waveform lands below CPU bf16 (12.9 vs 16.6 dB SNR against float32 at 5 s of audio) |
| `SGLANG_RBLN_COMPILE_AUDIO=1` | compile the LTX-2 audio VAE decoder and the vocoder as one graph on one NPU: the vocoder's own if `SGLANG_RBLN_COMPONENT_DEVICES` places it (`vocoder=8`), else the last tensor-parallel one. Audio latent in, waveform out with nothing on the host: the latent denormalization runs in the graph, the latent goes in flat and the waveform comes out padded to whole device tiles, so no edge needs a host pad, transpose or crop. 0.34 s against 2.2-3.2 s on the CPU at 5 s of audio; its precision is the vocoder's (see `SGLANG_RBLN_VOCODER_ON_NPU`). Takes the place of `SGLANG_RBLN_VOCODER_ON_NPU` |
| `SGLANG_RBLN_BLOCKS_PER_REGION=7` | cap the blocks compiled as one region graph (see [Compiled graphs](#compiled-graphs); 1 compiles each block alone, and for FLUX 3 Action any value replaces the one-graph step with block regions). Set it only when one graph does not fit or does not compile |
| `SGLANG_RBLN_PROMPT_ENHANCER=1` | serve a request's `enhance_prompt` with the checkpoint's own `prompt_enhancer/` component (LTX-2.5: a Gemma-4 instruct model) when no `--prompt-enhancer-config` endpoint is set: its decoder layers compile as one region on `SGLANG_RBLN_COMPONENT_DEVICES` `prompt_enhancer=…` and it decodes greedily with a static KV cache (one prefill graph, one decode graph). Text prompts only |
| `SGLANG_RBLN_PROMPT_ENHANCER_WHOLE_GRAPH=0` | compile the prompt enhancer's decoder layers region by region instead of its whole forward from the token ids. The default compiles the forward, so embeddings, masks and rotary run on the device and a decode step is one device function rather than one per layer (LTX-2.5's Gemma-4 enhancer on one CA22, 855 prompt tokens: 28.7 ms a token against 34 ms on the host). Set it to 0 only if a checkpoint's decoder will not compile whole |
| `SGLANG_RBLN_PROMPT_ENHANCER_MAX_SEQ_LEN=2048` | the static KV cache the prompt enhancer compiles for (default 2048). It is the prefill and decode graphs' shape, so every request that fits runs the same two graphs instead of a new pair per prompt length; a prompt and rewrite needing more still runs, on graphs of its own. Each step attends over the whole length; measured on one CA22 with LTX-2.5's Gemma-4 enhancer, a 2048 cache costs nothing against a prompt-shaped 896 one (28.7 ms a step against 30.8 ms) |
| `SGLANG_RBLN_PROMPT_ENHANCER_SYSTEM_PROMPT=<file>` | system prompt for `SGLANG_RBLN_PROMPT_ENHANCER` (default: upstream's enhancer system prompt) |
| `SGLANG_RBLN_PROMPT_ENHANCER_USER_TEMPLATE='user prompt: {prompt}'` | user turn for `SGLANG_RBLN_PROMPT_ENHANCER`, with `{prompt}` and `{task}` (default: upstream's `{"task", "prompt"}` JSON) |
| `SGLANG_RBLN_PROMPT_ENHANCER_MAX_NEW_TOKENS=512` | longest rewrite `SGLANG_RBLN_PROMPT_ENHANCER` generates (default 512, upstream's) |
| `SGLANG_RBLN_PROMPT_ENHANCER_NO_REPEAT_NGRAM=0` | ban repeating any n-gram of this size in the rewrite (0 = off; the LTX-2.5 recipe uses 5) |

`--enable-torch-compile` and `--regional-compile` are on by default; pass either
flag explicitly to change it.

## Supported models

| Model | Pipeline | Endpoint |
|---|---|---|
| `Lightricks/LTX-2.5-Diffusers` | `LTX2Pipeline` | `/v1/videos` |
| `nvidia/Cosmos3-Edge`, `nvidia/Cosmos3-Super` | `Cosmos3Pipeline` | `/v1/images/generations`, `/v1/videos` |
| `black-forest-labs/flux-3-action-droid` | `Flux3ActionPipeline` | `/v1/actions/generations` |

FLUX 3 Action does not fit one NPU: serve it with `SGLANG_RBLN_NUM_DEVICES=4`
or more.
It denoises the future video jointly with the actions; send
`"parameters": {"return_video": true}` to also get that video, decoded by the
video VAE, as `data[0].video` (`format` `mp4`, `fps`, `shape` `[frames, height,
width, 3]`, `b64_json`). The video VAE decodes on the host unless
`SGLANG_RBLN_COMPILE_VAE=1`; compiled, its decoder runs as one graph on the last
tensor-parallel NPU.

## Limits

- Several compiler fixes this plugin relies on (tensor-parallel LayerNorm
  statistics, the Cosmos3 VAE regions) are not in a
  rebel-compiler release yet. Without them, FLUX 3 actions drift from the CPU
  result at TP > 1 and `SGLANG_RBLN_COMPILE_VAE=1` fails for Cosmos3.
- RBLN NPUs have no float32 arithmetic. In `device` mode the Cosmos3 timestep
  embedder, text pathway and final norm run on the host. In `host` mode the
  LTX-2 vocoder loads in float32 and runs on the CPU unless
  `SGLANG_RBLN_VOCODER_ON_NPU=1`.
- The first request of each shape pays the compile, which takes minutes for a
  large video DiT. Restarts reuse the on-disk cache.
- The unit tests run on a CPU host with `RBLN_FORCE_NPU_NAME` set. Tests marked
  `npu` and the end-to-end serve tests need an NPU host, so CI runs lint only.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Upstream workarounds are tracked by
`tests/test_patch_inventory.py` and `tests/test_compiler_inventory.py`, which
fail once the upstream change lands.

## License

[Apache License 2.0](LICENSE)

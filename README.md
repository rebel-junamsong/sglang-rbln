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
SGLANG_RBLN_COMPONENT_DEVICES=text_encoder=4,5,6,7 \
sglang serve \
  --model-path Lightricks/LTX-2.5-Diffusers \
  --pipeline-class-name LTX2Pipeline
```

The Gemma text encoder does not fit next to the DiT, so it gets NPUs 4–7 of
its own. The video decoder is compiled across the DiT's four NPUs.

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

`SGLANG_RBLN_EXECUTION_MODE` picks where activations live between compiled
graphs:

- `host`: activations are CPU tensors; each compiled graph copies its inputs to
  the NPU and its outputs back. Needs only `rebel-compiler`.
- `device`: activations are `torch-rbln` tensors that stay on the NPU, and
  eager operations between graphs run through `torch-rbln`. With
  `SGLANG_RBLN_NUM_DEVICES` > 1, compiled regions keep activations resident on
  the tensor-parallel NPUs.
- `auto` (default): `device` when `torch-rbln` is installed, otherwise `host`.

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
| `SGLANG_RBLN_COMPONENT_DEVICES=text_encoder=4,5,6,7` | run the named pipeline components (`;`-separated, e.g. `text_encoder=4,5,6,7;vae=8`) on their own NPUs, given as indices into `RBLN_VISIBLE_DEVICES`; each is tensor-parallel over its list, in place of `SGLANG_RBLN_NUM_DEVICES`, so a component that does not fit next to the DiT gets NPUs of its own |
| `SGLANG_RBLN_COMPILE_VAE=1` | also compile the video VAE for the NPU: FLUX 3 Action with one graph per encoder and decoder on the last tensor-parallel NPU, and Cosmos3 with one graph per Wan VAE block, each compiled for a single NPU and spread over the tensor-parallel NPUs (needs rebel-compiler with rebellions-sw/rebel_compiler#14989); it otherwise runs on the host |
| `SGLANG_RBLN_COMPILE_ONLY=1` | compile the graphs and stop before running them |
| `SGLANG_RBLN_COMPILE_STRICT=0` | let a graph that fails to compile run on the host instead of failing the request (strict is on by default, so a compile failure is never silently timed as NPU work) |
| `SGLANG_RBLN_LOG_GRAPHS=1` | log every graph handed to the compiler |
| `SGLANG_RBLN_DUMP_GRAPHS=<dir>` | write each graph (readable FX and example inputs) to `<dir>` |
| `SGLANG_RBLN_RUNTIME_REPORTS=1` | log the runtime's device / host-op / buffer-prep time split every 100 graph runs |
| `SGLANG_RBLN_PROFILE_DIR=<dir>` | write an RBLN Profiler trace (`*.pb`, Perfetto) of one request to `<dir>`; start the server with `RBLN_PROFILER=1` |
| `SGLANG_RBLN_PROFILE_REQUEST=3` | which request (counting from 1, warmups excluded) `SGLANG_RBLN_PROFILE_DIR` captures |
| `SGLANG_RBLN_VOCODER_ON_NPU=1` | compile the LTX-2 vocoder for one NPU (off by default: the CPU is faster below about 2 s of audio) |
| `SGLANG_RBLN_BLOCKS_PER_REGION=7` | cap the blocks compiled as one region graph (see [Compiled graphs](#compiled-graphs); 1 compiles each block alone, and for FLUX 3 Action any value replaces the one-graph step with block regions). Set it only when one graph does not fit or does not compile |

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

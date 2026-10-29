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
used unchanged, and the DiT blocks are compiled with
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

### LTX-2.5

```bash
sglang serve \
  --model-path Lightricks/LTX-2.5-Diffusers \
  --pipeline-class-name LTX2Pipeline
```

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
sglang serve --model-path nvidia/Cosmos3-Edge
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

## Configuration

| Variable | Effect |
|---|---|
| `SGLANG_RBLN_EXECUTION_MODE` | `auto` (default), `host`, or `device`: where activations live between compiled graphs |
| `SGLANG_RBLN_CACHE_DIR` | compiled-graph cache (default `~/.cache/sglang_rbln`) |
| `SGLANG_RBLN_DISABLE_COMPILE_CACHE=1` | compile without the cache |
| `SGLANG_RBLN_STATIC_OUTPUT=1` | reuse one device output buffer per graph output instead of allocating per run |
| `SGLANG_RBLN_CHECK_GRAPHS=1` | rerun each compiled graph once on the CPU and log the deviation of the NPU output (precision triage) |
| `SGLANG_RBLN_NUM_DEVICES=4` | compile every graph for tensor parallelism over that many NPUs (rebel `num_devices`) |
| `SGLANG_RBLN_COMPILE_VAE=1` | also compile the video VAE for the NPU: FLUX 3 Action, and Cosmos3 with one graph per Wan VAE block, each compiled for a single NPU and spread over the tensor-parallel NPUs (needs rebel-compiler with rebellions-sw/rebel_compiler#14989); it otherwise runs on the host |
| `SGLANG_RBLN_COMPILE_ONLY=1` | compile the graphs and stop before running them |
| `SGLANG_RBLN_COMPILE_STRICT=1` | fail a request whose graph cannot be compiled instead of running it on the host |
| `SGLANG_RBLN_LOG_GRAPHS=1` | log every graph handed to the compiler |
| `SGLANG_RBLN_DUMP_GRAPHS=<dir>` | write each graph (readable FX and example inputs) to `<dir>` |
| `SGLANG_RBLN_RUNTIME_REPORTS=1` | log the runtime's device / host-op / buffer-prep time split every 100 graph runs |
| `SGLANG_RBLN_PROFILE_DIR=<dir>` | write an RBLN Profiler trace (`*.pb`, Perfetto) of one request to `<dir>`; start the server with `RBLN_PROFILER=1` |
| `SGLANG_RBLN_PROFILE_REQUEST=3` | which request (counting from 1, warmups excluded) `SGLANG_RBLN_PROFILE_DIR` captures |
| `SGLANG_RBLN_VOCODER_ON_NPU=1` | compile the LTX-2 vocoder for one NPU (off by default: the CPU is faster below about 2 s of audio) |
| `SGLANG_RBLN_BLOCKS_PER_REGION=7` | compile that many consecutive DiT blocks as one regional graph (default 4; 1 compiles each block alone) |
| `SGLANG_RBLN_FLUX3_ONE_GRAPH=1` | compile each FLUX 3 Action DiT entry point (text context, conditioning streams, a whole denoise step) as one graph instead of per-block regions |

`--enable-torch-compile` and `--regional-compile` are on by default; pass either
flag explicitly to change it.

## Supported models

| Model | Pipeline | Endpoint |
|---|---|---|
| `Lightricks/LTX-2.5-Diffusers` | `LTX2Pipeline` | `/v1/videos` |
| `nvidia/Cosmos3-Edge`, `nvidia/Cosmos3-Super` | `Cosmos3Pipeline` | `/v1/images/generations`, `/v1/videos` |
| `black-forest-labs/flux-3-action-droid` | `Flux3ActionPipeline` | `/v1/actions/generations` |

FLUX 3 Action does not fit one NPU: serve it with `SGLANG_RBLN_NUM_DEVICES=4`.
It denoises the future video jointly with the actions; send
`"parameters": {"return_video": true}` to also get that video, decoded by the
video VAE, as `data[0].video` (`format` `mp4`, `fps`, `shape` `[frames, height,
width, 3]`, `b64_json`). The video VAE decodes on the host unless
`SGLANG_RBLN_COMPILE_VAE=1`; compiled, it needs an NPU's memory to itself.
Tensor parallelism needs a compiler that keeps LayerNorm statistics whole across
devices; older ones return actions that drift from the CPU result.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Upstream workarounds are tracked by
`tests/test_patch_inventory.py` and `tests/test_compiler_inventory.py`, which
fail once the upstream change lands.

## License

[Apache License 2.0](LICENSE)

# sglang-rbln

Out-of-tree RBLN NPU platform plugin for [SGLang Diffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen),
backed by torch-rbln. Targets: LTX-2.5 (`LTX2Pipeline`, one-stage, distilled)
and Cosmos3 (`Cosmos3OmniPipeline`) through `sglang serve`.

The platform is a `Platform` subclass selected through an entry point;
everything the platform seams cannot express goes through tracked hooks.

## Install

Python 3.11 to 3.14. Install the RBLN driver first and confirm `rbln-smi`
sees the NPUs ([installation guide](https://docs.rbln.ai/latest/getting_started/installation_guide.html)).
`rebel-compiler` comes from the RBLN index, which needs an RBLN Portal account:

```bash
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.rbln.ai/simple \
  -e '.[test]'
```

Pins: `sglang[diffusion]==0.5.22`, `rebel-compiler==0.12.0`. sglang has no
CUDA-free distribution; the PyTorch CPU index keeps `torch` itself CPU-only,
and the CUDA kernel packages sglang depends on install but are never imported
on this platform.

The `rbln` torch device needs `torch-rbln` (`.[device]`, installed with
`--extra-index-url https://download.pytorch.org/whl/cpu` per the guide). The
published build (0.1.8) requires `torch==2.9.1+cpu` while sglang 0.5.22
requires `torch==2.14.1`, so the two do not install together yet.

## Execution modes

| `SGLANG_RBLN_EXECUTION_MODE` | Tensors | `is_cpu()` | Dispatch key | When |
|---|---|---|---|---|
| `device` | `rbln` | False | `PrivateUse1` | torch-rbln importable |
| `host` | `cpu` | True | `CPU` | torch-rbln absent |

`auto` (default) picks `device` when torch-rbln is importable, else `host`.
Both modes run CustomOps through the `rbln` key: a forward registered with
`CustomOp.register_oot_forward` wins, anything else takes `forward_native`.

## Compile and run

`torch.compile` is on by default on this platform (`--enable-torch-compile`
turns it off explicitly). SGLang hands the DiT blocks to the `rbln` backend,
which compiles each FX graph with rebel-compiler and runs it on the NPU;
inputs and outputs stay CPU tensors in `host` mode.

| Variable | Effect |
|---|---|
| `SGLANG_RBLN_CACHE_DIR` | compiled-graph cache (default `~/.cache/sglang_rbln`) |
| `SGLANG_RBLN_DISABLE_COMPILE_CACHE=1` | compile without the cache |
| `SGLANG_RBLN_COMPILE_ONLY=1` | compile the graphs and stop before running them |

## Select the platform

| Select with | Platform class | Notes |
|---|---|---|
| unset | `sglang_rbln.platform:RblnPlatform` | Auto-detected when `/dev/rbln*` exists and `rebel` is installed, and no other plugin answers |
| `SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rbln` | same | Required when several platform plugins are installed |

On a host without an NPU, `RBLN_FORCE_NPU_NAME=RBLN-CA25` makes the platform
activate and names the compile target.

## Serve LTX-2.5

```bash
SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rbln sglang serve \
  --model-path Lightricks/LTX-2.5-Diffusers \
  --pipeline-class-name LTX2Pipeline
```

The checkpoint is gated; accept the license on Hugging Face and log in first.
Text-to-video with audio goes through `POST /v1/videos`; see the
[cookbook](https://docs.sglang.io/cookbook/diffusion/LTX/LTX2.5).

## Serve Cosmos3

```bash
SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rbln sglang serve \
  --model-path nvidia/Cosmos3-Edge --num-gpus 1
```

Text-to-image goes through `POST /v1/images/generations`; see the
[cookbook](https://docs.sglang.io/cookbook/diffusion/Cosmos/Cosmos3). Guardrails
need `cosmos-guardrail`; `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1` turns them off.

## Layout

```
sglang_rbln/
  __init__.py          activate() for the `rbln` platform entry point
  platform.py          RblnPlatform: identity, execution mode, compile seams, device/memory queries
  compilation.py       the `rbln` torch.compile backend and its options
  attention.py         torch SDPA with key/value heads expanded for rebel-compiler
  device.py            import-free NPU presence probe; execution mode; rebel and host queries
  hooks/__init__.py    register() for the `rbln_hooks` plugin entry point
  hooks/triton_norm.py keeps the Triton norm kernels out of the import path
  hooks/vsa_import.py  makes `import vsa` fail cleanly without a CUDA device
  hooks/cosmos3_compile.py compiles Cosmos3 GEN layers through the platform backend
tests/                 host-runnable; device-mode and NPU cases are strict-xfail elsewhere
tests/e2e/             `sglang serve` round trip on a public tiny model (`pytest -m e2e`)
```

## Rules for new code

- Prefer a `Platform` seam or `CustomOp.register_oot_forward` over a hook.
- Every hook carries Why / How (type, target) / Upstream PR or issue / Remove when.
- `activate()` runs in every spawned process: never import `rebel` or
  `torch_rbln` from it.

## Test

```bash
pytest            # unit
pytest -m e2e     # sglang serve round trip
```

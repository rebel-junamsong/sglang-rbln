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

`torch.compile` and `--regional-compile` are on by default on this platform
(pass either flag explicitly to turn it off). SGLang hands each DiT block to
the `rbln` backend, which compiles its FX graph with rebel-compiler and runs it
on the NPU; shapes are static per block and inputs and outputs stay CPU
tensors in `host` mode. In `host` mode the audio vocoder is loaded in fp32
(`--component-precisions.vocoder` overrides it), since SGLang runs it without
autocast on CPU tensors.

| Variable | Effect |
|---|---|
| `SGLANG_RBLN_CACHE_DIR` | compiled-graph cache (default `~/.cache/sglang_rbln`) |
| `SGLANG_RBLN_DISABLE_COMPILE_CACHE=1` | compile without the cache |
| `SGLANG_RBLN_COMPILE_ONLY=1` | compile the graphs and stop before running them |
| `SGLANG_RBLN_COMPILE_STRICT=1` | fail the request on a graph rebel-compiler cannot compile instead of running it eagerly on the host |

## Select the platform

| Select with | Platform class | Notes |
|---|---|---|
| unset | `sglang_rbln.platform:RBLNPlatform` | Auto-detected when `/dev/rbln*` exists and `rebel` is installed, and no other plugin answers |
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

The model card's SGLang example
([nvidia/Cosmos3-Super](https://huggingface.co/nvidia/Cosmos3-Super)) runs
unchanged apart from the device count:

```bash
sglang serve \
  --model-path nvidia/Cosmos3-Super \
  --num-gpus 1
```

```bash
job_id=$(curl -sS -X POST http://localhost:30000/v1/videos \
  --form-string "prompt=A small warehouse robot moves a blue box across a clean floor." \
  --form-string "negative_prompt=blurry, distorted, low quality" \
  --form-string "size=1280x720" \
  --form-string "num_frames=81" \
  --form-string "fps=24" \
  --form-string "num_inference_steps=35" \
  --form-string "guidance_scale=4.0" \
  --form-string "flow_shift=10.0" \
  --form-string "seed=42" \
  --form-string 'extra_params={"guardrails":true,"use_resolution_template":false,"use_duration_template":false}' \
  | python -c 'import json, sys; print(json.load(sys.stdin)["id"])')
while true; do
  status=$(curl -sS "http://localhost:30000/v1/videos/${job_id}" \
  | python -c 'import json, sys; print(json.load(sys.stdin)["status"])')
  [ "$status" = "completed" ] && break
  [ "$status" = "failed" ] && exit 1
  sleep 1
done
curl -sS -L "http://localhost:30000/v1/videos/${job_id}/content" \
  -o cosmos3_super_t2v_output.mp4
```

`nvidia/Cosmos3-Edge` (4B) serves the same way and is the quick check.
Guardrails need `cosmos-guardrail`; `SGLANG_DISABLE_COSMOS3_GUARDRAILS=1`
turns them off (then pass `"guardrails":false`). Text-to-image goes through
`POST /v1/images/generations`.

## Layout

```
sglang_rbln/
  __init__.py          activate() for the `rbln` platform entry point
  platform.py          RBLNPlatform: identity, execution mode, compile seams, device/memory queries
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

## Hooks are tracked debt

sglang is pinned exactly and every hook works around one upstream line;
`tests/test_hook_inventory.py` fails when a bump removes that line, which is
the signal to delete the hook.

| Hook | Upstream line | Upstream fix |
|---|---|---|
| `triton_norm` | `layernorm.py` imports the Triton norms on every non-CPU platform | gate the import on `has_triton()` |
| `vsa_import` | `video_sparse_attn.py` catches only `ImportError` around `import vsa` | also catch the CUDA `RuntimeError` |
| `cosmos3_compile` | `cosmos3.py` compiles GEN layers with hard-coded inductor kwargs | build them through `resolve_torch_compile_kwargs` |

Everything else goes through `Platform` seams (`get_compile_backend`,
`get_compile_options`, `get_attn_backend_cls_str`, `apply_server_args_defaults`)
and survives upstream changes.

Compiler gaps follow the same rule: the model side carries a workaround so the
model runs on the NPU today, the gap is pinned down by a test, and the fix lands
in rebel-compiler. `tests/test_compiler_inventory.py` fails once the installed
compiler no longer needs the workaround.

| Workaround | Compiler gap | Fix |
|---|---|---|
| `attention.py` expands key/value heads before SDPA | the SDPA converter ignores `enable_gqa`, so grouped-query attention fails to trace (`BatchDot: batch dimensions don't match`) and the graph runs eagerly on the host | [rebel_compiler#14989](https://github.com/rebellions-sw/rebel_compiler/pull/14989) |

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

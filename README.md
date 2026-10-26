# sglang-rbln

Out-of-tree RBLN NPU platform plugin for [SGLang Diffusion](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen),
backed by torch-rbln. First target: NVIDIA Cosmos3 (`Cosmos3OmniPipeline`).

The platform is a `Platform` subclass selected through an entry point;
everything the platform seams cannot express goes through tracked hooks.

## Install

Install the RBLN driver first and confirm `rbln-smi` sees the NPUs
([installation guide](https://docs.rbln.ai/latest/getting_started/installation_guide.html)).
`rebel-compiler` comes from the RBLN index, which needs an RBLN Portal account:

```bash
pip install --extra-index-url https://pypi.rbln.ai/simple -e '.[test]'
```

Pins: `sglang[diffusion]==0.5.22`, `rebel-compiler==0.12.0`.

The `rbln` torch device needs `torch-rbln` (`.[device]`, installed with
`--extra-index-url https://download.pytorch.org/whl/cpu` per the guide). The
published build (0.1.8) requires `torch==2.9.1+cpu` while sglang 0.5.22
requires `torch==2.14.1`, so the two do not install together yet; the device
tests are strict-xfail until a torch-rbln built against sglang's torch exists.

## Select the platform

| Select with | Platform class | Notes |
|---|---|---|
| unset | `sglang_rbln.platform:RblnPlatform` | Auto-detected when `/dev/rbln*` exists and `rebel` is installed, and no other plugin answers |
| `SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rbln` | same | Required when several platform plugins are installed |

On a host without an NPU, `RBLN_FORCE_NPU_NAME=RBLN-CA25` makes the platform
activate and names the compile target.

```bash
SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rbln sglang generate --model-path nvidia/Cosmos3-Nano ...
```

## Layout

```
sglang_rbln/
  __init__.py      activate() for the `rbln` platform entry point
  platform.py      RblnPlatform: identity, device/memory queries, subsystem choices
  device.py        import-free NPU presence probe; lazy rebel queries
  hooks/           register() for the `rbln_hooks` plugin entry point
tests/             host-runnable; NPU and torch-rbln cases are strict-xfail off-host
```

## Rules for new code

- Prefer a `Platform` seam or `CustomOp.register_oot_forward` over a hook.
- Every hook carries Why / How (type, target) / Upstream PR or issue / Remove when.
- `activate()` runs in every spawned process: never import `rebel` or
  `torch_rbln` from it.

## Test

```bash
pytest
```

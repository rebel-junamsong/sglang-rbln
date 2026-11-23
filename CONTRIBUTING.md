# Contributing

## Set up

```bash
uv sync --frozen --extra test && source .venv/bin/activate
uvx pre-commit install
```

The committed `uv.lock` pins the `rebel-compiler` and `torch-rbln` builds the
results are measured with (see the README's Install section).

## Check

```bash
uvx pre-commit run --all-files
pytest
pytest -m e2e
```

On a host without an NPU, set `RBLN_FORCE_NPU_NAME` to the NPU the compiler
should target. Tests marked `npu` are strict `xfail` where the
host has no NPU, and run for real otherwise.

## Rules

- Comments and docstrings state an invariant or why the code is the way it is
  (the upstream behaviour it works around), not a narrative of what the code
  does or how it got there.
- Prefer a `Platform` seam or `CustomOp.register_oot_forward` over a hook.
  A hook works around exactly one upstream line and comes with a row in
  `tests/test_patch_inventory.py`, so a bump that removes the line fails the
  test and the hook gets deleted.
- A patch lives at the path of the upstream module it patches under
  `sglang_rbln/patches/`, exposes `register()`, and is called from
  `sglang_rbln.patches.register`.
- A workaround for a compiler gap comes with a test that fails once the
  compiler no longer needs it.
- Environment variables are read through `sglang_rbln.envs`; every
  `SGLANG_RBLN_*` one a user sets has a row in the README configuration table.
- `activate()` runs in every spawned process: never import `rebel` or `torch`
  there.
- Every change ships its test.

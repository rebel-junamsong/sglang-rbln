# Contributing

## Set up

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.rbln.ai/simple -e '.[test]'
uv pip install flashinfer-python==0.7.1
pre-commit install
```

Add the `device` extra to run the `device` execution-mode tests.

## Check

```bash
ruff check sglang_rbln tests && ruff format --check sglang_rbln tests
pytest
pytest -m e2e
```

On a host without an NPU, set `RBLN_FORCE_NPU_NAME` to the NPU the compiler
should target. Tests marked `npu` or `device_mode` are strict `xfail` where the
host has no NPU or the execution mode is `host`, and run for real otherwise.

## Rules

- No comments or docstrings in code; the README and the tests carry the
  explanations.
- Prefer a `Platform` seam or `CustomOp.register_oot_forward` over a hook.
  A hook works around exactly one upstream line and comes with a row in
  `tests/test_patch_inventory.py`, so a bump that removes the line fails the
  test and the hook gets deleted.
- A patch lives at the path of the upstream module it patches under
  `sglang_rbln/patches/`, exposes `register()`, and is called from
  `sglang_rbln.patches.register`.
- A workaround for a compiler gap comes with a row in
  `tests/test_compiler_inventory.py` and a link to the compiler fix.
- Environment variables are read through `sglang_rbln.envs`; every
  `SGLANG_RBLN_*` one has a row in the README configuration table.
- `activate()` runs in every spawned process: never import `rebel` or `torch`
  there.
- Every change ships its test.

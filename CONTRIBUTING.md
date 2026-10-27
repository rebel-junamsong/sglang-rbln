# Contributing

## Set up

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.rbln.ai/simple -e '.[test]'
pre-commit install
```

`rebel-compiler` comes from the RBLN package index; the unit tests need it
installed, and the cases marked `xfail` on a host without an NPU run for real
on one.

## Check

```bash
ruff check sglang_rbln tests && ruff format --check sglang_rbln tests
pytest
pytest -m e2e
```

## Rules

- No comments or docstrings in code; the README and the tests carry the
  explanations.
- Prefer a `Platform` seam or `CustomOp.register_oot_forward` over a hook.
  A hook works around exactly one upstream line and comes with a row in
  `tests/test_hook_inventory.py`, so a bump that removes the line fails the
  test and the hook gets deleted.
- A workaround for a compiler gap comes with a row in
  `tests/test_compiler_inventory.py` and a link to the compiler fix.
- `activate()` runs in every spawned process: never import `rebel` or `torch`
  there.
- Every change ships its test.

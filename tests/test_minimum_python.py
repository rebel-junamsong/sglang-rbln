# SPDX-License-Identifier: Apache-2.0
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_lint_checks_the_declared_minimum_python():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    minimum = re.match(r">=(\d+)\.(\d+)", project["project"]["requires-python"])
    major, minor = minimum.groups()
    workflow = (ROOT / ".github/workflows/lint.yml").read_text()
    assert project["tool"]["ruff"]["target-version"] == f"py{major}{minor}"
    assert f"--target={major}.{minor}-" in workflow

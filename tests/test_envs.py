# SPDX-License-Identifier: Apache-2.0
import re
from pathlib import Path

import pytest

from sglang_rbln import envs


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), (" YES ", True), ("0", False), ("", False)],
)
def test_flags_accept_truthy_spellings(monkeypatch, value, expected):
    monkeypatch.setenv("SGLANG_RBLN_LOG_GRAPHS", value)
    assert envs.SGLANG_RBLN_LOG_GRAPHS is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), ("", True), ("1", True), ("0", False), (" false ", False)],
    ids=["unset", "empty", "on", "off", "spelled-off"],
)
def test_compile_strict_is_on_unless_turned_off(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("SGLANG_RBLN_COMPILE_STRICT", raising=False)
    else:
        monkeypatch.setenv("SGLANG_RBLN_COMPILE_STRICT", value)
    assert envs.SGLANG_RBLN_COMPILE_STRICT is expected


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("SGLANG_RBLN_NUM_DEVICES", "4", 4),
        ("SGLANG_RBLN_DUMP_GRAPHS", "/tmp/graphs", Path("/tmp/graphs")),
        ("SGLANG_RBLN_EXECUTION_MODE", " Device ", "device"),
        ("SGLANG_RBLN_PROFILE_DIR", "/tmp/trace", Path("/tmp/trace")),
        ("SGLANG_RBLN_PROFILE_REQUEST", "5", 5),
    ],
)
def test_values_are_parsed(monkeypatch, name, value, expected):
    monkeypatch.setenv(name, value)
    assert getattr(envs, name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("RBLN_FORCE_NPU_NAME", None),
        ("SGLANG_RBLN_NUM_DEVICES", None),
        ("SGLANG_RBLN_DUMP_GRAPHS", None),
        ("SGLANG_RBLN_EXECUTION_MODE", "auto"),
        ("SGLANG_RBLN_PROFILE_DIR", None),
        ("SGLANG_RBLN_PROFILE_REQUEST", 3),
        ("SGLANG_RBLN_BLOCKS_PER_REGION", None),
        ("SGLANG_RBLN_CACHE_DIR", Path.home() / ".cache" / "sglang_rbln"),
    ],
)
def test_defaults(monkeypatch, name, expected):
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) == expected


def test_unknown_names_raise():
    with pytest.raises(AttributeError, match="SGLANG_RBLN_UNKNOWN"):
        _ = envs.SGLANG_RBLN_UNKNOWN


def test_readme_documents_every_plugin_variable():
    readme = Path(__file__).parents[1] / "README.md"
    documented = re.findall(r"^\| `(SGLANG_RBLN_\w+)", readme.read_text(), re.M)
    assert sorted(documented) == sorted(
        name for name in envs.environment_variables if name.startswith("SGLANG_RBLN_")
    )

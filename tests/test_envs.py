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
    monkeypatch.setenv("SGLANG_RBLN_COMPILE_ONLY", value)
    assert envs.SGLANG_RBLN_COMPILE_ONLY is expected


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("SGLANG_RBLN_NUM_DEVICES", "4", 4),
        (
            "SGLANG_RBLN_COMPONENT_DEVICES",
            "text_encoder=4,5,6,7; vae=8;",
            {"text_encoder": (4, 5, 6, 7), "vae": (8,)},
        ),
        ("SGLANG_RBLN_LTX2_DETAILING_LORA", "/tmp/lora", Path("/tmp/lora")),
        ("SGLANG_RBLN_PROFILE_DIR", "/tmp/trace", Path("/tmp/trace")),
        ("SGLANG_RBLN_PROFILE_REQUEST", "5", 5),
    ],
    ids=[
        "num-devices",
        "component-devices",
        "detailing-lora",
        "profile-dir",
        "profile-request",
    ],
)
def test_values_parse_to_their_types(monkeypatch, name, value, expected):
    monkeypatch.setenv(name, value)
    assert getattr(envs, name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("RBLN_FORCE_NPU_NAME", None),
        ("SGLANG_RBLN_NUM_DEVICES", None),
        ("SGLANG_RBLN_COMPONENT_DEVICES", {}),
        ("SGLANG_RBLN_LTX2_DETAILING_LORA", None),
        ("SGLANG_RBLN_PROFILE_DIR", None),
        ("SGLANG_RBLN_PROFILE_REQUEST", 3),
        ("SGLANG_RBLN_BLOCKS_PER_REGION", None),
        ("SGLANG_RBLN_CACHE_DIR", Path.home() / ".cache" / "sglang_rbln"),
    ],
    ids=[
        "force-npu-name",
        "num-devices",
        "component-devices",
        "detailing-lora",
        "profile-dir",
        "profile-request",
        "blocks-per-region",
        "cache-dir",
    ],
)
def test_unset_variables_take_their_defaults(monkeypatch, name, expected):
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) == expected


def test_unknown_names_raise():
    with pytest.raises(AttributeError, match="SGLANG_RBLN_UNKNOWN"):
        _ = envs.SGLANG_RBLN_UNKNOWN


# Set by the platform from --tp-size for the worker processes, not by users.
INTERNAL_VARIABLES = {"SGLANG_RBLN_NUM_DEVICES"}


def test_readme_documents_every_user_facing_variable():
    readme = Path(__file__).parents[1] / "README.md"
    documented = re.findall(r"^\| `(SGLANG_RBLN_\w+)", readme.read_text(), re.M)
    assert sorted(documented) == sorted(
        name
        for name in envs.environment_variables
        if name.startswith("SGLANG_RBLN_") and name not in INTERNAL_VARIABLES
    )


def _declared_bool_defaults() -> dict[str, bool]:
    """The `TYPE_CHECKING` block's bool defaults, read from the source.

    They are annotations, so they do not exist at runtime: the value a caller
    actually gets comes from `environment_variables`. The two are written
    separately, and only the second one is read.
    """
    import ast

    source = Path(envs.__file__).read_text()
    block = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING"
    )
    return {
        node.target.id: node.value.value
        for node in block.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, bool)
    }


@pytest.mark.parametrize(
    ("name", "declared"), sorted(_declared_bool_defaults().items())
)
def test_the_declared_default_is_the_one_a_caller_gets(monkeypatch, name, declared):
    monkeypatch.delenv(name, raising=False)
    assert getattr(envs, name) is declared

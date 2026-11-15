# SPDX-License-Identifier: Apache-2.0
"""Every dtype cast in the plugin, each with the upstream code it mirrors.

The modeling keeps upstream's dtypes. A new
``.float()``/``.half()``/``.bfloat16()``/``.double()`` or ``.to(<dtype>)`` fails
here until it is either removed or listed with the upstream cast it mirrors.
"""

from __future__ import annotations

import ast
from pathlib import Path

import sglang_rbln

PACKAGE = Path(sglang_rbln.__file__).parent
DTYPE_METHODS = {"float", "half", "bfloat16", "double"}
TORCH_DTYPES = {
    "float32",
    "float16",
    "bfloat16",
    "float64",
    "half",
    "float",
    "double",
    "long",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "bool",
}

ALLOWED = {
    (
        "patches/entrypoints/prompt_enhancement.py",
        "greedy_decode",
        "logits.float()",
    ): "generate takes the softcapped logits to float32 before choosing",
    (
        "patches/models/ltx_2.py",
        "_blend",
        "(1 - ramp).to(torch.float32)",
    ): "upstream's blend weights are Python floats",
    (
        "patches/models/ltx_2.py",
        "_blend",
        "ramp.to(torch.float32)",
    ): "upstream's blend weights are Python floats",
    (
        "patches/stages/ltx_2/detailing.py",
        "append_reference",
        "reference[0].to(device=latents.device, dtype=latents.dtype)",
    ): "reference tokens join the latent sequence in the latents' dtype",
    (
        "patches/stages/ltx_2/detailing.py",
        "append_reference",
        "tokens.to(clean.dtype)",
    ): "reference tokens join the clean latents in their dtype",
    (
        "patches/stages/ltx_2/upsampling.py",
        "upsample_video_latents.build",
        "self.vae.latents_mean.view(shape).to(latents.dtype)",
    ): "upstream denormalizes with the statistics in the latents' dtype",
    (
        "patches/stages/ltx_2/upsampling.py",
        "upsample_video_latents.build",
        "self.vae.latents_std.view(shape).to(latents.dtype)",
    ): "upstream denormalizes with the statistics in the latents' dtype",
    (
        "patches/torch_rbln/autocast.py",
        "_cast.cast",
        "value.to(dtype)",
    ): "the autocast policy's own cast, CUDA autocast's for rbln tensors",
}


def _names_a_dtype(node: ast.expr) -> bool:
    source = ast.unparse(node)
    return "dtype" in source or source in {f"torch.{name}" for name in TORCH_DTYPES}


def _is_cast(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Attribute):
        return False
    if call.func.attr in DTYPE_METHODS:
        return not call.args and not call.keywords
    return call.func.attr == "to" and (
        any(map(_names_a_dtype, call.args))
        or any(keyword.arg == "dtype" for keyword in call.keywords)
    )


def _casts(path: Path):
    scopes: list[str] = []

    def visit(node: ast.AST):
        scoped = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        if scoped:
            scopes.append(node.name)
        if isinstance(node, ast.Call) and _is_cast(node):
            yield ".".join(scopes), ast.unparse(node)
        for child in ast.iter_child_nodes(node):
            yield from visit(child)
        if scoped:
            scopes.pop()

    yield from visit(ast.parse(path.read_text()))


def test_every_cast_is_listed_with_its_reason():
    found = {
        (path.relative_to(PACKAGE).as_posix(), scope, source)
        for path in sorted(PACKAGE.rglob("*.py"))
        for scope, source in _casts(path)
    }
    assert found - ALLOWED.keys() == set(), "new dtype casts in the plugin"
    assert ALLOWED.keys() - found == set(), "listed casts that are gone"

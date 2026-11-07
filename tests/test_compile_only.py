# SPDX-License-Identifier: Apache-2.0
"""A compile-only run builds the artifacts a device run then loads from the cache.

Each case compiles one small graph in a fresh process on rbln's dummy device, the way
the plugin compiles a region, and reports the artifact it wrote.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("torch_rbln")

COMPILE_ONE_GRAPH = textwrap.dedent(
    """
    import sys

    import torch

    from sglang_rbln.compilation import backend

    if sys.argv[1] == "run":
        # A dummy device refuses to execute; the artifact is written before that.
        import torch_rbln._internal.torch_compile_patch_helpers as helpers

        helpers.raise_if_dummy_execution = lambda *args, **kwargs: None
    backend.ensure_compile_backend()
    device = sys.argv[2]
    torch.manual_seed(0)
    module = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU()).to(device)
    module.compile(**backend.static_compile_kwargs(module))
    with torch.no_grad():
        module(torch.randn(1, 64).to(device))
    """
)


def compile_one_graph(tmp_path: Path, mode: str, device: str) -> list[str]:
    cache = tmp_path / f"{mode}-{device}"
    env = {
        **os.environ,
        "RBLN_DUMMY_DEVICE": "1",
        "RBLN_FORCE_NPU_NAME": os.environ.get("RBLN_FORCE_NPU_NAME") or "RBLN-CA22",
        "SGLANG_RBLN_CACHE_DIR": str(cache),
        "SGLANG_RBLN_COMPILE_ONLY": "1" if mode == "compile_only" else "0",
    }
    subprocess.run(
        [sys.executable, "-c", COMPILE_ONE_GRAPH, mode, device],
        env=env,
        check=True,
        capture_output=True,
        timeout=600,
    )
    return sorted(path.name for path in cache.glob("*.rbln"))


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory) -> dict[tuple[str, str], list[str]]:
    tmp_path = tmp_path_factory.mktemp("compile_only")
    cases = [("compile_only", "rbln"), ("run", "rbln"), ("compile_only", "cpu")]
    return {case: compile_one_graph(tmp_path, *case) for case in cases}


def test_compile_only_writes_the_artifact_a_device_run_loads(artifacts):
    compiled_only = artifacts["compile_only", "rbln"]
    assert compiled_only
    assert compiled_only == artifacts["run", "rbln"]


def test_a_host_compile_is_keyed_apart_from_the_device_run(artifacts):
    """Host inputs give host outputs, with no device views, so another artifact."""
    assert artifacts["compile_only", "cpu"] != artifacts["run", "rbln"]

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.compilation import placement, residency
from sglang_rbln.patches.models import ltx_2_text_features as features

TOKENS = 16


@pytest.mark.parametrize(
    ("placed", "given", "expected"),
    [
        ((4, 5, 6, 7), None, torch.device("rbln", 1)),
        ((4, 5, 6, 7), "cpu", "cpu"),
        (None, None, None),
    ],
    ids=["placed", "explicit", "unplaced"],
)
def test_text_features_are_returned_on_the_connectors_npus(
    monkeypatch, placed, given, expected
):
    # Upstream returns them on the platform device, the host: 367 MiB down, and up
    # again into each connector.
    monkeypatch.setattr(placement, "_components", {"connectors": placed})
    monkeypatch.setattr(residency, "logical_device_on", lambda devices: 1)
    seen = []

    def upstream(stage, text, server_args, **kwargs):
        seen.append(kwargs.get("device"))

    features.encode_text(upstream, None, "a yak", None, device=given)
    assert seen == [expected]


@pytest.mark.parametrize("target", features.HOOKS)
def test_wraps_text_encoding_and_packing(target):
    assert target in features.register()
    assert any(
        hook is features.HOOKS[target] and kind is HookType.AROUND
        for kind, hook, _ in HookRegistry._hooks[target]
    )


LTX25_CONFIG = SimpleNamespace(
    dit_config=SimpleNamespace(
        arch_config=SimpleNamespace(caption_proj_before_connector=True)
    )
)
LAYERS = 4  # Gemma's hidden states: the embedding output and one per layer


def _gemma_postprocess():
    from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
        _gemma_postprocess_func,
    )

    # the hook wraps the module's function; the graph runs the one underneath it
    return getattr(_gemma_postprocess_func, "__wrapped__", _gemma_postprocess_func)


def test_host_hidden_states_are_postprocessed_by_upstream_as_is():
    seen = []

    def upstream(outputs, text_inputs, pipeline_config=None):
        seen.append({state.device.type for state in outputs.hidden_states})
        return "packed"

    outputs = SimpleNamespace(hidden_states=(torch.zeros(1, TOKENS, 8),) * 3)
    text_inputs = {"attention_mask": torch.ones(1, TOKENS)}
    assert features.postprocess_text_features(upstream, outputs, text_inputs) == (
        "packed"
    )
    assert seen == [{"cpu"}]
    assert features._postprocess_graphs == {}


def test_resident_hidden_states_are_postprocessed_in_one_graph_on_the_device(
    dummy_device, monkeypatch, tmp_path
):
    """Eager, the stack of the encoder's hidden states would gather each one to a
    single NPU; as one graph of upstream's code it reads them where they are."""
    import rebel

    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(features, "_postprocess_graphs", {})
    compile_module = torch.nn.Module.compile

    def compile_resident(module, *args, **kwargs):
        # as the server's residency hook does: the call takes its inputs to the NPUs
        compile_module(module, *args, **kwargs)
        residency.install_resident_call(module)

    monkeypatch.setattr(torch.nn.Module, "compile", compile_resident)
    # the embedding output comes from the host, every layer's from the NPUs
    hidden_states = (
        torch.randn(1, 128, 64, dtype=torch.bfloat16),
        *(
            torch.randn(1, 128, 64, dtype=torch.bfloat16).to("rbln:0")
            for _ in range(LAYERS)
        ),
    )
    outputs = SimpleNamespace(hidden_states=hidden_states)
    text_inputs = {"attention_mask": torch.ones(1, 128, dtype=torch.long)}

    packed = [
        features.postprocess_text_features(
            _gemma_postprocess(), outputs, text_inputs, LTX25_CONFIG
        )
        for _ in range(2)
    ]
    assert len(features._postprocess_graphs) == 1
    assert {(out.device.type, tuple(out.shape)) for out in packed} == {
        ("rbln", (1, 128, 64 * (LAYERS + 1)))
    }
    (artifact,) = tmp_path.rglob("*.rbln")
    stats = rebel.RBLNCompiledModel.inspect(str(artifact))["subgraph"]["stats"]
    assert [stat["device_func_count"] for stat in stats] == [1]
    host_ops = {
        op
        for stat in stats
        for ops in stat["host_op_reason"].values()
        for op in ops
        if not op.startswith("args_")
    }
    assert host_ops == set()


CROSS_SET_PROBE = """
import json, os, sys, tempfile
from types import SimpleNamespace
os.environ["SGLANG_RBLN_CACHE_DIR"] = tempfile.mkdtemp()
import torch, torch_rbln  # noqa: F401
import sglang.multimodal_gen.runtime.entrypoints.cli.serve  # noqa: F401
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import _gemma_postprocess_func
from sglang_rbln.compilation import backend, residency
from sglang_rbln.patches.models import ltx_2_text_features as features

backend.ensure_compile_backend()
residency.register()
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry
HookRegistry.apply_hooks()

crossings = []
index = lambda device: torch.device(device).index or 0
original_to, original_copy = torch.Tensor.to, torch.Tensor.copy_


def to(self, *args, **kwargs):
    target = kwargs.get("device", args[0] if args else None)
    if (
        isinstance(target, (str, torch.device))
        and torch.device(target).type == "rbln"
        and self.device.type == "rbln"
        and index(target) != index(self.device)
    ):
        crossings.append(("to", str(self.device), str(target)))
    return original_to(self, *args, **kwargs)


def copy_(self, src, *args, **kwargs):
    if self.device.type == src.device.type == "rbln" and index(self) != index(src):
        crossings.append(("copy_", str(src.device), str(self.device)))
    return original_copy(self, src, *args, **kwargs)


torch.Tensor.to, torch.Tensor.copy_ = to, copy_
original_across = residency.copy_across_devices
residency.copy_across_devices = lambda tensor, device, *a, **k: (
    crossings.append(("across", str(tensor.device), str(device)))
    or original_across(tensor, device, *a, **k)
)

config = SimpleNamespace(
    dit_config=SimpleNamespace(
        arch_config=SimpleNamespace(caption_proj_before_connector=True)
    )
)
states = (torch.randn(1, 128, 64, dtype=torch.bfloat16),) + tuple(
    torch.randn(1, 128, 64, dtype=torch.bfloat16).to("rbln:1") for _ in range(4)
)
outputs = SimpleNamespace(hidden_states=states)
mask = {"attention_mask": torch.ones(1, 128, dtype=torch.long)}
with residency.DeviceBoundary():
    packed = features.postprocess_text_features(
        _gemma_postprocess_func, outputs, mask, config
    )
print(json.dumps({"device": str(packed.device), "crossings": crossings}))
"""


def test_hidden_states_on_the_encoders_npus_never_cross_to_another_set(
    rbln_platform_env,
):
    env = {
        **os.environ,
        **rbln_platform_env,
        "RBLN_DUMMY_DEVICE": "1",
        "RBLN_DEVICE_MAP": "[0],[1]",
        "PYTHONWARNINGS": "ignore",
    }
    completed = subprocess.run(
        [sys.executable, "-c", CROSS_SET_PROBE],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    (line,) = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    result = json.loads(line)
    assert result == {"device": "rbln:1", "crossings": []}

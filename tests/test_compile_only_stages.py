# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
import torch
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.patches.stages import compile_only

RUN_VERIFICATION = PipelineStage._run_verification


def _uninitialised_embedding() -> VerificationResult:
    result = VerificationResult()
    embedding = torch.full((1, 4, 8), float("nan"))
    result.add_check("prompt_embeds", [embedding], StageValidators.list_of_tensors)
    return result


def test_upstream_rejects_a_compile_only_stage_output():
    # A compile-only run leaves stage outputs uninitialised; this check would stop
    # it before the DiT compiles.
    with pytest.raises(Exception, match="prompt_embeds"):
        RUN_VERIFICATION(object(), _uninitialised_embedding(), "Stage", "input")


def test_compile_only_skips_stage_output_verification():
    result = _uninitialised_embedding()
    assert compile_only.skip_verification(object(), result, "Stage", "input") is None


def test_replaces_the_upstream_stage_verification():
    target = compile_only.register()
    assert any(
        hook is compile_only.skip_verification and kind is HookType.REPLACE
        for kind, hook, _ in HookRegistry._hooks[target]
    )

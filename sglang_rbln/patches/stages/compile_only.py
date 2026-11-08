# SPDX-License-Identifier: Apache-2.0
"""A compile-only run checks no stage's values.

Compile-only builds every graph and runs none, so each stage hands on outputs that hold
no values. The stages' own checks read those values (a NaN in the text embedding fails
the latent preparation), and a staged compile-only run stopped at the first of them,
before the DiT compiled.
"""

from __future__ import annotations

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.PipelineStage."
    "_run_verification"
)


def skip_verification(*args, **kwargs) -> None:
    return None


def register() -> str:
    HookRegistry.register(HOOK_TARGET, skip_verification, HookType.REPLACE)
    return HOOK_TARGET

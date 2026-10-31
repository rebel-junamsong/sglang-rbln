# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from typing import NamedTuple

import torch

DIT = "sglang.multimodal_gen.runtime.models.dits.flux3"
STAGE = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".flux3_action"
)


class Flux3SegmentState(NamedTuple):
    name: str
    hidden: torch.Tensor
    rope: torch.Tensor
    vec: torch.Tensor
    joint_mod: tuple[torch.Tensor, ...]

    @property
    def length(self) -> int:
        return self.hidden.shape[1]


def register() -> tuple[str, ...]:
    for module in (DIT, STAGE):
        importlib.import_module(module).Flux3SegmentState = Flux3SegmentState
    return DIT, STAGE

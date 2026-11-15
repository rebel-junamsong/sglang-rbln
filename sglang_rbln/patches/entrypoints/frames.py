# SPDX-License-Identifier: Apache-2.0
"""Convert upstream's float32 frames to uint8 with torch's multi-threaded host ops.

Same clamp, scale and truncating cast as upstream, so the same bytes; upstream's
single-threaded numpy took 0.8 s at 1280x704x121.
"""

from __future__ import annotations

import numpy as np
import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

HOOK_TARGET = "sglang.multimodal_gen.runtime.entrypoints.utils._sample_to_uint8_frames"


def sample_to_uint8_frames(convert, sample):
    if (
        getattr(sample, "dtype", None) != np.float32
        or sample.shape[-1:] not in [(1,), (3,), (4,)]
        or sample.ndim != 4
    ):
        return convert(sample)
    return list(torch.from_numpy(sample).clamp(0.0, 1.0).mul(255.0).byte().numpy())


def register() -> str:
    HookRegistry.register(HOOK_TARGET, sample_to_uint8_frames, HookType.AROUND)
    return HOOK_TARGET

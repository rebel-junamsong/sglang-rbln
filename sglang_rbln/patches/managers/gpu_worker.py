# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import logging

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import envs

logger = logging.getLogger(__name__)

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.managers.gpu_worker.GPUWorker.execute_forward"
)
request_counter = itertools.count(1)


def profile_request(execute_forward, worker, batch, *args, **kwargs):
    directory = envs.SGLANG_RBLN_PROFILE_DIR
    if (
        directory is None
        or any(req.is_warmup for req in batch)
        or next(request_counter) != envs.SGLANG_RBLN_PROFILE_REQUEST
    ):
        return execute_forward(worker, batch, *args, **kwargs)
    from rebel.profiler import profile

    directory.mkdir(parents=True, exist_ok=True)
    with profile(output_dir=str(directory)):
        result = execute_forward(worker, batch, *args, **kwargs)
    logger.info(
        "rbln profile of request %s written to %s", batch[0].request_id, directory
    )
    return result


def register() -> str:
    HookRegistry.register(HOOK_TARGET, profile_request, HookType.AROUND)
    return HOOK_TARGET

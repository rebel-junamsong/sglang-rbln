# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import itertools
import logging

from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import envs
from sglang_rbln.compilation import shapes

logger = logging.getLogger(__name__)

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.managers.gpu_worker.GPUWorker.execute_forward"
)
_request_counter = itertools.count(1)


def profile_request(execute_forward, worker, batch, *args, **kwargs):
    directory = envs.SGLANG_RBLN_PROFILE_DIR
    if (
        directory is None
        or any(req.is_warmup for req in batch)
        or next(_request_counter) != envs.SGLANG_RBLN_PROFILE_REQUEST
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


def key_request_shape(execute_forward, worker, batch, *args, **kwargs):
    token = shapes.current_shape_key.set(
        shapes.shape_key(worker.pipeline, worker.server_args.pipeline_config, batch[0])
    )
    try:
        return execute_forward(worker, batch, *args, **kwargs)
    finally:
        shapes.current_shape_key.reset(token)


def run_request(execute_forward, worker, batch, *args, **kwargs):
    return key_request_shape(
        functools.partial(profile_request, execute_forward),
        worker,
        batch,
        *args,
        **kwargs,
    )


def register() -> str:
    HookRegistry.register(HOOK_TARGET, run_request, HookType.AROUND)
    return HOOK_TARGET

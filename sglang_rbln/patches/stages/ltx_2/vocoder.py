# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from sglang_rbln import envs

logger = logging.getLogger(__name__)

COMPILED = "_rbln_vocoder_compiled"


def compile_vocoder(forward, stage, batch, server_args):
    from sglang.multimodal_gen.runtime.utils.torch_compile import (
        build_torch_compile_kwargs,
    )

    vocoder = stage.vocoder
    if (
        envs.SGLANG_RBLN_VOCODER_ON_NPU
        and server_args.enable_torch_compile
        and not getattr(vocoder, COMPILED, False)
    ):
        compile_kwargs = build_torch_compile_kwargs(mode=None, module=vocoder)
        compile_kwargs["dynamic"] = False
        if num_devices := compile_kwargs["options"].get("num_devices"):
            compile_kwargs["options"] |= {"num_devices": 1, "device": num_devices - 1}
        vocoder.compile(**compile_kwargs)
        setattr(vocoder, COMPILED, True)
        logger.info("Compiling the LTX-2 vocoder: %s", compile_kwargs)
    return forward(stage, batch, server_args)

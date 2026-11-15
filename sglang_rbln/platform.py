# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch
from sglang.multimodal_gen.runtime.platforms import PlatformEnum
from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform

from sglang_rbln import device, envs
from sglang_rbln.compilation import backend as compilation
from sglang_rbln.compilation import residency

NUM_DEVICES_ENV = "SGLANG_RBLN_NUM_DEVICES"
TORCH_RBLN_FALLBACK_ENV = "TORCH_RBLN_DISABLE_FALLBACK"
TORCH_RBLN_FALLBACK_DEFAULT = "compile_error"
# A compile-only run builds the artifacts a device run loads, without an NPU.
DUMMY_DEVICE_ENV = "RBLN_DUMMY_DEVICE"

compilation.ensure_compile_backend()


def take_tensor_parallel_degree(server_args) -> None:
    """Split each model over the NPUs `--tp-size` (else `--num-gpus`) asks for.

    One worker process drives all of a model's NPUs: the degree becomes
    `SGLANG_RBLN_NUM_DEVICES`, which the spawned workers inherit, and upstream's
    own degree goes back to 1.
    """
    degree = next(
        (
            value
            for flag in ("tp_size", "num_gpus")
            if server_args.is_arg_explicitly_set(flag)
            and (value := getattr(server_args, flag) or 1) > 1
        ),
        None,
    )
    if degree is None:
        return
    declared = envs.SGLANG_RBLN_NUM_DEVICES
    if declared not in (None, degree):
        raise ValueError(
            f"SGLANG_RBLN_NUM_DEVICES={declared} contradicts the requested "
            f"tensor-parallel degree {degree}"
        )
    os.environ[NUM_DEVICES_ENV] = str(degree)
    server_args.tp_size = 1
    server_args.num_gpus = 1


LAYERWISE_OFFLOAD_FLAGS = (
    "dit_layerwise_offload",
    "layerwise_offload_components",
    "offload_during_compile",
)


def reject_layerwise_offload(server_args) -> None:
    """Refuse the flags that layerwise-offload a model.

    Upstream's offload manager moves a model to `device_type`, the host here, so it
    takes the resident weights off their NPUs and nothing moves them back: every
    request after it fails with a device mismatch. The component offload flags
    (`--dit-cpu-offload` and the like) need no check: upstream keeps every component
    resident on a platform that reports `is_cpu()`.
    """
    requested = [
        "--" + flag.replace("_", "-")
        for flag in LAYERWISE_OFFLOAD_FLAGS
        if server_args.is_arg_explicitly_set(flag) and getattr(server_args, flag)
    ]
    if requested:
        raise ValueError(
            f"{', '.join(requested)}: layerwise offload moves the model's weights "
            "off its NPUs, which the rbln platform keeps them on"
        )


class RBLNPlatform(CpuPlatform):
    """Upstream sees the host: its tensors are cpu ones, and the plugin's compiled
    regions take them onto their NPU sets, where their outputs and the eager ops
    between the regions stay (`compilation.residency`). So every answer is the CPU
    platform's but the platform's identity, the compile backend and the device
    names."""

    _enum = PlatformEnum.OOT
    device_name = "rbln"
    simple_compile_backend = compilation.COMPILE_BACKEND_NAME

    def init_backend(self) -> None:
        os.environ.setdefault(TORCH_RBLN_FALLBACK_ENV, TORCH_RBLN_FALLBACK_DEFAULT)
        if envs.SGLANG_RBLN_COMPILE_ONLY:
            os.environ.setdefault(DUMMY_DEVICE_ENV, "1")
        residency.configure_device_group()

    def apply_server_args_defaults(self, server_args) -> None:
        take_tensor_parallel_degree(server_args)
        if not server_args.is_arg_explicitly_set("enable_torch_compile"):
            server_args.enable_torch_compile = True
        # The server warmup would layerwise-offload the DiT for its compile.
        if not server_args.is_arg_explicitly_set("offload_during_compile"):
            server_args.offload_during_compile = False
        reject_layerwise_offload(server_args)
        # Upstream's LTX2VocoderWithBWE.forward opens its float32 autocast region
        # only off the CPU, so on the host its bf16 weights meet a float32 input.
        server_args.component_precisions.setdefault("vocoder", "fp32")

    def get_compile_options(self, module: torch.nn.Module) -> dict[str, object]:
        return compilation.compile_options(module)

    def is_cpu(self) -> bool:
        return True

    def get_dispatch_key_name(self) -> str:
        return "rbln"

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device("cpu")

    @classmethod
    def set_device(cls, torch_device: torch.device) -> None:
        pass

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return device.npu_name(device_id)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return f"{device.npu_name(device_id)}:{device_id}"

    def tensor_on_device(self, tensor: torch.Tensor) -> bool:
        return tensor.device.type == self.device_type

    @classmethod
    def is_amp_supported(cls) -> bool:
        return True

    @classmethod
    def is_float64_supported(cls) -> bool:
        return True

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

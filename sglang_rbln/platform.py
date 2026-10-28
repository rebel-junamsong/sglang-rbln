# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.platforms import Platform, PlatformEnum
from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform

from sglang_rbln import device
from sglang_rbln.compilation import backend as compilation

ATTENTION_BACKEND_PATH = "sglang_rbln.layers.attention.backends.sdpa.RBLNSdpaBackend"
EXECUTION_MODE = device.execution_mode()
RUNS_ON_DEVICE = EXECUTION_MODE is device.ExecutionMode.DEVICE
compilation.ensure_compile_backend()


TORCH_RBLN_FALLBACK_ENV = "TORCH_RBLN_DISABLE_FALLBACK"
TORCH_RBLN_FALLBACK_DEFAULT = "compile_error"
OFFLOAD_FLAGS = (
    "dit_cpu_offload",
    "text_encoder_cpu_offload",
    "image_encoder_cpu_offload",
    "vae_cpu_offload",
)


class RBLNPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "rbln"
    device_type = "rbln" if RUNS_ON_DEVICE else "cpu"
    dispatch_key = "PrivateUse1" if RUNS_ON_DEVICE else "CPU"
    execution_mode = EXECUTION_MODE
    simple_compile_backend = compilation.COMPILE_BACKEND_NAME

    def init_backend(self) -> None:
        if not RUNS_ON_DEVICE:
            return
        os.environ.setdefault(TORCH_RBLN_FALLBACK_ENV, TORCH_RBLN_FALLBACK_DEFAULT)
        try:
            import torch_rbln  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "torch-rbln is required for execution mode 'device' "
                "(pip install 'sglang-rbln[device]')."
            ) from e

    def apply_server_args_defaults(self, server_args) -> None:
        if not server_args.is_arg_explicitly_set("enable_torch_compile"):
            server_args.enable_torch_compile = True
        if not server_args.is_arg_explicitly_set("regional_compile"):
            server_args.regional_compile = True
        if self.is_cpu():
            server_args.component_precisions.setdefault("vocoder", "fp32")
            return
        for flag in OFFLOAD_FLAGS:
            if (
                not server_args.is_arg_explicitly_set(flag)
                and getattr(server_args, flag) is None
            ):
                setattr(server_args, flag, False)

    def get_compile_options(self, module: torch.nn.Module) -> dict[str, object]:
        return compilation.compile_options(module)

    def is_cpu(self) -> bool:
        return not RUNS_ON_DEVICE

    def get_dispatch_key_name(self) -> str:
        return "rbln"

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        if RUNS_ON_DEVICE:
            return torch.device("rbln", envs.LOCAL_RANK)
        return torch.device("cpu")

    def get_device(self, local_rank: int) -> torch.device:
        if RUNS_ON_DEVICE:
            return torch.device("rbln", local_rank)
        return torch.device("cpu")

    @classmethod
    def set_device(cls, torch_device: torch.device) -> None:
        if RUNS_ON_DEVICE:
            torch.rbln.set_device(torch_device)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return device.npu_name(device_id)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return f"{device.npu_name(device_id)}:{device_id}"

    def tensor_on_device(self, tensor: torch.Tensor) -> bool:
        return tensor.device.type == self.device_type

    @classmethod
    def is_float64_supported(cls) -> bool:
        return not RUNS_ON_DEVICE

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return True

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return _memory_info(device_id).total

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        return float(_memory_info(_device_index(device)).used)

    @classmethod
    def get_available_gpu_memory(cls, device_id=None, *args, **kwargs) -> float:
        return _memory_info(_device_index(device_id)).free / (1 << 30)

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    def get_attn_backend_cls_str(self, *args, **kwargs) -> str:
        return ATTENTION_BACKEND_PATH

    def get_torch_distributed_backend_str(self) -> str:
        return "gloo"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        if RUNS_ON_DEVICE:
            return super().get_device_communicator_cls()
        return CpuPlatform.get_device_communicator_cls()

    @classmethod
    def enable_dit_layerwise_offload_by_default(cls) -> bool:
        return False


def _device_index(torch_device) -> int:
    if torch_device is None or isinstance(torch_device, int):
        return torch_device or 0
    return torch.device(torch_device).index or 0


def _memory_info(device_id: int) -> device.MemoryInfo:
    if RUNS_ON_DEVICE:
        return device.npu_memory_info(device_id)
    return device.host_memory_info()

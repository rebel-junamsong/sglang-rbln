# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import torch
from sglang.multimodal_gen import envs as sglang_envs
from sglang.multimodal_gen.runtime.platforms import Platform, PlatformEnum
from sglang.multimodal_gen.runtime.platforms.cpu import CpuPlatform
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum

from sglang_rbln import device, envs
from sglang_rbln.compilation import backend as compilation
from sglang_rbln.compilation import residency

EXECUTION_MODE = device.execution_mode()
NUM_DEVICES_ENV = "SGLANG_RBLN_NUM_DEVICES"
TORCH_RBLN_FALLBACK_ENV = "TORCH_RBLN_DISABLE_FALLBACK"
TORCH_RBLN_FALLBACK_DEFAULT = "compile_error"
# A compile-only run builds the artifacts a device run loads, so it compiles on device
# tensors as that run does: rbln's dummy device gives them without any NPU.
DUMMY_DEVICE_ENV = "RBLN_DUMMY_DEVICE"
OFFLOAD_FLAGS = (
    "dit_cpu_offload",
    "text_encoder_cpu_offload",
    "image_encoder_cpu_offload",
    "vae_cpu_offload",
)

compilation.ensure_compile_backend()


# Resolved on each use, not at import: the platform is imported before the server
# arguments are parsed, and `--tp-size` sets the NPUs per model (see
# `take_tensor_parallel_degree`).
def resident_regions() -> bool:
    return device.resident_regions()


def runs_on_device() -> bool:
    return EXECUTION_MODE is device.ExecutionMode.RESIDENT and not resident_regions()


def take_tensor_parallel_degree(server_args) -> None:
    """Split each model over the NPUs `--tp-size` (else `--num-gpus`) asks for.

    The compiler shards a graph over its NPUs, so one worker process drives them:
    the degree becomes `SGLANG_RBLN_NUM_DEVICES`, which the spawned workers inherit,
    and upstream's own degree goes back to 1.
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


class RBLNPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "rbln"
    execution_mode = EXECUTION_MODE
    simple_compile_backend = compilation.COMPILE_BACKEND_NAME

    @property
    def device_type(self) -> str:
        return "rbln" if runs_on_device() else "cpu"

    @property
    def dispatch_key(self) -> str:
        return "PrivateUse1" if runs_on_device() else "CPU"

    def init_backend(self) -> None:
        if not (runs_on_device() or resident_regions()):
            return
        os.environ.setdefault(TORCH_RBLN_FALLBACK_ENV, TORCH_RBLN_FALLBACK_DEFAULT)
        if envs.SGLANG_RBLN_COMPILE_ONLY:
            os.environ.setdefault(DUMMY_DEVICE_ENV, "1")
        if resident_regions():
            residency.configure_device_group()
            return
        try:
            import torch_rbln  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "torch-rbln is required for execution mode 'resident' "
                "(pip install torch-rbln)."
            ) from e

    def apply_server_args_defaults(self, server_args) -> None:
        resident_before = resident_regions()
        take_tensor_parallel_degree(server_args)
        if resident_regions() and not resident_before:
            from sglang_rbln import patches

            patches.register_resident(apply=True)
        if not server_args.is_arg_explicitly_set("enable_torch_compile"):
            server_args.enable_torch_compile = True
        if not server_args.is_arg_explicitly_set("regional_compile"):
            server_args.regional_compile = True
        if resident_regions() and not server_args.is_arg_explicitly_set(
            "disable_autocast"
        ):
            # Upstream opens autocast on the platform's device type, the host's here,
            # while resident regions take rbln tensors, which only rbln autocast casts.
            server_args.disable_autocast = True
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
        return not runs_on_device()

    def get_dispatch_key_name(self) -> str:
        return "rbln"

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        if runs_on_device():
            return torch.device("rbln", sglang_envs.LOCAL_RANK)
        return torch.device("cpu")

    def get_device(self, local_rank: int) -> torch.device:
        if runs_on_device():
            return torch.device("rbln", local_rank)
        return torch.device("cpu")

    @classmethod
    def set_device(cls, torch_device: torch.device) -> None:
        if runs_on_device():
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
    def is_amp_supported(cls) -> bool:
        return True

    @classmethod
    def is_float64_supported(cls) -> bool:
        return not runs_on_device()

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

    def get_attn_backend_cls_str(
        self,
        selected_backend: AttentionBackendEnum | None,
        head_size: int,
        dtype: torch.dtype,
    ) -> str:
        # `Platform`'s own returns "": take the CPU platform's SDPA selection.
        return CpuPlatform.get_attn_backend_cls_str(selected_backend, head_size, dtype)

    def get_torch_distributed_backend_str(self) -> str:
        return "gloo"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        if runs_on_device():
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
    if runs_on_device():
        return device.npu_memory_info(device_id)
    return device.host_memory_info()

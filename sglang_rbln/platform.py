# SPDX-License-Identifier: Apache-2.0
"""The out-of-tree RBLN platform (``SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rbln``)."""

from __future__ import annotations

import torch
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.platforms import Platform, PlatformEnum

from sglang_rbln import device

SDPA_BACKEND = (
    "sglang.multimodal_gen.runtime.layers.attention.backends.sdpa.SDPABackend"
)


class RblnPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name = "rbln"
    device_type = "rbln"
    dispatch_key = "PrivateUse1"

    def init_backend(self) -> None:
        try:
            import torch_rbln  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "torch-rbln is required for the rbln diffusion platform "
                "(pip install 'sglang-rbln[device]')."
            ) from e

    def get_dispatch_key_name(self) -> str:
        return "rbln"

    @classmethod
    def get_local_torch_device(cls) -> torch.device:
        return torch.device("rbln", envs.LOCAL_RANK)

    def get_device(self, local_rank: int) -> torch.device:
        return torch.device("rbln", local_rank)

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return device.npu_name(device_id)

    @classmethod
    def get_device_uuid(cls, device_id: int = 0) -> str:
        return f"{device.npu_name(device_id)}:{device_id}"

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return None

    def tensor_on_device(self, t: torch.Tensor) -> bool:
        return t.device.type == self.device_type

    @classmethod
    def is_float64_supported(cls) -> bool:
        return False

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        total, _, _ = device.memory_info(device_id)
        return total

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        _, used, _ = device_memory(device)
        return float(used)

    @classmethod
    def get_available_gpu_memory(cls, device=None, *args, **kwargs) -> float:
        _, _, free = device_memory(device)
        return free / (1 << 30)

    @classmethod
    def empty_cache(cls) -> None:
        pass

    @classmethod
    def synchronize(cls) -> None:
        pass

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    def get_attn_backend_cls_str(self, *args, **kwargs) -> str:
        return SDPA_BACKEND

    def get_torch_distributed_backend_str(self) -> str:
        return "gloo"

    @classmethod
    def enable_dit_layerwise_offload_by_default(cls) -> bool:
        return False


def device_memory(dev=None) -> tuple[int, int, int]:
    index = 0 if dev is None else (torch.device(dev).index or 0)
    return device.memory_info(index)

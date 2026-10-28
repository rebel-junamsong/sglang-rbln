# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from importlib import import_module

import torch
from torch.utils._pytree import tree_flatten

PATCHED_MODULES = (
    "torch_rbln._internal.ops_utils",
    "torch_rbln._internal.register_ops",
)
PATCHED_NAME = "is_type_promotion_allowed"


def _category(dtype: torch.dtype) -> int:
    if dtype is torch.bool:
        return 0
    if dtype.is_complex:
        return 3
    if dtype.is_floating_point:
        return 2
    return 1


def result_dtype(tensors: list[torch.Tensor]) -> torch.dtype:
    dimensioned = [tensor.dtype for tensor in tensors if tensor.dim()]
    zero_dim = [tensor.dtype for tensor in tensors if not tensor.dim()]
    if not dimensioned:
        return functools.reduce(torch.promote_types, zero_dim)
    dtype = functools.reduce(torch.promote_types, dimensioned)
    if zero_dim:
        zero_dim_dtype = functools.reduce(torch.promote_types, zero_dim)
        if _category(zero_dim_dtype) > _category(dtype):
            dtype = torch.promote_types(dtype, zero_dim_dtype)
    return dtype


def is_type_promotion_allowed(input_tensors, output_tensor: torch.Tensor) -> bool:
    tensors = [
        value
        for value in tree_flatten(input_tensors)[0]
        if isinstance(value, torch.Tensor)
    ]
    if not tensors:
        raise TypeError("No tensor inputs found in input_tensors")
    promoted = torch.promote_types(result_dtype(tensors), output_tensor.dtype)
    return promoted == output_tensor.dtype


def register() -> bool:
    for module_name in PATCHED_MODULES:
        setattr(import_module(module_name), PATCHED_NAME, is_type_promotion_allowed)
    return True

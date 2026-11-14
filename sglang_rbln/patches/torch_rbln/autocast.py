# SPDX-License-Identifier: Apache-2.0
"""Autocast for rbln tensors, with CUDA autocast's cast policy.

torch-rbln declares no AMP dtype, so ``torch.autocast("rbln")`` disables itself, and
registers no kernel at ``AutocastPrivateUse1``, so every op would fall through and a
bf16 module fed float32 inputs fails at its first conv or linear. These kernels cast as
``aten/src/ATen/autocast_mode.cpp`` does for CUDA: the lower-precision ops run in the
autocast dtype, the float32 ops in float32, and the promote ops in the widest
input dtype.
"""

from __future__ import annotations

import functools

import torch
import torch_rbln  # noqa: F401
from torch.utils._pytree import tree_flatten, tree_map

DEVICE_TYPE = "rbln"
DISPATCH_KEY = "AutocastPrivateUse1"
AMP_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

LOWER_PRECISION_FP = (
    "_convolution",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_tbc",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "convolution",
    "prelu",
    "addmm",
    "addmv",
    "addr",
    "matmul",
    "einsum",
    "mm",
    "mv",
    "linalg_vecdot",
    "linear",
    "addbmm",
    "baddbmm",
    "bmm",
    "chain_matmul",
    "linalg_multi_dot",
    "_thnn_fused_lstm_cell",
    "_thnn_fused_gru_cell",
    "lstm_cell",
    "gru_cell",
    "rnn_tanh_cell",
    "rnn_relu_cell",
    "_scaled_dot_product_flash_attention",
    "scaled_dot_product_attention",
)
FP32 = (
    "acos",
    "asin",
    "cosh",
    "erfinv",
    "exp",
    "expm1",
    "log",
    "log10",
    "log2",
    "log1p",
    "reciprocal",
    "rsqrt",
    "sinh",
    "tan",
    "pow",
    "softplus",
    "layer_norm",
    "native_layer_norm",
    "rms_norm",
    "group_norm",
    "frobenius_norm",
    "nuclear_norm",
    "cosine_similarity",
    "poisson_nll_loss",
    "cosine_embedding_loss",
    "nll_loss",
    "nll_loss2d",
    "hinge_embedding_loss",
    "kl_div",
    "l1_loss",
    "smooth_l1_loss",
    "huber_loss",
    "mse_loss",
    "margin_ranking_loss",
    "multilabel_margin_loss",
    "soft_margin_loss",
    "triplet_margin_loss",
    "multi_margin_loss",
    "binary_cross_entropy_with_logits",
    "dist",
    "pdist",
    "cdist",
    "renorm",
    "logsumexp",
    "upsample_nearest1d",
    "_upsample_nearest_exact1d",
    "upsample_nearest2d",
    "_upsample_nearest_exact2d",
    "upsample_nearest3d",
    "_upsample_nearest_exact3d",
    "upsample_linear1d",
    "upsample_bilinear2d",
    "_upsample_bilinear2d_aa",
    "upsample_trilinear3d",
    "upsample_bicubic2d",
    "_upsample_bicubic2d_aa",
)
FP32_SET_OPT_DTYPE = (
    "prod",
    "softmax",
    "log_softmax",
    "cumprod",
    "cumsum",
    "linalg_vector_norm",
    "linalg_matrix_norm",
    "sum",
)
PROMOTE = (
    "addcdiv",
    "addcmul",
    "atan2",
    "bilinear",
    "cross",
    "dot",
    "vdot",
    "grid_sampler",
    "index_put",
    "tensordot",
    "scatter_add",
)

_libraries: list[torch.library.Library] = []


def _eligible(value) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.device.type == DEVICE_TYPE
        and value.is_floating_point()
        and value.dtype is not torch.float64
    )


def _cast(dtype: torch.dtype, args, kwargs):
    def cast(value):
        return value.to(dtype) if _eligible(value) else value

    return tree_map(cast, args), tree_map(cast, kwargs)


def _without_autocast(op, args, kwargs):
    with torch._C._ExcludeDispatchKeyGuard(
        torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastPrivateUse1)
    ):
        return op(*args, **kwargs)


def _lower_precision(op, *args, **kwargs):
    dtype = torch.get_autocast_dtype(DEVICE_TYPE)
    return _without_autocast(op, *_cast(dtype, args, kwargs))


def _fp32(op, *args, **kwargs):
    return _without_autocast(op, *_cast(torch.float32, args, kwargs))


def _fp32_set_opt_dtype(op, *args, **kwargs):
    names = [argument.name for argument in op._schema.arguments]
    leaves, _ = tree_flatten((args, kwargs))
    if "dtype" in names and any(_eligible(leaf) for leaf in leaves):
        position = names.index("dtype")
        if position < len(args):
            if args[position] is None:
                args = (*args[:position], torch.float32, *args[position + 1 :])
        elif kwargs.get("dtype") is None:
            kwargs = {**kwargs, "dtype": torch.float32}
    return _without_autocast(op, args, kwargs)


def _promote(op, *args, **kwargs):
    leaves, _ = tree_flatten((args, kwargs))
    dtypes = [leaf.dtype for leaf in leaves if _eligible(leaf)]
    if not dtypes:
        return _without_autocast(op, args, kwargs)
    dtype = functools.reduce(torch.promote_types, dtypes)
    return _without_autocast(op, *_cast(dtype, args, kwargs))


def _overloads(name: str):
    packet = getattr(torch.ops.aten, name, None)
    if packet is None:
        return
    for overload in packet.overloads():
        if "out" not in overload:
            yield getattr(packet, overload)


def register() -> list[str]:
    """Register the policy at ``AutocastPrivateUse1``; returns the ops it covers."""
    if _libraries:
        return []
    torch.rbln.get_amp_supported_dtype = lambda: list(AMP_DTYPES)
    fallthrough = torch.library.Library("_", "IMPL")
    fallthrough.fallback(torch.library.fallthrough_kernel, DISPATCH_KEY)
    library = torch.library.Library("aten", "IMPL")
    _libraries.extend((fallthrough, library))
    registered = []
    for names, policy in (
        (LOWER_PRECISION_FP, _lower_precision),
        (FP32, _fp32),
        (FP32_SET_OPT_DTYPE, _fp32_set_opt_dtype),
        (PROMOTE, _promote),
    ):
        for name in names:
            for op in _overloads(name):
                library.impl(op, functools.partial(policy, op), DISPATCH_KEY)
                registered.append(str(op))
    return registered

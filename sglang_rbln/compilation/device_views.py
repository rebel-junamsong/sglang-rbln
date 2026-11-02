# SPDX-License-Identifier: Apache-2.0
"""Feed a compiled graph the base of a strided device view, and slice inside the graph.

rebel takes contiguous device inputs only, so a strided view (Gemma-4's
``per_layer_inputs[:, :, i, :]``, one slice per decoder layer) used to reach the
graph as an eager ``.contiguous()`` copy: one more eager device op per input and
step, and a copy the runtime then fails to resolve in the next graph run
(``RUN_INTERNAL (Invalid key_vaddr)``; see
rebel_requests/eager_contiguous_device_view_invalid_key.md).

When such a view is ``narrow``/``select`` slices of a contiguous base (its ``_base``,
or under ``torch.inference_mode``, where views record none, its storage laid out by
its strides), the graph's
placeholder takes the base instead and rebuilds the view from the same slices
(rebel lowers those; it does not lower ``as_strided``). The rebuild is checked to
reproduce the view's shape, strides and offset exactly; any other view keeps the
contiguous copy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.fx

Slice = tuple  # ("narrow", dim, start, length) | ("select", dim, index)


def apply_slices(base: torch.Tensor, slices: Sequence[Slice]) -> torch.Tensor:
    view = base
    for kind, *args in slices:
        view = getattr(view, kind)(*args)
    return view


def storage_base(view: torch.Tensor) -> torch.Tensor | None:
    """The contiguous tensor that ``view`` slices: ``view._base``, or, for a view
    made under ``torch.inference_mode`` (which records no ``_base``), its storage
    laid out with the view's own strides -- each dim as long as the next-outer
    stride allows, the outermost as long as the storage allows. None when the
    strides do not tile the storage that way."""
    if view._base is not None:
        return view._base
    strides = view.stride()
    if view.dim() == 0 or strides[-1] != 1:
        return None
    numel = view.untyped_storage().nbytes() // view.element_size()
    sizes = []
    for dim, stride in enumerate(strides):
        outer = numel if dim == 0 else strides[dim - 1]
        if stride <= 0 or outer % stride:
            return None
        sizes.append(outer // stride)
    base = torch.as_strided(view, sizes, strides, 0)
    return base if base.is_contiguous() else None


def slices_of(
    view: torch.Tensor, base: torch.Tensor | None = None
) -> tuple[Slice, ...] | None:
    """The narrow/select slices that turn ``base`` (default: ``storage_base(view)``)
    into ``view``, or None when slicing cannot rebuild it exactly."""
    base = storage_base(view) if base is None else base
    if (
        base is None
        or not base.is_contiguous()
        or base.dtype != view.dtype
        or base.device != view.device
        or base.untyped_storage().data_ptr() != view.untyped_storage().data_ptr()
    ):
        return None
    offset = view.storage_offset() - base.storage_offset()
    starts = []
    for stride in base.stride():
        starts.append(offset // stride if stride else 0)
        offset = offset % stride if stride else offset
    if offset:
        return None
    slices: list[Slice] = []
    kept = 0
    want = list(zip(view.shape, view.stride(), strict=True))
    for size, stride, start in zip(base.shape, base.stride(), starts, strict=True):
        if (
            kept < len(want)
            and start + want[kept][0] <= size
            and (want[kept][1] == stride or want[kept][0] == 1 == size)
        ):
            length = want[kept][0]
            if start or length != size:
                slices.append(("narrow", kept, start, length))
            kept += 1
        else:
            slices.append(("select", kept, start))
    if kept != len(want):
        return None
    try:
        rebuilt = apply_slices(base, slices)
    except (IndexError, RuntimeError):
        return None
    if rebuilt.shape != view.shape or rebuilt.storage_offset() != view.storage_offset():
        return None
    if any(
        a != b
        for a, b, size in zip(rebuilt.stride(), view.stride(), view.shape, strict=True)
        if size > 1
    ):
        return None
    return tuple(slices)


def is_offset_view(value: torch.Tensor) -> bool:
    """A device input the graph cannot take as is: strided, or starting inside its
    storage (a one-row slice is "contiguous" but its address is not a buffer's)."""
    return not value.is_contiguous() or value.storage_offset() != 0


@dataclass(frozen=True)
class RebasedViews:
    """Placeholder index -> the slices its argument's base takes to become it."""

    slices: dict[int, tuple[Slice, ...]]
    base_shapes: dict[int, tuple[int, ...]]

    def bases(self, args: Sequence) -> list:
        args = list(args)
        for index, slices in self.slices.items():
            view = args[index]
            base = _base_of_shape(view, self.base_shapes[index])
            if base is None or slices_of(view, base) != slices:
                raise RuntimeError(
                    f"graph input {index} is no longer the same view of its base "
                    f"({tuple(view.shape)}, stride {view.stride()}, offset "
                    f"{view.storage_offset()}); it was compiled as {slices}"
                )
            args[index] = base
        return args


def _base_of_shape(view: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor | None:
    """The contiguous tensor of ``shape`` laid over ``view``'s storage, if it fits."""
    numel = view.untyped_storage().nbytes() // view.element_size()
    if math.prod(shape) != numel:
        return None
    strides = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]
    return torch.as_strided(view, shape, strides, 0)


def rebase_strided_views(
    graph_module: torch.fx.GraphModule,
    example_inputs: Sequence,
    is_device: Callable[[torch.Tensor], bool],
) -> RebasedViews | None:
    """Rewrite placeholders fed by sliced device views (strided, or offset into
    their storage) to take the base instead."""
    placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != len(example_inputs):
        return None
    plan: dict[int, tuple[Slice, ...]] = {}
    base_shapes: dict[int, tuple[int, ...]] = {}
    for index, value in enumerate(example_inputs):
        if (
            isinstance(value, torch.Tensor)
            and is_device(value)
            and is_offset_view(value)
            and (base := storage_base(value)) is not None
            and (slices := slices_of(value, base))
        ):
            plan[index] = slices
            base_shapes[index] = tuple(base.shape)
    # Every other input on a rebased storage is rebased onto the same base: one
    # buffer (one runtime key) must reach the graph in one shape.
    storages = {
        example_inputs[i].untyped_storage().data_ptr(): shape
        for i, shape in base_shapes.items()
    }
    for index, value in enumerate(example_inputs):
        if index in plan or not isinstance(value, torch.Tensor) or not is_device(value):
            continue
        shape = storages.get(value.untyped_storage().data_ptr())
        if shape is None or tuple(value.shape) == shape:
            continue
        base = _base_of_shape(value, shape)
        slices = slices_of(value, base) if base is not None else None
        if slices is None:
            continue
        plan[index] = slices
        base_shapes[index] = shape
    if not plan:
        return None
    graph = graph_module.graph
    body = next(n for n in graph.nodes if n.op != "placeholder")
    for index, slices in plan.items():
        node = placeholders[index]
        with graph.inserting_before(body):
            chain = []
            current = node
            for kind, *args in slices:
                current = graph.call_method(kind, (current, *args))
                chain.append(current)
        first = chain[0]
        node.replace_all_uses_with(
            current, delete_user_cb=lambda user, first=first: user is not first
        )
        node.meta.pop("example_value", None)
        node.meta.pop("val", None)
    graph.lint()
    graph_module.recompile()
    return RebasedViews(plan, base_shapes)

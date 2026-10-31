# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import copy
import itertools
import logging
import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
from torch import nn

from sglang_rbln import envs

logger = logging.getLogger(__name__)

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.utils.torch_compile.compile_matching_submodules"
)


Slot = int | str


def _leading(output: Any) -> tuple:
    return output if isinstance(output, tuple) else (output,)


def _restore(values: tuple, arity: int) -> Any:
    return values if arity > 1 else values[0]


def _slot_values(args: tuple, kwargs: dict, slots: tuple[Slot, ...]) -> tuple:
    return tuple(args[s] if isinstance(s, int) else kwargs.get(s) for s in slots)


def _fill(args: tuple, kwargs: dict, slots: tuple[Slot, ...], values: tuple):
    args, kwargs = list(args), dict(kwargs)
    for slot, value in zip(slots, values, strict=True):
        if isinstance(slot, int):
            args[slot] = value
        else:
            kwargs[slot] = value
    return tuple(args), kwargs


def _find_slots(previous: tuple, args: tuple, kwargs: dict) -> tuple[Slot, ...]:
    candidates: list[tuple[Slot, Any]] = [*enumerate(args), *kwargs.items()]
    slots = []
    for value in previous:
        found = [
            slot
            for slot, arg in candidates
            if isinstance(value, torch.Tensor) and arg is value
        ]
        if len(found) != 1:
            raise RuntimeError(
                "a block's outputs are not passed unchanged to the next block; "
                "set SGLANG_RBLN_BLOCKS_PER_REGION=1 for this model"
            )
        slots.append(found[0])
    return tuple(slots)


class BlockGroup(nn.Module):
    def __init__(self, blocks: Sequence[nn.Module], keep_outputs: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.keep_outputs = keep_outputs
        self.slots: tuple[Slot, ...] = ()

    def forward(self, values: tuple, calls: list[tuple[tuple, dict]]) -> Any:
        kept: tuple = ()
        for block, (args, kwargs) in zip(self.blocks, calls, strict=True):
            args, kwargs = _fill(args, kwargs, self.slots, values)
            values = _leading(block(*args, **kwargs))
            kept += values
        return kept if self.keep_outputs else _restore(values, len(self.slots))


class RegionDoesNotFit(RuntimeError):
    def __init__(self, needed: list[int], free: list[int]) -> None:
        super().__init__(f"needs {needed} bytes per node, {free} free")
        self.needed = needed
        self.free = free


def _node_devices(device: Any, nodes: int) -> list[int]:
    if isinstance(device, list | tuple):
        return list(device)
    return [(device or 0) + node for node in range(nodes)]


def check_fits(compiled_model: Any, device: Any = None) -> None:
    import rebel

    needed = list(compiled_model.get_alloc_per_node())
    infos = [
        rebel.get_device_memory_info(index)
        for index in _node_devices(device, len(needed))
    ]
    free = [info.free - info.granularity for info in infos]
    if any(need > room for need, room in zip(needed, free, strict=True)):
        raise RegionDoesNotFit(needed, free)


@contextlib.contextmanager
def checking_fit() -> Iterator[None]:
    try:
        from rebel import RBLNCompiledModel
    except ImportError:
        yield
        return
    create_runtime = RBLNCompiledModel.create_runtime

    def checked(compiled_model, *args, **kwargs):
        check_fits(compiled_model, kwargs.get("device", args[0] if args else None))
        return create_runtime(compiled_model, *args, **kwargs)

    RBLNCompiledModel.create_runtime = checked
    try:
        yield
    finally:
        RBLNCompiledModel.create_runtime = create_runtime


def _does_not_fit(error: BaseException | None) -> bool:
    while error is not None:
        if isinstance(error, RegionDoesNotFit) or "Lack of device memory" in str(error):
            return True
        error = getattr(error, "inner_exception", None) or error.__cause__
    return False


class Region:
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        compile_kwargs: dict,
        keep_outputs: bool = False,
    ) -> None:
        self.blocks = list(blocks)
        self.compile_kwargs = compile_kwargs
        self.keep_outputs = keep_outputs
        self.group: BlockGroup | None = BlockGroup(self.blocks, keep_outputs)
        self.group.compile(**compile_kwargs)
        self.parts: list[Region] = []
        self.loaded = False

    def _split(self, error: BaseException) -> None:
        half = (len(self.blocks) + 1) // 2
        logger.warning(
            "A %d-block region does not fit (%s); splitting it into %d + %d blocks",
            len(self.blocks),
            error,
            half,
            len(self.blocks) - half,
        )
        slots = self.group.slots
        self.parts = [
            Region(self.blocks[:half], self.compile_kwargs, self.keep_outputs),
            Region(self.blocks[half:], self.compile_kwargs, self.keep_outputs),
        ]
        for part in self.parts:
            part.group.slots = slots
        self.group = None

    def __call__(self, values: tuple, calls: list[tuple[tuple, dict]]) -> Any:
        if self.parts:
            output, kept = None, ()
            for part in self.parts:
                output = part(values, calls[: len(part.blocks)])
                calls = calls[len(part.blocks) :]
                if self.keep_outputs:
                    kept += output
                    values = output[-len(values) :]
                else:
                    values = _leading(output)
            return kept if self.keep_outputs else output
        if self.loaded:
            return self.group(values, calls)
        try:
            with checking_fit():
                output = self.group(values, calls)
        except Exception as error:
            if len(self.blocks) == 1 or not _does_not_fit(error):
                raise
            self._split(error)
            return self(values, calls)
        self.loaded = True
        return output


class DeferredBlock:
    def __init__(self, group: BlockGroup, index: int, state: dict) -> None:
        self.group = group
        self.index = index
        self.state = state

    def _learn(self, args: tuple, kwargs: dict) -> Any:
        state = self.state
        if self.index:
            slots = _find_slots(state["previous"], args, kwargs)
            if self.group.slots and slots != self.group.slots:
                raise RuntimeError(
                    "blocks of one region take the previous outputs in different "
                    "arguments; set SGLANG_RBLN_BLOCKS_PER_REGION=1 for this model"
                )
            self.group.slots = slots
        output = state["eager"][self.index](*args, **kwargs)
        state["previous"] = _leading(output)
        state["calls"].append(None)
        if self.index == len(self.group.blocks) - 1:
            state["learned"] = True
            state["calls"].clear()
            del state["previous"]
        return output

    def __call__(self, *args, **kwargs):
        state = self.state
        if len(state["calls"]) != self.index:
            state["calls"].clear()
            raise RuntimeError(
                f"block {self.index} of a {len(self.group.blocks)}-block region ran "
                "out of order; set SGLANG_RBLN_BLOCKS_PER_REGION=1 for this model"
            )
        if not state.get("learned"):
            return self._learn(args, kwargs)
        slots = self.group.slots
        values = _slot_values(args, kwargs, slots)
        if self.index == 0:
            state["values"] = values
        empty = (None,) * len(slots)
        state["calls"].append(_fill(args, kwargs, slots, empty))
        if self.index < len(self.group.blocks) - 1:
            if self.group.keep_outputs:
                values = tuple(map(torch.empty_like, values))
                state.setdefault("pending", []).append(values)
            return _restore(values, len(slots))
        calls = list(state["calls"])
        state["calls"].clear()
        output = state["compiled"](state.pop("values"), calls)
        if not self.group.keep_outputs:
            return output
        *kept, last = itertools.batched(output, len(slots))
        pending = state.pop("pending", [])
        for placeholder, value in zip(
            itertools.chain(*pending), itertools.chain(*kept), strict=True
        ):
            placeholder.copy_(value)
        return _restore(last, len(slots))


def consecutive_runs(
    matches: Sequence[tuple[str, nn.Module]],
) -> list[list[nn.Module]]:
    runs: list[list[nn.Module]] = []
    previous: tuple[str, int] | None = None
    for name, submodule in matches:
        parent, _, index = name.rpartition(".")
        current = (parent, int(index)) if index.isdigit() else None
        follows = (
            current is not None
            and previous is not None
            and current[0] == previous[0]
            and current[1] == previous[1] + 1
        )
        if not follows:
            runs.append([])
        runs[-1].append(submodule)
        previous = current
    return runs


def graph_count(blocks: Sequence[nn.Module], size: int | None) -> int:
    return math.ceil(len(blocks) / (size or len(blocks)))


def group_blocks(
    blocks: Sequence[nn.Module],
    size: int | None,
    compile_kwargs: dict,
    keep_outputs: bool = False,
) -> list[Region]:
    size = size or len(blocks)
    groups = []
    for start in range(0, len(blocks), size):
        members = list(blocks[start : start + size])
        if len(members) == 1:
            members[0].compile(**compile_kwargs)
            continue
        region = Region(
            [copy.copy(member) for member in members], compile_kwargs, keep_outputs
        )
        state: dict[str, Any] = {
            "calls": [],
            "compiled": region,
            "eager": [member.forward for member in members],
        }
        for index, member in enumerate(members):
            member.forward = DeferredBlock(region.group, index, state)
        groups.append(region)
    return groups


def compile_block_groups(
    compile_matching_submodules: Callable, module: nn.Module, *, compile_kwargs: dict
) -> int:
    size = envs.SGLANG_RBLN_BLOCKS_PER_REGION
    if size == 1:
        return compile_matching_submodules(module, compile_kwargs=compile_kwargs)
    conditions = getattr(module, "_compile_conditions", ())
    matches = [
        (name, submodule)
        for name, submodule in module.named_modules()
        if name and any(condition(name, submodule) for condition in conditions)
    ]
    if not matches:
        return compile_matching_submodules(module, compile_kwargs=compile_kwargs)
    runs = consecutive_runs(matches)
    for run in runs:
        group_blocks(run, size, compile_kwargs)
    return sum(graph_count(run, size) for run in runs)


def register() -> str:
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    HookRegistry.register(HOOK_TARGET, compile_block_groups, HookType.AROUND)
    return HOOK_TARGET

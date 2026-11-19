# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextvars
import copy
import itertools
import logging
import math
import pickle
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn
from torch.utils._pytree import tree_leaves, tree_map

from sglang_rbln import envs
from sglang_rbln.compilation import placement, shapes
from sglang_rbln.compilation.backend import forget_frames, release_graphs

logger = logging.getLogger(__name__)

# Set on a module once its blocks are grouped: the number of graphs they compile to.
GRAPH_COUNT = "_rbln_region_graphs"
HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.utils.torch_compile.compile_matching_submodules"
)


# Where a block takes the previous block's output: a positional index or a kwarg name.
Slot = int | str


def _as_tuple(output: Any) -> tuple:
    return output if isinstance(output, tuple) else (output,)


def _from_tuple(values: tuple, arity: int) -> Any:
    return values if arity > 1 else values[0]


def _slot_values(args: tuple, kwargs: dict, slots: tuple[Slot, ...]) -> tuple:
    return tuple(args[s] if isinstance(s, int) else kwargs.get(s) for s in slots)


def _fill_slots(args: tuple, kwargs: dict, slots: tuple[Slot, ...], values: tuple):
    args, kwargs = list(args), dict(kwargs)
    for slot, value in zip(slots, values, strict=True):
        if isinstance(slot, int):
            args[slot] = value
        else:
            kwargs[slot] = value
    return tuple(args), kwargs


def _find_slots(returned: tuple, args: tuple, kwargs: dict) -> tuple[Slot, ...]:
    """The slot each of the previous block's outputs (`returned`) is passed in."""
    candidates: list[tuple[Slot, Any]] = [*enumerate(args), *kwargs.items()]
    slots = []
    for value in returned:
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


def _copied_leaf(leaf: Any) -> Any:
    if isinstance(leaf, torch.Tensor):
        return leaf.clone()
    if isinstance(leaf, nn.Module):
        return leaf
    try:
        return copy.deepcopy(leaf)
    except (TypeError, RuntimeError, pickle.PicklingError):
        return leaf


def _copied_state(value: Any) -> Any:
    return tree_map(
        lambda leaf: leaf.clone() if isinstance(leaf, torch.Tensor) else leaf, value
    )


def call_sandboxed(function: Callable, args: tuple, kwargs: dict) -> Any:
    """Run `function` on copies, so its side effects reach no caller state."""

    def run() -> Any:
        for var, value in contextvars.copy_context().items():
            var.set(_copied_state(value))
        copied_args, copied_kwargs = tree_map(_copied_leaf, (args, kwargs))
        return function(*copied_args, **copied_kwargs)

    return contextvars.copy_context().run(run)


def _as_placeholder(leaf: Any) -> Any:
    if isinstance(leaf, torch.Tensor):
        return torch.empty(leaf.shape, dtype=leaf.dtype, device="meta")
    return leaf


def run_shape_only(function: Callable, args: tuple, kwargs: dict) -> Any:
    """`function`'s output, as meta tensors, from a sandboxed run on fake tensors:
    no op runs on the host or the NPUs. A function that reads its data, which fake
    tensors cannot, runs sandboxed on copies instead."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    try:
        with FakeTensorMode(allow_non_fake_inputs=True):
            output = call_sandboxed(function, args, kwargs)
    except Exception:  # noqa: BLE001 - any op a fake tensor cannot run
        logger.debug("block 0 reads its data; learning its slots eagerly")
        output = call_sandboxed(function, args, kwargs)
    return tree_map(_as_placeholder, output)


class BlockGroup(nn.Module):
    """A region's blocks as one module: each block takes the previous one's outputs in
    `slots`, and with `keep_outputs` the group returns every block's outputs."""

    def __init__(self, blocks: Sequence[nn.Module], keep_outputs: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.keep_outputs = keep_outputs
        self.slots: tuple[Slot, ...] = ()

    def forward(self, values: tuple, calls: list[tuple[tuple, dict]]) -> Any:
        kept: tuple = ()
        for block, (args, kwargs) in zip(self.blocks, calls, strict=True):
            args, kwargs = _fill_slots(args, kwargs, self.slots, values)
            values = _as_tuple(block(*args, **kwargs))
            kept += values
        return kept if self.keep_outputs else _from_tuple(values, len(self.slots))


class RegionDoesNotFit(RuntimeError):
    def __init__(self, needed: list[int], free: list[int]) -> None:
        super().__init__(f"needs {needed} bytes per node, {free} free")
        self.needed = needed
        self.free = free


def _runtime_npus(device: Any, count: int) -> list[int]:
    """The NPUs of a runtime created for `device`: a list, or the first of `count`."""
    if isinstance(device, list | tuple):
        return list(device)
    return [(device or 0) + npu for npu in range(count)]


POOLED_KEYS = ("Intermediate", "Input", "Output")
# (context, NPUs) -> key -> bytes per NPU recorded so far
_pools: dict[tuple, dict[str, list[int]]] = {}
runtimes_created = 0


def _pooled(compiled_model: Any, needed: list[int]) -> dict[str, list[int]]:
    """The per-NPU sizes of POOLED_KEYS when the per-key sizes add up to more than
    the runtime's reported total (`needed`); empty when the total covers them."""
    by_key = getattr(compiled_model, "get_alloc_per_node_by_key", None)
    if by_key is None:
        return {}
    sizes = {key: list(values) for key, values in by_key().items()}
    npu_count = len(needed)
    totals = [
        sum(values[npu] if npu < len(values) else 0 for values in sizes.values())
        for npu in range(npu_count)
    ]
    if all(total <= need for total, need in zip(totals, needed, strict=True)):
        return {}
    return {
        key: [*sizes[key], *[0] * (npu_count - len(sizes[key]))]
        for key in POOLED_KEYS
        if key in sizes
    }


def _pool_key(compiled_model: Any, npus: list[int]) -> tuple:
    return compiled_model.select_executor().get_context_id(), tuple(npus)


def check_fits(compiled_model: Any, device: Any = None) -> None:
    """Raise RegionDoesNotFit unless the runtime fits the room rebel reports on its
    NPUs."""
    import rebel

    needed = list(compiled_model.get_alloc_per_node())
    npus = _runtime_npus(device, len(needed))
    if pooled := _pooled(compiled_model, needed):
        held = _pools.get(_pool_key(compiled_model, npus), {})
        for key, sizes in pooled.items():
            pool = held.get(key, [0] * len(npus))
            for npu, size in enumerate(sizes):
                needed[npu] += max(0, size - pool[npu])
    infos = list(map(rebel.get_device_memory_info, npus))
    if None in infos:  # no memory info (compile-only): nothing to check against
        return
    free = [
        info.free - info.granularity + rebel.get_evictable_memory(index)
        for index, info in zip(npus, infos, strict=True)
    ]
    if any(need > room for need, room in zip(needed, free, strict=True)):
        raise RegionDoesNotFit(needed, free)


def _record_pools(compiled_model: Any, device: Any) -> None:
    needed = list(compiled_model.get_alloc_per_node())
    if not (pooled := _pooled(compiled_model, needed)):
        return
    npus = _runtime_npus(device, len(needed))
    held = _pools.setdefault(_pool_key(compiled_model, npus), {})
    for key, sizes in pooled.items():
        pool = held.setdefault(key, [0] * len(npus))
        held[key] = [max(have, size) for have, size in zip(pool, sizes, strict=True)]


def _checked_create_runtime(create_runtime: Callable) -> Callable:
    def checked(compiled_model, *args, **kwargs):
        global runtimes_created
        device = kwargs.get("device", args[0] if args else None)
        check_fits(compiled_model, device)
        runtime = create_runtime(compiled_model, *args, **kwargs)
        _record_pools(compiled_model, device)
        runtimes_created += 1
        return runtime

    return checked


_installed = False


def install_fit_check() -> bool:
    """Check every runtime this process creates, whatever compiled it: a block
    region, or a module compiled straight through `Module.compile` (the VAE
    coders, the vocoder, the duration head). One that does not fit raises
    RegionDoesNotFit with the bytes it needs and the bytes free, not an
    out-of-memory error with no sizes.
    """
    from rebel import RBLNCompiledModel

    global _installed
    if _installed:
        return False
    RBLNCompiledModel.create_runtime = _checked_create_runtime(
        RBLNCompiledModel.create_runtime
    )
    _installed = True
    return True


# Runtime failures to place a loaded graph's buffers. No typed error carries them.
OUT_OF_DEVICE_MEMORY = (
    "Lack of device memory",
    "SYS_ENOMEM",
    "RBLNCachingAllocator::malloc block != nullptr",
)


def _does_not_fit(error: BaseException | None) -> bool:
    # Codegen refuses a graph that needs more DRAM than a node has with this type.
    from rebel.core.exception import RBLNDeviceMemoryError

    while error is not None:
        if isinstance(error, (RegionDoesNotFit, RBLNDeviceMemoryError)):
            return True
        if any(message in str(error) for message in OUT_OF_DEVICE_MEMORY):
            return True
        error = getattr(error, "inner_exception", None) or error.__cause__
    return False


def run_on_cpu_if_it_does_not_fit(call: Callable, module: nn.Module, what: str) -> Any:
    """Run `call`; if `module` does not fit its NPUs, run it eager instead.

    A module compiled by default has to degrade, not fail: the request still
    has to be served. `nn.Module.compile` only installs a compiled call, so
    dropping it leaves the eager module the caller already holds.
    """
    try:
        return call()
    except Exception as error:
        if not _does_not_fit(error):
            raise
        module._compiled_call_impl = None
        logger.warning(
            "The LTX-2 %s does not fit its NPUs (%s); it stays on the CPU", what, error
        )
        return call()


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
        self.group = BlockGroup(self.blocks, keep_outputs)
        # The group is a new module with no placement: it takes the NPUs of its first
        # block's submodules (the blocks are unplaced shallow copies).
        placed = map(placement.devices_of, self.blocks[0].modules())
        if devices := next(filter(None, placed), None):
            placement.place(self.group, devices)
        self.group.compile(**compile_kwargs)
        # call key -> the two halves this region was split into at that key
        self.splits: dict[tuple, list[Region]] = {}
        # call keys whose runtimes are already loaded and fit
        self.fitted: set[tuple] = set()

    @staticmethod
    def _key(values: tuple, calls: list[tuple[tuple, dict]]) -> tuple:
        """The request's shape key and the tensor shapes this call compiles for:
        the stages of one request (two-stage refinement) call one region at
        different shapes."""
        return shapes.current_shape_key.get(), tuple(
            (tuple(leaf.shape), leaf.dtype)
            for leaf in tree_leaves((values, calls))
            if isinstance(leaf, torch.Tensor)
        )

    def _split(self, key: tuple, error: BaseException) -> None:
        half = (len(self.blocks) + 1) // 2
        logger.warning(
            "A %d-block region does not fit at shape %s (%s); "
            "splitting it into %d + %d blocks",
            len(self.blocks),
            key[0],
            error,
            half,
            len(self.blocks) - half,
        )
        self.splits[key] = [
            Region(self.blocks[:half], self.compile_kwargs, self.keep_outputs),
            Region(self.blocks[half:], self.compile_kwargs, self.keep_outputs),
        ]
        for part in self.splits[key]:
            part.group.slots = self.group.slots

    def _release(self) -> None:
        """Drop what an attempt that did not fit left loaded; otherwise its runtimes
        stay on the NPUs and every later request fails.

        A fit-check refusal loads nothing; a runtime that ran out of memory holds its
        device memory until its graph is dropped. The groups share their graphs'
        cache, so every group lets go of its graphs, and this region's other shapes
        go through the fit check again when next called.
        """
        release_graphs(self.group)
        self.fitted.clear()

    def __call__(self, values: tuple, calls: list[tuple[tuple, dict]]) -> Any:
        key = self._key(values, calls)
        if parts := self.splits.get(key):
            output, kept = None, ()
            for part in parts:
                output = part(values, calls[: len(part.blocks)])
                calls = calls[len(part.blocks) :]
                if self.keep_outputs:
                    kept += output
                    values = output[-len(values) :]
                else:
                    values = _as_tuple(output)
            return kept if self.keep_outputs else output
        if key in self.fitted:
            return self.group(values, calls)
        runtimes_before = runtimes_created
        try:
            output = self.group(values, calls)
        except Exception as error:
            if not _does_not_fit(error):
                raise
            if runtimes_created > runtimes_before:
                forget_frames(error)
                self._release()
            if len(self.blocks) == 1:
                raise
            self._split(key, error)
            return self(values, calls)
        self.fitted.add(key)
        return output


class DeferredBlock:
    """One block's forward inside a region: records the call, and the last block runs
    the whole region.

    `state` is shared by the region's blocks: "eager" (the blocks' own forwards),
    "compiled" (the Region), "calls" (this pass's recorded calls), "values" (the
    region's inputs), "returned" (what the previous block handed back), "pending"
    (the placeholders for kept outputs), "first" (block 0's call, held until block
    1 shows the slots) and "learned" (the slots are known).
    """

    def __init__(self, group: BlockGroup, index: int, state: dict) -> None:
        self.group = group
        self.index = index
        self.state = state

    def _learn(self, args: tuple, kwargs: dict) -> Any:
        """Find the slots on the first pass: block 0 runs on fake tensors for its
        output's shapes, which the caller holds as placeholders until the region
        runs, and block 1's call shows where that output went."""
        state = self.state
        if self.index == 0:
            state["first"] = (args, kwargs)
            state["calls"].append(None)
            output = run_shape_only(state["eager"][0], args, kwargs)
            state["returned"] = _as_tuple(output)
            return output
        slots = _find_slots(state["returned"], args, kwargs)
        self.group.slots = slots
        state["learned"] = True
        first_args, first_kwargs = state.pop("first")
        state["values"] = _slot_values(first_args, first_kwargs, slots)
        state["calls"] = [
            _fill_slots(first_args, first_kwargs, slots, (None,) * len(slots))
        ]
        if self.group.keep_outputs:
            state["pending"] = [state["returned"]]
        return None

    def __call__(self, *args, **kwargs):
        state = self.state
        if len(state["calls"]) != self.index:
            state["calls"].clear()
            raise RuntimeError(
                f"block {self.index} of a {len(self.group.blocks)}-block region ran "
                "out of order; set SGLANG_RBLN_BLOCKS_PER_REGION=1 for this model"
            )
        if not state.get("learned"):
            if self.index == 0:
                return self._learn(args, kwargs)
            self._learn(args, kwargs)
        slots = self.group.slots
        values = _slot_values(args, kwargs, slots)
        if self.index == 0:
            state["values"] = values
        elif any(
            value is not returned
            for value, returned in zip(values, state["returned"], strict=True)
        ):
            state["calls"].clear()
            raise RuntimeError(
                "a block's outputs are not passed unchanged to the next block; "
                "set SGLANG_RBLN_BLOCKS_PER_REGION=1 for this model"
            )
        empty = (None,) * len(slots)
        state["calls"].append(_fill_slots(args, kwargs, slots, empty))
        if self.index < len(self.group.blocks) - 1:
            if self.group.keep_outputs:
                values = self._placeholders(values)
                state.setdefault("pending", []).append(values)
            state["returned"] = values
            return _from_tuple(values, len(slots))
        calls = list(state["calls"])
        state["calls"].clear()
        output = state["compiled"](state.pop("values"), calls)
        if not self.group.keep_outputs:
            return output
        width = len(slots)
        pending = state.pop("pending", [])
        kept = zip(itertools.chain(*pending), output[:-width], strict=True)
        for placeholder, value in kept:
            _become(placeholder, value)
        return _from_tuple(output[-width:], width)

    @staticmethod
    def _placeholders(values: tuple) -> tuple:
        """What the caller holds for each kept output until the region runs; it then
        takes the output itself, which stays on the NPUs."""
        return tuple(
            torch.empty(value.shape, dtype=value.dtype, device="meta")
            for value in values
        )


def _become(placeholder: torch.Tensor, value: torch.Tensor) -> None:
    """Make the meta placeholder the caller holds for a kept output hold the region's
    output.

    `swap_tensors` refuses a tensor anything references weakly, and the output can
    be: a split region's output is the next region's input, which dynamo's guards
    reference. So the placeholder takes a fresh alias of the output's storage, which
    nothing references yet.
    """
    alias = value.detach()
    alias.__dict__.update(value.__dict__)
    torch.utils.swap_tensors(placeholder, alias)


def consecutive_runs(
    matches: Sequence[tuple[str, nn.Module]],
) -> list[list[nn.Module]]:
    """Group named modules into runs of consecutive indices under one parent
    (`blocks.3`, `blocks.4`, ...)."""
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
    regions = []
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
        regions.append(region)
    return regions


def compile_block_groups(
    compile_matching_submodules: Callable, module: nn.Module, *, compile_kwargs: dict
) -> int:
    # Every denoising stage of a two-stage pipeline compiles the transformer it shares.
    if (count := getattr(module, GRAPH_COUNT, None)) is not None:
        return count
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
    count = sum(graph_count(run, size) for run in runs)
    setattr(module, GRAPH_COUNT, count)
    return count


def register() -> str:
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    HookRegistry.register(HOOK_TARGET, compile_block_groups, HookType.AROUND)
    return HOOK_TARGET

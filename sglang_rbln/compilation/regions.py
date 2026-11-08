# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import contextvars
import copy
import itertools
import logging
import math
import pickle
import weakref
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
from torch import nn
from torch.utils._pytree import tree_leaves, tree_map

from sglang_rbln import device, envs
from sglang_rbln.compilation import placement, shapes
from sglang_rbln.compilation.backend import release_graphs

logger = logging.getLogger(__name__)

GRAPH_COUNT = "_rbln_region_graphs"
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


_state_vars: weakref.WeakSet = weakref.WeakSet()


def register_state(var: Any) -> None:
    _state_vars.add(var)


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
        originals = [(var, var.value) for var in _state_vars]
        for var, value in originals:
            var.value = _copied_state(value)
        try:
            copied_args, copied_kwargs = tree_map(_copied_leaf, (args, kwargs))
            return function(*copied_args, **copied_kwargs)
        finally:
            for var, value in originals:
                var.value = value

    return contextvars.copy_context().run(run)


# One nested compile region per block class, made outside tracing: Dynamo
# inlines a region it creates while tracing (functools caches included).
_layer_regions: dict[Callable, Callable] = {}


class BlockGroup(nn.Module):
    def __init__(self, blocks: Sequence[nn.Module], keep_outputs: bool = False) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.keep_outputs = keep_outputs
        self.slots: tuple[Slot, ...] = ()
        if envs.SGLANG_RBLN_ROLL_BLOCKS:
            for forward in {type(block).forward for block in blocks}:
                if forward not in _layer_regions:
                    _layer_regions[forward] = torch.compiler.nested_compile_region(
                        forward
                    )

    def _call_block(self, block: nn.Module, args: tuple, kwargs: dict) -> Any:
        # As calls of one region, the blocks reach the compiler as a layer loop
        # it can compile once (RBLN_EXP_ROLL_LAYERS_ANY_DEVICE on CA22/CA25).
        if region := _layer_regions.get(type(block).forward):
            return region(block, *args, **kwargs)
        return block(*args, **kwargs)

    def forward(self, values: tuple, calls: list[tuple[tuple, dict]]) -> Any:
        kept: tuple = ()
        for block, (args, kwargs) in zip(self.blocks, calls, strict=True):
            args, kwargs = _fill(args, kwargs, self.slots, values)
            values = _leading(self._call_block(block, args, kwargs))
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


# Buffers the runtimes of one sharing session on the same NPUs share: one block per kind
# and node, as large as the largest runtime that has needed it.
POOLED_KEYS = ("Intermediate", "Input", "Output")
# (session, NPUs) -> pooled kind -> bytes per node the session's pool holds now
_pools: dict[tuple, dict[str, list[int]]] = {}
runtimes_created = 0


def _pooled(compiled_model: Any, needed: list[int]) -> dict[str, list[int]]:
    """The pooled buffers the runtime needs per node, when its own total leaves them
    out; empty when they are counted in it."""
    by_key = getattr(compiled_model, "get_alloc_per_node_by_key", None)
    if by_key is None:
        return {}
    sizes = {key: list(values) for key, values in by_key().items()}
    pad = len(needed)
    totals = [
        sum(values[node] if node < len(values) else 0 for values in sizes.values())
        for node in range(pad)
    ]
    if all(total <= need for total, need in zip(totals, needed, strict=True)):
        return {}
    return {
        key: [*sizes[key], *[0] * (pad - len(sizes[key]))]
        for key in POOLED_KEYS
        if key in sizes
    }


def _pool_key(compiled_model: Any, nodes: list[int]) -> tuple:
    return compiled_model.select_executor().get_context_id(), tuple(nodes)


def check_fits(compiled_model: Any, device: Any = None) -> None:
    """Raise RegionDoesNotFit unless the runtime fits the room on its NPUs: their free
    memory, and what evicting idle graphs' programs would free.

    In a sharing session the runtime's own total leaves out the pooled buffers. A
    pooled buffer the runtime needs larger than the session's pool holds grows: the
    runtime frees the old block and allocates the larger one, so the growth is what
    it takes beyond its own total.
    """
    import rebel

    needed = list(compiled_model.get_alloc_per_node())
    nodes = _node_devices(device, len(needed))
    if pooled := _pooled(compiled_model, needed):
        held = _pools.get(_pool_key(compiled_model, nodes), {})
        for key, sizes in pooled.items():
            pool = held.get(key, [0] * len(nodes))
            for node, size in enumerate(sizes):
                needed[node] += max(0, size - pool[node])
    # Loading a graph evicts the programs of graphs not running now; they reload on
    # their next run. Wheels before the query count none.
    evictable = getattr(rebel, "get_evictable_memory", lambda index: 0)
    free = [
        info.free - info.granularity + evictable(index)
        for index, info in zip(
            nodes, map(rebel.get_device_memory_info, nodes), strict=True
        )
    ]
    if any(need > room for need, room in zip(needed, free, strict=True)):
        raise RegionDoesNotFit(needed, free)


def _record_pools(compiled_model: Any, device: Any) -> None:
    needed = list(compiled_model.get_alloc_per_node())
    if not (pooled := _pooled(compiled_model, needed)):
        return
    nodes = _node_devices(device, len(needed))
    held = _pools.setdefault(_pool_key(compiled_model, nodes), {})
    for key, sizes in pooled.items():
        pool = held.setdefault(key, [0] * len(nodes))
        held[key] = [max(have, size) for have, size in zip(pool, sizes, strict=True)]


def _checked_create_runtime(create_runtime: Callable) -> Callable:
    def checked(compiled_model, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        check_fits(compiled_model, device)
        global runtimes_created
        runtime = create_runtime(compiled_model, *args, **kwargs)
        _record_pools(compiled_model, device)
        runtimes_created += 1
        return runtime

    return checked


_installed = False


def install_fit_check() -> bool:
    """Check every runtime this process creates, whatever compiled it.

    `checking_fit` only covers block regions, so a module compiled straight
    through `Module.compile` -- the VAE coders, the vocoder, the duration head
    -- reached the device unchecked and failed with a bare SYS_ENOMEM from the
    runtime instead of a RegionDoesNotFit naming what it needed.
    """
    global _installed
    if _installed:
        return False
    try:
        from rebel import RBLNCompiledModel
    except ImportError:
        return False
    RBLNCompiledModel.create_runtime = _checked_create_runtime(
        RBLNCompiledModel.create_runtime
    )
    _installed = True
    return True


@contextlib.contextmanager
def checking_fit() -> Iterator[None]:
    if _installed:
        yield
        return
    try:
        from rebel import RBLNCompiledModel
    except ImportError:
        yield
        return
    create_runtime = RBLNCompiledModel.create_runtime
    RBLNCompiledModel.create_runtime = _checked_create_runtime(create_runtime)
    try:
        yield
    finally:
        RBLNCompiledModel.create_runtime = create_runtime


OUT_OF_DEVICE_MEMORY = ("Lack of device memory", "SYS_ENOMEM")


def _does_not_fit(error: BaseException | None) -> bool:
    while error is not None:
        if isinstance(error, RegionDoesNotFit) or any(
            message in str(error) for message in OUT_OF_DEVICE_MEMORY
        ):
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
        self.group: BlockGroup | None = BlockGroup(self.blocks, keep_outputs)
        # A new module, so it takes the NPUs its blocks are placed on. The blocks
        # are shallow copies, placed through their submodules.
        placed = map(placement.devices_of, self.blocks[0].modules())
        if devices := next(filter(None, placed), None):
            placement.place(self.group, devices)
        self.group.compile(**compile_kwargs)
        self.parts: dict[tuple, list[Region]] = {}
        self.loaded: set[tuple] = set()

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
        self.parts[key] = [
            Region(self.blocks[:half], self.compile_kwargs, self.keep_outputs),
            Region(self.blocks[half:], self.compile_kwargs, self.keep_outputs),
        ]
        for part in self.parts[key]:
            part.group.slots = self.group.slots

    def _release(self) -> None:
        """Drop what an attempt that did not fit left loaded: an abandoned shape kept
        its runtimes on the DiT's NPUs, and every later request failed.

        A fit-check refusal loads nothing; a runtime that ran out of memory after it
        was created holds its programs until its graph goes. The groups share their
        graphs' cache, so every group lets go of its graphs, and this region's other
        shapes go through the fit check again when next called.
        """
        release_graphs(self.group)
        self.loaded.clear()

    def __call__(self, values: tuple, calls: list[tuple[tuple, dict]]) -> Any:
        key = self._key(values, calls)
        if parts := self.parts.get(key):
            output, kept = None, ()
            for part in parts:
                output = part(values, calls[: len(part.blocks)])
                calls = calls[len(part.blocks) :]
                if self.keep_outputs:
                    kept += output
                    values = output[-len(values) :]
                else:
                    values = _leading(output)
            return kept if self.keep_outputs else output
        if key in self.loaded:
            return self.group(values, calls)
        created = runtimes_created
        try:
            with checking_fit():
                output = self.group(values, calls)
        except Exception as error:
            if not _does_not_fit(error):
                raise
            if runtimes_created > created:
                self._release()
            if len(self.blocks) == 1:
                raise
            self._split(key, error)
            return self(values, calls)
        self.loaded.add(key)
        return output


class DeferredBlock:
    def __init__(self, group: BlockGroup, index: int, state: dict) -> None:
        self.group = group
        self.index = index
        self.state = state

    def _learn(self, args: tuple, kwargs: dict) -> Any:
        state = self.state
        if self.index == 0:
            state["first"] = (args, kwargs)
            state["calls"].append(None)
            output = call_sandboxed(state["eager"][0], args, kwargs)
            state["returned"] = _leading(output)
            return output
        slots = _find_slots(state["returned"], args, kwargs)
        self.group.slots = slots
        state["learned"] = True
        first_args, first_kwargs = state.pop("first")
        state["values"] = _slot_values(first_args, first_kwargs, slots)
        state["calls"] = [_fill(first_args, first_kwargs, slots, (None,) * len(slots))]
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
        state["calls"].append(_fill(args, kwargs, slots, empty))
        if self.index < len(self.group.blocks) - 1:
            if self.group.keep_outputs:
                values = self._placeholders(values)
                state.setdefault("pending", []).append(values)
            state["returned"] = values
            return _restore(values, len(slots))
        calls = list(state["calls"])
        state["calls"].clear()
        output = state["compiled"](state.pop("values"), calls)
        if not self.group.keep_outputs:
            return output
        width = len(slots)
        pending = state.pop("pending", [])
        kept = zip(itertools.chain(*pending), output[:-width], strict=True)
        for placeholder, value in kept:
            if device.resident_regions():
                torch.utils.swap_tensors(placeholder, value)
            else:
                placeholder.copy_(value)
        return _restore(output[-width:], width)

    @staticmethod
    def _placeholders(values: tuple) -> tuple:
        """What the caller holds for each kept output until the region runs. With
        resident regions it takes the output itself, which stays on the NPUs."""
        if not device.resident_regions():
            return tuple(map(torch.empty_like, values))
        return tuple(
            torch.empty(value.shape, dtype=value.dtype, device="meta")
            for value in values
        )


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

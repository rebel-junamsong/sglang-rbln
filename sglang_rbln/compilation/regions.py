# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from torch import nn

from sglang_rbln import envs

HOOK_TARGET = (
    "sglang.multimodal_gen.runtime.utils.torch_compile.compile_matching_submodules"
)


def _leading(output: Any) -> tuple:
    return output if isinstance(output, tuple) else (output,)


def _restore(leading: tuple, arity: int) -> Any:
    return leading if arity > 1 else leading[0]


class BlockGroup(nn.Module):
    """Runs consecutive blocks as one region, each fed the previous one's outputs."""

    def __init__(self, blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, leading: tuple, calls: list[tuple[tuple, dict]]) -> Any:
        output: Any = leading
        for block, (rest, kwargs) in zip(self.blocks, calls, strict=True):
            output = type(block).forward(block, *_leading(output), *rest, **kwargs)
        return output


class DeferredBlock:
    """Forward of one block in a group: record the call, run the group at its end."""

    def __init__(self, group: BlockGroup, index: int, state: dict) -> None:
        self.group = group
        self.index = index
        self.state = state

    def __call__(self, *args, **kwargs):
        state = self.state
        if len(state["calls"]) != self.index:
            state["calls"].clear()
            raise RuntimeError(
                f"block {self.index} of a {len(self.group.blocks)}-block region ran "
                "out of order; set SGLANG_RBLN_BLOCKS_PER_REGION=1 for this model"
            )
        arity = state.get("arity")
        if arity is None:
            output = state["eager"][self.index](*args, **kwargs)
            state["calls"].append(None)
            if self.index == len(self.group.blocks) - 1:
                state["arity"] = len(_leading(output))
                state["calls"].clear()
            return output
        if self.index == 0:
            state["leading"] = args[:arity]
        state["calls"].append((args[arity:], kwargs))
        if self.index < len(self.group.blocks) - 1:
            return _restore(args[:arity], arity)
        calls = list(state["calls"])
        state["calls"].clear()
        return state["compiled"](state.pop("leading"), calls)


def _consecutive_runs(
    module: nn.Module, matches: Sequence[tuple[str, nn.Module]]
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


def group_blocks(
    blocks: Sequence[nn.Module], size: int, compile_kwargs: dict
) -> list[BlockGroup]:
    groups = []
    for start in range(0, len(blocks), size):
        members = list(blocks[start : start + size])
        if len(members) == 1:
            members[0].compile(**compile_kwargs)
            continue
        group = BlockGroup(members)
        group.compile(**compile_kwargs)
        state: dict[str, Any] = {
            "calls": [],
            "compiled": group,
            "eager": [member.forward for member in members],
        }
        for index, member in enumerate(members):
            member.forward = DeferredBlock(group, index, state)
        groups.append(group)
    return groups


def compile_block_groups(
    compile_matching_submodules: Callable, module: nn.Module, *, compile_kwargs: dict
) -> int:
    size = envs.SGLANG_RBLN_BLOCKS_PER_REGION
    if size <= 1:
        return compile_matching_submodules(module, compile_kwargs=compile_kwargs)
    conditions = getattr(module, "_compile_conditions", ())
    matches = [
        (name, submodule)
        for name, submodule in module.named_modules()
        if name and any(condition(name, submodule) for condition in conditions)
    ]
    if not matches:
        return compile_matching_submodules(module, compile_kwargs=compile_kwargs)
    regions = 0
    for run in _consecutive_runs(module, matches):
        groups = group_blocks(run, size, compile_kwargs)
        regions += len(groups) + len(run) - sum(len(g.blocks) for g in groups)
    return regions


def register() -> str:
    from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

    HookRegistry.register(HOOK_TARGET, compile_block_groups, HookType.AROUND)
    return HOOK_TARGET

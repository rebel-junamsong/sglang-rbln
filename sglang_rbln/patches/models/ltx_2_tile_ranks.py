# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 diffusion decoder tiles spread over NPUs as decode-parallel ranks.

Upstream spreads the decoder's stage-4/5 tiles over the ranks of its
decode-parallel group (``_tile_shard``) and gathers them back
(``_all_gather_tiles``) before blending. Here each NPU of the decoder's device
set is such a rank, run as a thread of the one server process: the upstream
tiled decode runs once per rank, a rank's tile graphs are compiled for its own
NPU, and the gather is in-process. The runtime releases the GIL while a graph
runs, so the ranks' tiles execute concurrently. Stages 1-3 precede the tiling
and run once; every rank reads their result.
"""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Callable, Sequence
from typing import Any

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

MODEL = "sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder"
TILED_DECODE = f"{MODEL}.LTX2VideoDiffusionDecoderModel.tiled_decode"
TILE_SHARD = f"{MODEL}.LTX2VideoDiffusionDecoderModel._tile_shard"
ALL_GATHER_TILES = f"{MODEL}._all_gather_tiles"
STAGES_1_TO_3 = f"{MODEL}.LTX2VideoDiffusionDecoder3d.forward_stages_1_to_3"
TILE_DEVICES = "_rbln_tile_devices"

_current = threading.local()


class _RankDone(BaseException):
    """A rank other than 0 has handed over its tiles: the rest is rank 0's."""


def hand_over(tile: torch.Tensor) -> torch.Tensor:
    """Bring a rank's tile to where rank 0 blends; host tensors need no move.

    The one place a device-to-device copy goes once tiles stay on their NPU.
    """
    return tile


class TileRanks:
    """One tiled decode, shared by the threads that act as its ranks."""

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.barrier = threading.Barrier(world_size)
        self.lock = threading.Lock()
        self.tiles: dict[int, torch.Tensor] = {}
        self.features: dict[int, Any] = {}
        self.features_ready = threading.Event()

    def gather(
        self, rank: int, local_tiles: Sequence, local_indices: Sequence, total: int
    ):
        with self.lock:
            self.tiles.update(
                (index, hand_over(tile))
                for index, tile in zip(local_indices, local_tiles, strict=True)
            )
        self.barrier.wait()
        # Only rank 0 returns the decode. Upstream blends the gathered tiles in
        # place, so a second rank blending them too would corrupt rank 0's.
        if rank != 0:
            raise _RankDone
        return [self.tiles[index] for index in range(total)]


def current_rank() -> tuple[int, TileRanks] | None:
    return getattr(_current, "rank", None)


def tile_devices(model) -> tuple[int, ...]:
    return getattr(model, TILE_DEVICES, ())


def tiled_decode_over_ranks(tiled_decode, model, hidden_states, *args, **kwargs):
    devices = tile_devices(model)
    if len(devices) < 2 or current_rank() is not None:
        return tiled_decode(model, hidden_states, *args, **kwargs)
    ranks = TileRanks(len(devices))
    positional = bool(args)
    generator = args[0] if positional else kwargs.get("generator")
    state = generator.get_state() if generator is not None else None
    # Grad and inference mode are per thread: carry the caller's to each rank.
    grad_enabled = torch.is_grad_enabled()
    inference = torch.is_inference_mode_enabled()

    def rank_call(rank: int):
        rank_generator = None
        if state is not None:
            rank_generator = torch.Generator(device=generator.device)
            rank_generator.set_state(state)
        if positional:
            call_args, call_kwargs = (rank_generator, *args[1:]), kwargs
        else:
            call_args, call_kwargs = args, {**kwargs, "generator": rank_generator}
        _current.rank = (rank, ranks)
        try:
            with torch.inference_mode(inference), torch.set_grad_enabled(grad_enabled):
                output = tiled_decode(model, hidden_states, *call_args, **call_kwargs)
            return output, rank_generator
        except _RankDone:
            return None, rank_generator
        except BaseException:
            # Release the ranks waiting on this one instead of hanging them.
            ranks.barrier.abort()
            ranks.features_ready.set()
            raise
        finally:
            _current.rank = None

    with concurrent.futures.ThreadPoolExecutor(len(devices)) as pool:
        futures = [pool.submit(rank_call, rank) for rank in range(len(devices))]
        outputs = [future.result() for future in futures]
    output, rank_generator = outputs[0]
    # Every rank walked the same draws, as a single rank would have.
    if generator is not None:
        generator.set_state(rank_generator.get_state())
    return output


def tile_shard_of_rank(tile_shard, model):
    rank = current_rank()
    if rank is None:
        return tile_shard(model)
    index, ranks = rank
    return index, ranks.world_size, ranks


def gather_rank_tiles(
    all_gather_tiles, local_tiles, local_indices, total, group, *args
):
    if isinstance(group, TileRanks):
        return group.gather(current_rank()[0], local_tiles, local_indices, total)
    return all_gather_tiles(local_tiles, local_indices, total, group, *args)


def shared_stages_1_to_3(forward, decoder, hidden_states, *args, **kwargs):
    rank = current_rank()
    if rank is None:
        return forward(decoder, hidden_states, *args, **kwargs)
    index, ranks = rank
    if index == 0:
        try:
            ranks.features["value"] = forward(decoder, hidden_states, *args, **kwargs)
        except BaseException as error:
            ranks.features["error"] = error
            raise
        finally:
            ranks.features_ready.set()
    ranks.features_ready.wait()
    if "error" in ranks.features:
        raise RuntimeError("stages 1-3 failed on rank 0") from ranks.features["error"]
    return ranks.features["value"]


class RankDispatch:
    """A decoder phase compiled once per NPU; each rank calls its own NPU's."""

    def __init__(self, compiled: Sequence[Callable]) -> None:
        self.compiled = list(compiled)
        self.lock = threading.Lock()
        self.seen: set[tuple] = set()

    def __call__(self, *args, **kwargs):
        rank = current_rank()
        index = rank[0] if rank is not None else 0
        fn = self.compiled[index % len(self.compiled)]
        signature = (index, *_shapes(args), *_shapes(kwargs.values()))
        if signature in self.seen:
            return fn(*args, **kwargs)
        # Dynamo compiles a new shape on its first call; keep that one at a time.
        with self.lock:
            output = fn(*args, **kwargs)
            self.seen.add(signature)
        return output


def _shapes(values) -> tuple:
    return tuple(
        (tuple(value.shape), value.dtype)
        if isinstance(value, torch.Tensor)
        else type(value).__name__
        for value in values
    )


def register() -> tuple[str, ...]:
    HookRegistry.register(TILED_DECODE, tiled_decode_over_ranks, HookType.AROUND)
    HookRegistry.register(TILE_SHARD, tile_shard_of_rank, HookType.AROUND)
    HookRegistry.register(ALL_GATHER_TILES, gather_rank_tiles, HookType.AROUND)
    HookRegistry.register(STAGES_1_TO_3, shared_stages_1_to_3, HookType.AROUND)
    return TILED_DECODE, TILE_SHARD, ALL_GATHER_TILES, STAGES_1_TO_3

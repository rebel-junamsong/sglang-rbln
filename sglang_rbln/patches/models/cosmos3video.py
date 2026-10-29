# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType
from torch import nn

MODULE = "sglang.multimodal_gen.runtime.models.dits.cosmos3video"
HOOK_TARGET = f"{MODULE}.Cosmos3TimestepEmbedder.forward"
QK_NORM_ROPE_HOOK_TARGET = f"{MODULE}._apply_qwen3_qk_norm_rope"
TEXT_KV_HOOK_TARGET = f"{MODULE}.Cosmos3UndDecoderLayer.forward"
ROPE_HOOK_TARGET = f"{MODULE}._apply_qwen3_rope_from_cache"
POSITION_IDS_HOOK_TARGET = f"{MODULE}.Cosmos3OmniTransformer._compute_rope_position_ids"
HOST_STATE = "_rbln_host_state"


def embed_timesteps_on_host(forward, embedder: nn.Module, t: torch.Tensor):
    device = next(embedder.parameters()).device
    if device.type != "rbln":
        return forward(embedder, t)
    host_state = getattr(embedder, HOST_STATE, None)
    if host_state is None:
        host_state = {
            name: tensor.cpu()
            for name, tensor in (
                *embedder.named_parameters(),
                *embedder.named_buffers(),
            )
        }
        object.__setattr__(embedder, HOST_STATE, host_state)
    with torch.no_grad():
        embedding = torch.func.functional_call(embedder, host_state, (t.cpu(),))
    return embedding.to(device)


def split_qk_norm_rope(
    q, k, q_norm, k_norm, head_dim, cos_sin_cache, positions, **kwargs
):
    from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
        _apply_qwen3_qk_norm_rope_split,
    )

    return _apply_qwen3_qk_norm_rope_split(
        q, k, q_norm, k_norm, head_dim, cos_sin_cache
    )


def rope_from_cache(
    q: torch.Tensor, k: torch.Tensor, cos_sin_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = q.shape[:2]
    half = q.shape[-1] // 2
    cos = cos_sin_cache[:, :half].view(batch_size, seq_len, 1, half).to(q.dtype)
    sin = cos_sin_cache[:, half:].view(batch_size, seq_len, 1, half).to(q.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    return rotate(q), rotate(k)


def grid_positions(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    device: torch.device,
    fps: float | None,
    base_fps: float,
    temporal_compression_factor: int,
    base_temporal_compression_factor: int | None = None,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, float, int]:
    frames = torch.arange(grid_t, device=device) + start_frame_offset
    last_frame = grid_t - 1 + start_frame_offset
    if fps is not None and grid_t > 1:
        base_factor = base_temporal_compression_factor or temporal_compression_factor
        tps, base_tps = fps / temporal_compression_factor, base_fps / base_factor
        frames = frames.float() / tps * base_tps
        last_frame = last_frame / tps * base_tps
    t = frames.view(-1, 1).expand(-1, grid_h * grid_w).flatten()
    h = torch.arange(grid_h, device=device).view(1, -1, 1).expand(grid_t, -1, grid_w)
    w = torch.arange(grid_w, device=device).view(1, 1, -1).expand(grid_t, grid_h, -1)
    positions = torch.stack([t, h.flatten().to(t.dtype), w.flatten().to(t.dtype)])
    return positions, last_frame, max(grid_h, grid_w) - 1


def shift_positions(
    grid: tuple[torch.Tensor, float, int], offset: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    positions, last_frame, last_hw = grid
    temporal = positions[:1].unsqueeze(1) + offset.view(1, -1, 1)
    spatial = positions[1:].unsqueeze(1).expand(-1, offset.shape[0], -1)
    next_offset = torch.ceil(torch.clamp_min(offset + last_frame, last_hw)).long() + 1
    return torch.cat([temporal, spatial.to(temporal.dtype)]), next_offset


def rope_position_ids(
    model: nn.Module,
    text_mask: torch.Tensor,
    T: int,
    Hp: int,
    Wp: int,
    fps: float | None,
    device: torch.device,
    sound_frames: int = 0,
    action_frames: int = 0,
    action_fps: float | None = None,
    action_start_frame_offset: int = 1,
    control_frames: int | Sequence[int] = 0,
    share_vision_temporal_positions: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_lengths = text_mask.sum(dim=1).long()
    tokens = torch.arange(text_mask.shape[1], device=text_mask.device)
    text_positions = torch.where(tokens < text_lengths.view(-1, 1), tokens, 0)
    if isinstance(control_frames, int):
        control_frames = [control_frames]
    frame_counts = [*(int(count) for count in control_frames if int(count) > 0), T]
    media_offset = text_lengths + model.temporal_margin
    vision_offset = media_offset
    segments = []
    for frame_count in frame_counts:
        grid = grid_positions(
            frame_count,
            Hp,
            Wp,
            text_mask.device,
            fps if T > 1 else None,
            model.base_fps,
            model.temporal_compression_factor,
        )
        offset = media_offset if share_vision_temporal_positions else vision_offset
        positions, vision_offset = shift_positions(grid, offset)
        segments.append(positions)
    if action_frames > 0:
        grid = grid_positions(
            action_frames,
            1,
            1,
            text_mask.device,
            action_fps,
            model.base_fps,
            1,
            model.temporal_compression_factor,
            action_start_frame_offset,
        )
        segments.append(shift_positions(grid, media_offset)[0])
    if sound_frames > 0:
        grid = grid_positions(
            sound_frames,
            1,
            1,
            text_mask.device,
            model.sound_latent_fps,
            model.base_fps,
            model.temporal_compression_factor_sound,
        )
        segments.append(shift_positions(grid, media_offset)[0])
    dtype = segments[0].dtype
    for positions in segments[1:]:
        dtype = torch.promote_types(dtype, positions.dtype)
    vision_positions = torch.cat([positions.to(dtype) for positions in segments], 2)
    text_positions = torch.stack([text_positions] * 3)
    return text_positions.to(device), vision_positions.to(device)


def contiguous_text_kv(forward, layer: nn.Module, *args, **kwargs):
    hidden_states, key, value = forward(layer, *args, **kwargs)
    return hidden_states, key.contiguous(), value.contiguous()


def register() -> tuple[str, ...]:
    HookRegistry.register(HOOK_TARGET, embed_timesteps_on_host, HookType.AROUND)
    HookRegistry.register(
        QK_NORM_ROPE_HOOK_TARGET, split_qk_norm_rope, HookType.REPLACE
    )
    HookRegistry.register(TEXT_KV_HOOK_TARGET, contiguous_text_kv, HookType.AROUND)
    HookRegistry.register(ROPE_HOOK_TARGET, rope_from_cache, HookType.REPLACE)
    HookRegistry.register(POSITION_IDS_HOOK_TARGET, rope_position_ids, HookType.REPLACE)
    return (
        HOOK_TARGET,
        QK_NORM_ROPE_HOOK_TARGET,
        TEXT_KV_HOOK_TARGET,
        ROPE_HOOK_TARGET,
        POSITION_IDS_HOOK_TARGET,
    )


def run_on_host(module: nn.Module) -> bool:
    if next(module.parameters()).device.type != "rbln":
        return False
    host_state = {
        name: tensor.cpu()
        for name, tensor in (*module.named_parameters(), *module.named_buffers())
    }
    device_forward = module.forward
    reentered = False

    def forward(*args):
        nonlocal reentered
        if reentered:
            return device_forward(*args)
        reentered = True
        try:
            with torch.no_grad():
                out = torch.func.functional_call(
                    module,
                    host_state,
                    tuple(
                        arg.cpu() if isinstance(arg, torch.Tensor) else arg
                        for arg in args
                    ),
                )
        finally:
            reentered = False
        return torch.utils._pytree.tree_map_only(
            torch.Tensor, lambda tensor: tensor.to(args[0].device), out
        )

    module.forward = forward
    return True

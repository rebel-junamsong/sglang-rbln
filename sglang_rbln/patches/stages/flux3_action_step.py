# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import importlib
import time

import torch
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln import envs
from sglang_rbln.patches.models.flux3 import Flux3SegmentState

STAGES = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".flux3_action"
)
OBSERVATION_TARGET = f"{STAGES}.Flux3ActionObservationEncodingStage.forward"
DENOISE_TARGET = f"{STAGES}.Flux3ActionDenoisingStage.forward"
_STATE_TENSORS = 6


def _state_tensors(state: Flux3SegmentState) -> tuple[torch.Tensor, ...]:
    return (state.hidden, state.rope, state.vec, *state.joint_mod)


def _state_from_tensors(
    name: str, tensors: tuple[torch.Tensor, ...]
) -> Flux3SegmentState:
    hidden, rope, vec, *joint_mod = tensors
    return Flux3SegmentState(
        name=name, hidden=hidden, rope=rope, vec=vec, joint_mod=tuple(joint_mod)
    )


def _states_from_tensors(
    names: list[str], tensors: tuple[torch.Tensor, ...]
) -> list[Flux3SegmentState]:
    return [
        _state_from_tensors(
            name, tensors[i * _STATE_TENSORS : (i + 1) * _STATE_TENSORS]
        )
        for i, name in enumerate(names)
    ]


def _encode_streams(
    self,
    video: torch.Tensor,
    frame_ids: torch.Tensor,
    action: torch.Tensor,
    action_ids: torch.Tensor,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    video_cond = self.transformer.encode_stream(
        name="video_cond", x=video, ids=frame_ids, timesteps=zero
    )
    action_cond = self.transformer.encode_stream(
        name=f"{self.config.action_modality}_cond",
        x=action,
        ids=action_ids,
        timesteps=zero,
    )
    return (*_state_tensors(video_cond), *_state_tensors(action_cond))


def _target_streams(
    self,
    samples: tuple[torch.Tensor, ...],
    ropes: tuple[torch.Tensor, ...],
    t: torch.Tensor,
    *,
    order: list[str],
) -> list[Flux3SegmentState]:
    return [
        self.transformer.encode_stream(
            name=name,
            x=x.to(torch.bfloat16),
            ids=None,
            timesteps=t,
            rope=rope,
        )
        for name, x, rope in zip(order, samples, ropes, strict=True)
    ]


def _denoise_pass(
    self,
    context: Flux3SegmentState,
    targets: list[Flux3SegmentState],
    conditioning: list[Flux3SegmentState],
    *,
    order: list[str],
) -> tuple[torch.Tensor, ...]:
    video, action = targets
    video_cond, action_cond = conditioning
    out = self.transformer.denoise(
        context=context,
        streams=[video, video_cond, action, action_cond],
        targets=order,
    )
    return tuple(out[k] for k in order)


def _denoise_step(
    self,
    video: torch.Tensor,
    action: torch.Tensor,
    t: torch.Tensor,
    video_rope: torch.Tensor,
    action_rope: torch.Tensor,
    *states: torch.Tensor,
    order: list[str],
    state_names: list[str],
) -> tuple[torch.Tensor, ...]:
    video_cond, action_cond, *contexts = _states_from_tensors(state_names, states)
    targets = self._target_streams(
        (video, action), (video_rope, action_rope), t, order=order
    )
    return tuple(
        p
        for ctx in contexts
        for p in self._denoise_pass(
            ctx, targets, [video_cond, action_cond], order=order
        )
    )


def encode_observation(self, batch, server_args):
    stages = importlib.import_module(STAGES)
    start = time.perf_counter()
    cfg = self.config
    device = stages.get_local_torch_device()
    state = stages.vla_state(batch)
    observation = state["flux3_observation"]
    h, w = cfg.latent_hw
    with self.use_declared_component(component_name="vae", module=self.vae):
        frame = observation.canvas.to(device, torch.bfloat16)[None]
        latent = self.vae.encode_frame(frame)[..., :h, :w]
    video, video_ids = stages.pack_video(latent, first_frame=0, fps=cfg.fps)
    action, action_ids = self._state_tokens(observation.state, device)
    stream_inputs = (
        video_ids.to(device),
        action,
        action_ids.to(device),
        torch.zeros(1, device=device),
    )
    with (
        self.use_declared_component(
            component_name="transformer", module=self.transformer
        ),
        stages.set_forward_context(current_timestep=0, attn_metadata=None),
    ):
        tensors = self._encode_streams(video, *stream_inputs)
    state["flux3_conditioning"] = _states_from_tensors(
        ["video_cond", f"{cfg.action_modality}_cond"], tensors
    )
    stages.vla_timings(batch)["observation_ms"] = (time.perf_counter() - start) * 1000
    return batch


def denoise(self, batch, server_args):
    stages = importlib.import_module(STAGES)
    cfg = self.config
    device = stages.get_local_torch_device()
    state = stages.vla_state(batch)
    observation = state["flux3_observation"]
    contexts = state["flux3_contexts"]
    conditioning = state["flux3_conditioning"]
    guidance = state["flux3_guidance"]
    steps = batch.num_inference_steps
    seed = batch.seed[0] if isinstance(batch.seed, list) else batch.seed
    horizon = batch.action_horizon or cfg.action_horizon
    if horizon > cfg.action_horizon:
        raise ValueError(
            f"action_horizon {horizon} exceeds the policy chunk {cfg.action_horizon}"
        )

    start = time.perf_counter()
    samples, ids = self._noised_streams(seed)
    samples = {k: v.to(device) for k, v in samples.items()}
    order = list(samples)
    ropes = tuple(self.transformer.rope(ids[k].to(device)) for k in order)
    cfg_policy = stages._cfg_parallel_policy(contexts, server_args)
    state_names = [s.name for s in (*conditioning, *contexts)]
    states = tuple(t for s in (*conditioning, *contexts) for t in _state_tensors(s))
    denoise_step = functools.partial(
        self._denoise_step, order=order, state_names=state_names
    )
    step = 0

    def predict(current: dict[str, torch.Tensor], t: float) -> dict[str, torch.Tensor]:
        nonlocal step
        timestep = torch.full((1,), t, device=device, dtype=torch.float32)
        current_streams = tuple(current[k] for k in order)
        with stages.set_forward_context(
            current_timestep=step, attn_metadata=None, forward_batch=batch
        ):
            if cfg_policy is not None:
                targets = self._target_streams(
                    current_streams, ropes, timestep, order=order
                )
                raw = stages.run_cfg_parallel(
                    cfg_policy,
                    lambda branch: self._denoise_pass(
                        branch.kwargs["context"], targets, conditioning, order=order
                    ),
                )
            else:
                flat = denoise_step(*current_streams, timestep, *ropes, *states)
                raw = [
                    flat[i : i + len(order)] for i in range(0, len(flat), len(order))
                ]
            preds = [dict(zip(order, p, strict=True)) for p in raw]
        step += 1
        if len(preds) == 1:
            return {k: v.float() for k, v in preds[0].items()}
        cond, uncond = preds
        return {
            k: (uncond[k] + guidance[k] * (cond[k] - uncond[k])).float() for k in order
        }

    with self.use_declared_component(
        component_name="transformer", module=self.transformer
    ):
        result = stages.cosmos_unipc(
            samples,
            predict,
            scheduler=self.scheduler,
            n_steps=steps,
            shift=cfg.sampler_shift,
        )
    targets = result[cfg.action_modality][0].float() / cfg.action_scale
    actions = stages.targets_to_actions(
        targets, state=observation.state.to(device), config=cfg
    )
    state["actions"] = actions[None, :horizon]
    stages.synchronize_vla_action_tensor(actions)
    stages.vla_timings(batch)["denoise_ms"] = (time.perf_counter() - start) * 1000
    return batch


BACKPORTED = {
    "Flux3ActionObservationEncodingStage": (_encode_streams,),
    "Flux3ActionDenoisingStage": (_target_streams, _denoise_pass, _denoise_step),
}
FORWARDS = {OBSERVATION_TARGET: encode_observation, DENOISE_TARGET: denoise}


def register() -> tuple[str, ...]:
    if not envs.SGLANG_RBLN_FLUX3_ONE_GRAPH:
        return ()
    stages = importlib.import_module(STAGES)
    for stage, methods in BACKPORTED.items():
        for method in methods:
            setattr(getattr(stages, stage), method.__name__, method)
    for target, forward in FORWARDS.items():
        HookRegistry.register(target, forward, HookType.REPLACE)
    return tuple(FORWARDS)

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import weakref
from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers.video_processor import VideoProcessor
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import AutoencoderKLLTX2Video
from sglang.multimodal_gen.runtime.platforms.plugins import HookRegistry, HookType

from sglang_rbln.patches.models import ltx_2_vae_assembly as assembly

UPSTREAM = {
    name: getattr(AutoencoderKLLTX2Video, name)
    for name in (
        "tiled_decode",
        "_temporal_tiled_decode",
        "blend_v",
        "blend_h",
        "blend_t",
    )
}


def fake_vae(
    dtype, tiled_decode, temporal_tiled_decode, *, frames=24, size=96, stride=64
):
    """A VAE whose decoder is a fixed random upsampling, tiled like LTX-2's."""
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(3, 4, generator=generator).to(dtype)

    def decoder(z, temb, causal=None):
        x = torch.einsum("oc,bcthw->bothw", weight, z)
        x = x.repeat_interleave(8, 3).repeat_interleave(8, 4)
        return torch.cat([x[:, :, :1], x.repeat_interleave(4, 2)[:, :, 4:]], 2)

    vae = SimpleNamespace(
        decoder=decoder,
        use_tiling=True,
        spatial_compression_ratio=8,
        temporal_compression_ratio=4,
        tile_sample_min_height=size,
        tile_sample_min_width=size,
        tile_sample_stride_height=stride,
        tile_sample_stride_width=stride,
        tile_sample_min_num_frames=frames,
        tile_sample_stride_num_frames=frames - 8,
    )
    for name in ("blend_v", "blend_h", "blend_t"):
        setattr(vae, name, MethodType(UPSTREAM[name], vae))
    vae.tiled_decode = MethodType(tiled_decode, vae)
    vae._temporal_tiled_decode = MethodType(temporal_tiled_decode, vae)
    return vae


def latent(dtype, frames=9, height=20, width=28):
    generator = torch.Generator().manual_seed(1)
    return torch.randn(1, 4, frames, height, width, generator=generator).to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("name", ["blend_v", "blend_h", "blend_t"])
@pytest.mark.parametrize("extent", [0, 1, 7, 64, 200])
def test_blend_matches_upstream_bit_for_bit(name, dtype, extent):
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(1, 3, 9, 96, 80, generator=generator).to(dtype)
    b = torch.randn(1, 3, 9, 96, 80, generator=generator).to(dtype)
    dim = {"blend_v": 3, "blend_h": 4, "blend_t": 2}[name]
    expected = UPSTREAM[name](None, a.clone(), b.clone(), extent)
    actual = assembly.blend(a, b, extent, dim)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "shape", [(9, 20, 28), (9, 8, 8), (5, 12, 20)], ids=["2x3", "1x1", "short"]
)
def test_spatial_join_matches_upstream_bit_for_bit(dtype, shape):
    z = latent(dtype, *shape)
    upstream = fake_vae(
        dtype, UPSTREAM["tiled_decode"], UPSTREAM["_temporal_tiled_decode"]
    )
    ours = fake_vae(dtype, assembly.tiled_decode, assembly.temporal_tiled_decode)
    expected = upstream.tiled_decode(z, None).sample
    actual = ours.tiled_decode(z, None).sample
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("frames", [9, 13, 16], ids=["chunks", "short-last", "uneven"])
def test_temporal_join_matches_upstream_bit_for_bit(dtype, frames):
    z = latent(dtype, frames)
    upstream = fake_vae(
        dtype, UPSTREAM["tiled_decode"], UPSTREAM["_temporal_tiled_decode"], frames=16
    )
    ours = fake_vae(
        dtype, assembly.tiled_decode, assembly.temporal_tiled_decode, frames=16
    )
    expected = upstream._temporal_tiled_decode(z, None).sample
    actual = ours._temporal_tiled_decode(z, None).sample
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_a_chunk_left_with_no_frames_is_not_joined(monkeypatch):
    """The last temporal chunk can decode to a single frame, which upstream drops: an
    empty join that a compiled graph cannot run."""
    joined = []
    real = assembly._assembly

    def recording(vae, key, build, *tensors):
        joined.append([tuple(tensor.shape) for tensor in tensors])
        return real(vae, key, build, *tensors)

    monkeypatch.setattr(assembly, "_assembly", recording)
    z = latent(torch.float32, 9)
    ours = fake_vae(
        torch.float32, assembly.tiled_decode, assembly.temporal_tiled_decode, frames=16
    )
    upstream = fake_vae(
        torch.float32,
        UPSTREAM["tiled_decode"],
        UPSTREAM["_temporal_tiled_decode"],
        frames=16,
    )
    assert torch.equal(
        ours._temporal_tiled_decode(z, None).sample,
        upstream._temporal_tiled_decode(z, None).sample,
    )
    temporal = [
        shapes for shapes in joined if len(shapes) == 2 and shapes[0][3] == shapes[1][3]
    ]
    assert temporal and all(shapes[1][2] > 1 for shapes in temporal)


POSTPROCESS = VideoProcessor.postprocess_video


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_video_is_postprocessed_while_it_decodes(dtype):
    # Run after the decode, upstream's postprocess held the stage on the host for
    # 1.7 s at 1280x704x121 before the audio graph.
    ours = fake_vae(dtype, assembly.tiled_decode, assembly.temporal_tiled_decode)
    video = ours._temporal_tiled_decode(latent(dtype, 13), None).sample
    processor = VideoProcessor(vae_scale_factor=32)
    expected = POSTPROCESS(processor, video, output_type="np")
    called = []

    def upstream(*args, **kwargs):
        called.append(args)
        return POSTPROCESS(*args, **kwargs)

    actual = assembly.postprocess_video(upstream, processor, video, output_type="np")
    assert called == []
    assert actual.dtype == expected.dtype and np.array_equal(actual, expected)


@pytest.mark.parametrize("output_type", ["pt", "np"])
def test_any_other_postprocess_is_upstreams(output_type):
    ours = fake_vae(
        torch.float32, assembly.tiled_decode, assembly.temporal_tiled_decode
    )
    video = ours._temporal_tiled_decode(latent(torch.float32, 9), None).sample
    processor = VideoProcessor(vae_scale_factor=32)
    # "np" with its own denormalize choice per frame is not the precomputed array
    extra = {} if output_type == "pt" else {"do_denormalize": [False] * video.shape[2]}
    actual = assembly.postprocess_video(
        POSTPROCESS, processor, video, output_type, **extra
    )
    expected = POSTPROCESS(processor, video, output_type, **extra)
    assert type(actual) is type(expected)
    assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_host_tiles_are_not_compiled(monkeypatch):
    def compile_called(*args, **kwargs):
        raise AssertionError("host tiles must not compile an assembly")

    monkeypatch.setattr(torch.nn.Module, "compile", compile_called)
    ours = fake_vae(
        torch.float32, assembly.tiled_decode, assembly.temporal_tiled_decode
    )
    ours._temporal_tiled_decode(latent(torch.float32), None)


@pytest.mark.parametrize("target", assembly.TARGETS)
def test_replaces_the_upstream_joins(target):
    assert target in assembly.register()
    assert any(
        hook is assembly.TARGETS[target] and kind is HookType.REPLACE
        for kind, hook, _ in HookRegistry._hooks[target]
    )


class EagerDeviceOps(torch.overrides.TorchFunctionMode):
    """The torch calls made eagerly on rbln tensors, by name."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        leaves = [*args, *kwargs.values()]
        on_device = any(
            isinstance(leaf, torch.Tensor) and leaf.device.type == "rbln"
            for leaf in leaves
        )
        if on_device and not torch.compiler.is_compiling():
            self.names.append(getattr(func, "__name__", str(func)))
        return func(*args, **kwargs)


def test_device_tiles_are_joined_in_one_graph(dummy_device, monkeypatch, tmp_path):
    """Eager, each blend, slice and cat over the decoded tiles ran as its own op on
    the device, taking every tile to one NPU or to the host; the join is one graph."""
    from sglang_rbln.compilation import residency

    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compile_module = torch.nn.Module.compile

    def compile_resident(module, *args, **kwargs):
        # as the server's residency hook does: the call takes its inputs to the NPUs
        compile_module(module, *args, **kwargs)
        residency.install_resident_call(module)

    monkeypatch.setattr(torch.nn.Module, "compile", compile_resident)
    vae = fake_vae(
        torch.bfloat16, assembly.tiled_decode, assembly.temporal_tiled_decode
    )
    decode_on_host = vae.decoder
    vae.decoder = lambda z, temb, causal=None: decode_on_host(z, temb).to("rbln:0")
    eager = EagerDeviceOps()
    with residency.DeviceBoundary(), eager:
        joined = vae.tiled_decode(latent(torch.bfloat16), None).sample
    assert joined.device.type == "cpu"
    assert len(vae.__dict__[assembly.ASSEMBLIES]) == 1
    assert not {"cat", "mul", "add", "__getitem__"} & set(eager.names)
    # the joined video comes down once, not tile by tile
    assert eager.names.count("cpu") == 1


class LiveDeviceTensors(torch.overrides.TorchFunctionMode):
    """The most tensors of one size alive on rbln at once, over every torch call."""

    def __init__(self, numel: int) -> None:
        super().__init__()
        self.numel = numel
        self.seen: dict[int, weakref.ref] = {}
        self.peak = 0

    def _note(self, values) -> None:
        for value in values:
            if (
                isinstance(value, torch.Tensor)
                and value.device.type == "rbln"
                and value.numel() == self.numel
            ):
                self.seen[value.untyped_storage().data_ptr()] = weakref.ref(value)
        self.seen = {ptr: ref for ptr, ref in self.seen.items() if ref() is not None}
        self.peak = max(self.peak, len(self.seen))

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        self._note([*args, *kwargs.values()])
        result = func(*args, **kwargs)
        self._note(result if isinstance(result, (tuple, list)) else [result])
        return result


def test_a_device_decode_keeps_at_most_one_video_on_the_npu(
    dummy_device, monkeypatch, tmp_path
):
    """Left on the decoder's NPUs, the joined video went through upstream's
    post-processing there, every op adding a full-size tensor beside it: four at
    once on the NPU the DiT shares, which slowed the DiT's next request."""
    from sglang_rbln.compilation import residency

    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compile_module = torch.nn.Module.compile

    def compile_resident(module, *args, **kwargs):
        compile_module(module, *args, **kwargs)
        residency.install_resident_call(module)

    monkeypatch.setattr(torch.nn.Module, "compile", compile_resident)
    vae = fake_vae(
        torch.bfloat16, assembly.tiled_decode, assembly.temporal_tiled_decode
    )
    decode_on_host = vae.decoder
    vae.decoder = lambda z, temb, causal=None: decode_on_host(z, temb).to("rbln:0")
    z = latent(torch.bfloat16)
    numel = UPSTREAM["tiled_decode"](
        fake_vae(
            torch.bfloat16, UPSTREAM["tiled_decode"], UPSTREAM["_temporal_tiled_decode"]
        ),
        z,
        None,
    ).sample.numel()
    live = LiveDeviceTensors(numel)
    with residency.DeviceBoundary(), live:
        video = vae.tiled_decode(z, None).sample
        POSTPROCESS(VideoProcessor(vae_scale_factor=32), video, output_type="np")
    assert live.peak <= 1


def test_a_device_decode_is_postprocessed_across_frame_threads(
    dummy_device, monkeypatch, tmp_path
):
    """The joined video comes down whole; upstream's post-processing of it in one
    call held the stage 0.87 s before the audio graph at 1280x704x121, where chunks
    of frames across threads take 0.36 s and give the same array."""
    from sglang_rbln.compilation import residency

    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compile_module = torch.nn.Module.compile

    def compile_resident(module, *args, **kwargs):
        compile_module(module, *args, **kwargs)
        residency.install_resident_call(module)

    monkeypatch.setattr(torch.nn.Module, "compile", compile_resident)
    vae = fake_vae(
        torch.bfloat16, assembly.tiled_decode, assembly.temporal_tiled_decode
    )
    decode_on_host = vae.decoder
    vae.decoder = lambda z, temb, causal=None: decode_on_host(z, temb).to("rbln:0")
    with residency.DeviceBoundary():
        video = vae.tiled_decode(latent(torch.bfloat16), None).sample
    assert video.device.type == "cpu"
    ready = getattr(video, assembly.POSTPROCESSED)
    assert len(ready.jobs) == -(-video.shape[2] // assembly.FRAME_CHUNK) > 1
    processor = VideoProcessor(vae_scale_factor=32)
    called = []

    def upstream(*args, **kwargs):
        called.append(args)
        return POSTPROCESS(*args, **kwargs)

    actual = assembly.postprocess_video(upstream, processor, video, output_type="np")
    expected = POSTPROCESS(processor, video, output_type="np")
    assert called == []
    assert actual.dtype == expected.dtype and np.array_equal(actual, expected)


def save_path_frames(processor, video):
    """The bytes the server writes for upstream's float32 frames of ``video``."""
    from sglang.multimodal_gen.runtime.entrypoints import utils

    from sglang_rbln.patches.entrypoints import frames

    expected = POSTPROCESS(processor, video, output_type="np")
    return np.stack(
        [
            np.stack(frames.sample_to_uint8_frames(utils._sample_to_uint8_frames, clip))
            for clip in expected
        ]
    )


def edge_video(dtype, frames, height, width):
    """A decoded video whose values include the clamps' ends and every uint8 step."""
    generator = torch.Generator().manual_seed(2)
    video = torch.rand(1, 3, frames, height, width, generator=generator) * 2.4 - 1.2
    steps = torch.arange(256, dtype=torch.float32) / 255 * 2 - 1
    flat = video.view(-1)
    edges = torch.tensor([-1.0, 1.0, -1.5, 1.5, 0.0, -0.0, float("inf"), -float("inf")])
    flat[: edges.numel()] = edges
    flat[edges.numel() : edges.numel() + 3 * 256] = torch.cat(
        [steps, steps + 1e-3, steps - 1e-3]
    )
    return video.to(dtype)


def device_tiled_vae(monkeypatch, tmp_path, dtype):
    """A fake VAE whose decoded tiles sit on the (dummy) NPU, as a served decode's
    do, so the spatial join and its streamed postprocess run."""
    from sglang_rbln.compilation import residency

    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compile_module = torch.nn.Module.compile

    def compile_resident(module, *args, **kwargs):
        compile_module(module, *args, **kwargs)
        residency.install_resident_call(module)

    monkeypatch.setattr(torch.nn.Module, "compile", compile_resident)
    vae = fake_vae(dtype, assembly.tiled_decode, assembly.temporal_tiled_decode)
    decode_on_host = vae.decoder
    vae.decoder = lambda z, temb, causal=None: decode_on_host(z, temb).to("rbln:0")
    return vae


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("path", ["spatial", "temporal"])
def test_uint8_frames_are_the_save_paths_bytes(
    dtype, path, dummy_device, monkeypatch, tmp_path
):
    from sglang_rbln.compilation import residency

    z = latent(dtype, 13)
    with assembly.uint8_frames(), residency.DeviceBoundary():
        if path == "spatial":
            ours = device_tiled_vae(monkeypatch, tmp_path, dtype)
            video = ours.tiled_decode(z, None).sample
        else:
            ours = fake_vae(
                dtype, assembly.tiled_decode, assembly.temporal_tiled_decode
            )
            video = ours._temporal_tiled_decode(z, None).sample
        assert hasattr(video, assembly.POSTPROCESSED)
        processor = VideoProcessor(vae_scale_factor=32)
        called = []

        def upstream(*args, **kwargs):
            called.append(args)
            return POSTPROCESS(*args, **kwargs)

        actual = assembly.postprocess_video(
            upstream, processor, video, output_type="np"
        )
    assert called == []
    expected = save_path_frames(processor, video)
    assert actual.dtype == np.uint8 and actual.shape == expected.shape
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "size", [(544, 960), (704, 1280)], ids=["cookbook-960x544", "yak-1280x704"]
)
@pytest.mark.parametrize("chunked", [True, False], ids=["whole", "streamed"])
def test_uint8_frames_at_the_served_sizes(dtype, size, chunked):
    video = edge_video(dtype, 9, *size)
    with assembly.uint8_frames():
        ready = assembly.FramePostprocess(video.shape)
        if chunked:
            ready.add_in_chunks(video)
        else:
            for start in range(0, video.shape[2], 4):
                ready.add(video[:, :, start : start + 4], start)
        actual = ready.result()
    expected = save_path_frames(VideoProcessor(vae_scale_factor=32), video)
    assert actual.dtype == np.uint8
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("made_as_uint8", [True, False])
def test_a_caller_that_asks_the_other_type_gets_upstreams(made_as_uint8):
    ours = fake_vae(
        torch.bfloat16, assembly.tiled_decode, assembly.temporal_tiled_decode
    )
    z = latent(torch.bfloat16, 13)
    if made_as_uint8:
        with assembly.uint8_frames():
            video = ours._temporal_tiled_decode(z, None).sample
    else:
        video = ours._temporal_tiled_decode(z, None).sample
    processor = VideoProcessor(vae_scale_factor=32)
    asks_uint8 = assembly.uint8_frames() if not made_as_uint8 else nullcontext()
    with asks_uint8:
        actual = assembly.postprocess_video(
            POSTPROCESS, processor, video, output_type="np"
        )
    expected = POSTPROCESS(processor, video, output_type="np")
    assert actual.dtype == np.float32 and np.array_equal(actual, expected)

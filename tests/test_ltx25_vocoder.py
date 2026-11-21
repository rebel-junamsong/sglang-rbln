# SPDX-License-Identifier: Apache-2.0
import functools
import logging
from types import SimpleNamespace

import pytest
import torch
from sglang.multimodal_gen.configs.models.vocoder.ltx_vocoder import (
    LTXVocoderArchConfig,
    LTXVocoderConfig,
)
from sglang.multimodal_gen.runtime.models.vocoder.ltx_2_vocoder import LTX2Vocoder
from torch import nn

from sglang_rbln.compilation import backend as compilation
from sglang_rbln.compilation import placement
from sglang_rbln.patches.stages.ltx_2 import decoding_av, vocoder

pytestmark = pytest.mark.usefixtures("rbln_platform")


def upstream_forward(stage, batch, server_args):
    return "output"


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.full((), 2.0, dtype=torch.bfloat16))

    def forward(self, latents):
        return latents * self.scale


class Vocoder(nn.Module):
    def forward(self, spectrogram):
        return spectrogram.float() + 1


class AudioVae(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        # 2 channels x 4 mel bins: std 3, mean 1 everywhere
        self.register_buffer("latents_mean", torch.ones(8))
        self.register_buffer("latents_std", torch.full((8,), 3.0))

    def decode(self, latents, return_dict=True):
        return (self.decoder(latents),)


@pytest.fixture(name="audio_stage")
def audio_stage_fixture():
    return SimpleNamespace(audio_vae=AudioVae(), vocoder=Vocoder())


@pytest.fixture
def compiled_modules(monkeypatch):
    calls = []
    monkeypatch.setattr(
        nn.Module, "compile", lambda module, **kwargs: calls.append((module, kwargs))
    )
    return calls


def audio_forward(stage, batch, server_args):
    """The stage's audio path: denormalize (if the VAE still has statistics),
    decode, vocoder."""
    latents = torch.ones(1, 2, 3, 4)
    mean, std = stage.audio_vae.latents_mean, stage.audio_vae.latents_std
    if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
        latents = latents * std.view(1, 2, 1, 4) + mean.view(1, 2, 1, 4)
    spectrogram = stage.audio_vae.decode(latents, return_dict=False)[0]
    return stage.vocoder(spectrogram)


def test_compiles_the_audio_chain_once(audio_stage, compiled_modules, monkeypatch):
    # no text-encoder NPUs, so the chain runs across the DiT's
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    monkeypatch.setattr(placement, "text_encoder_devices", lambda: None)
    server_args = SimpleNamespace(enable_torch_compile=True)
    waveforms = [
        vocoder.compile_audio(audio_forward, audio_stage, None, server_args)
        for _ in range(2)
    ]
    assert len(compiled_modules) == 1
    chain, kwargs = compiled_modules[0]
    assert isinstance(chain, vocoder.AudioChain)
    assert (kwargs["backend"], kwargs["dynamic"]) == ("rbln", False)
    assert kwargs["options"]["num_devices"] == 4
    assert "device" not in kwargs["options"]
    # The stage denormalizes (1 * 3 + 1), the chain decodes (x 2) and vocodes (+ 1).
    assert audio_stage.audio_vae.latents_mean is not None
    for waveform in waveforms:
        assert torch.equal(waveform, torch.full((1, 2, 3, 4), 9.0))


@pytest.mark.parametrize("frames", [3, 25], ids=lambda n: f"{n}frames")
def test_chain_takes_the_latents_and_returns_the_waveform(audio_stage, frames):
    chain = vocoder.AudioChain(audio_stage.audio_vae, audio_stage.vocoder)
    waveform = chain(torch.ones(1, 2, frames, 4))
    assert torch.equal(waveform, torch.full((1, 2, frames, 4), 3.0))


def test_chain_runs_on_the_vocoders_placed_npu(audio_stage, compiled_modules):
    placement.place(audio_stage.vocoder, (8,))
    server_args = SimpleNamespace(enable_torch_compile=True)
    vocoder.compile_audio(audio_forward, audio_stage, None, server_args)
    options = compiled_modules[0][1]["options"]
    assert (options["num_devices"], options["device"]) == (1, [8])


def test_leaves_the_audio_modules_eager_without_torch_compile(
    audio_stage, compiled_modules
):
    server_args = SimpleNamespace(enable_torch_compile=False)
    vocoder.compile_audio(upstream_forward, audio_stage, None, server_args)
    assert compiled_modules == []
    assert audio_stage.audio_vae.latents_mean is not None
    assert torch.equal(
        audio_forward(audio_stage, None, None), torch.full((1, 2, 3, 4), 9.0)
    )


def test_decode_hook_compiles_video_decoder_and_audio_chain(compiled_modules):
    av_stage = SimpleNamespace(
        vae=SimpleNamespace(decoder=nn.Linear(4, 4)),
        audio_vae=AudioVae(),
        vocoder=Vocoder(),
    )
    batch = SimpleNamespace(use_diffusion_decoder=False)
    server_args = SimpleNamespace(enable_torch_compile=True)
    assert (
        decoding_av.compile_decoders(upstream_forward, av_stage, batch, server_args)
        == "output"
    )
    modules = [module for module, _ in compiled_modules]
    assert modules[0] is av_stage.vae.decoder
    assert isinstance(modules[1], vocoder.AudioChain)
    assert len(modules) == 2


def small_bwe_vocoder() -> LTX2Vocoder:
    arch = LTXVocoderArchConfig(
        hidden_channels=64,
        upsample_kernel_sizes=[11, 4],
        upsample_factors=[5, 2],
        resnet_kernel_sizes=[3, 7],
        resnet_dilations=[[1, 3, 5], [1, 3, 5]],
        act_fn="snakebeta",
        antialias=True,
        input_sampling_rate=16000,
        output_sampling_rate=48000,
        filter_length=32,
        window_length=32,
        hop_length=10,
        num_mel_channels=64,
        bwe_act_fn="snakebeta",
        bwe_final_bias=False,
        bwe_hidden_channels=32,
        bwe_upsample_factors=[6, 5],
        bwe_upsample_kernel_sizes=[12, 11],
        bwe_resnet_kernel_sizes=[3],
        bwe_resnet_dilations=[[1, 3, 5]],
    )
    model = LTX2Vocoder(LTXVocoderConfig(arch_config=arch))
    for buffer in model.mel_stft.buffers():
        buffer.copy_(torch.rand_like(buffer))
    return model.eval()


@pytest.mark.npu
def test_vocoder_runs_on_the_npu(caplog, monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_RBLN_CACHE_DIR", str(tmp_path))
    compilation.ensure_compile_backend()
    torch.manual_seed(0)
    model = small_bwe_vocoder()
    spectrogram = torch.randn(1, 2, 64, 64)
    with torch.no_grad():
        reference = model(spectrogram)
        compiled = torch.compile(
            model,
            backend=compilation.COMPILE_BACKEND_NAME,
            options=compilation.compile_options(model),
            fullgraph=True,
            dynamic=False,
        )
        with caplog.at_level(logging.WARNING, logger="rebel-compiler"):
            output = compiled(spectrogram)
    assert "Fallback to eager" not in caplog.text
    assert output.shape == reference.shape == (1, 2, 1920)
    snr = 10 * torch.log10(reference.pow(2).sum() / (output - reference).pow(2).sum())
    assert snr > 30


def test_the_audio_chain_compiles_for_the_text_encoder_npus(
    audio_stage, compiled_modules, monkeypatch
):
    # the DiT's set has no room for it beside both DiT shapes
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    monkeypatch.setattr(placement, "text_encoder_devices", lambda: (4, 5, 6, 7))
    compile_on = SimpleNamespace(enable_torch_compile=True)
    vocoder.compile_audio(audio_forward, audio_stage, None, compile_on)
    options = compiled_modules[0][1]["options"]
    assert (options["num_devices"], options["device"]) == (4, [4, 5, 6, 7])


def test_latents_in_another_layout_reuse_the_compiled_chain(audio_stage, monkeypatch):
    """The stage's latents come contiguous on the first request and as a transposed
    view from the second on; the layout-static chain traced them again."""
    monkeypatch.setattr(placement, "text_encoder_devices", lambda: None)
    compiles = []

    def counting(graph, example_inputs):
        compiles.append(graph)
        return graph.forward

    monkeypatch.setattr(
        vocoder,
        "compile_whole",
        lambda module, what: module.compile(backend=counting, dynamic=False),
    )
    torch._dynamo.reset()
    server_args = SimpleNamespace(enable_torch_compile=True)
    contiguous = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    transposed = torch.randn(1, 3, 2, 4, dtype=torch.bfloat16).transpose(1, 2)

    def stage_audio(stage, batch, server_args, latents):
        spectrogram = stage.audio_vae.decode(latents, return_dict=False)[0]
        return stage.vocoder(spectrogram)

    for latents in (contiguous, transposed):
        waveform = vocoder.compile_audio(
            functools.partial(stage_audio, latents=latents),
            audio_stage,
            None,
            server_args,
        )
        assert torch.equal(waveform, latents.float() * 2 + 1)
    assert len(compiles) == 1


def test_the_stage_takes_its_frames_as_uint8(compiled_modules):
    from sglang_rbln.patches.models import ltx_2_vae_assembly as assembly

    av_stage = SimpleNamespace(
        vae=SimpleNamespace(decoder=nn.Linear(4, 4)),
        audio_vae=AudioVae(),
        vocoder=Vocoder(),
    )
    batch = SimpleNamespace(use_diffusion_decoder=False)
    server_args = SimpleNamespace(enable_torch_compile=True)
    seen = []

    def forward(stage, batch, server_args):
        seen.append(assembly._UINT8_FRAMES.get())
        return "output"

    assert (
        decoding_av.compile_decoders(forward, av_stage, batch, server_args) == "output"
    )
    assert seen == [True]
    assert assembly._UINT8_FRAMES.get() is False


class RecordingVocoder(Vocoder):
    """The vocoder, noting the thread each run is on."""

    def __init__(self):
        super().__init__()
        self.threads = []

    def forward(self, spectrogram):
        import threading

        self.threads.append(threading.current_thread().name)
        return super().forward(spectrogram)


def upstream_audio(stage, batch, server_args):
    """LTX2AVDecodingStage's audio half: denormalize in the audio VAE's dtype,
    decode, align to the vocoder's dtype, vocoder."""
    from sglang.multimodal_gen.runtime.utils.precision import (
        align_tensor_to_module_dtype,
    )

    latents = batch.audio_latents.to(torch.bfloat16)
    mean = stage.audio_vae.latents_mean.to(torch.bfloat16).view(1, 2, 1, 4)
    std = stage.audio_vae.latents_std.to(torch.bfloat16).view(1, 2, 1, 4)
    spectrogram = stage.audio_vae.decode(latents * std + mean, return_dict=False)[0]
    spectrogram = align_tensor_to_module_dtype(spectrogram, stage.vocoder)
    return stage.vocoder(spectrogram)


AUDIO_ARGS = SimpleNamespace(
    enable_torch_compile=True,
    component_precisions={},
    pipeline_config=SimpleNamespace(audio_vae_precision="bf16"),
)


@pytest.fixture(name="av_stage")
def av_stage_fixture(monkeypatch):
    """A stage whose audio chain has run once on the spare NPUs, with the video
    decoder on the DiT's."""
    monkeypatch.setenv("SGLANG_RBLN_NUM_DEVICES", "4")
    stage = SimpleNamespace(
        vae=SimpleNamespace(decoder=nn.Linear(4, 4)),
        audio_vae=AudioVae(),
        vocoder=RecordingVocoder(),
    )
    monkeypatch.setattr(vocoder, "compile_whole", lambda module, what: None)
    placement.place(stage.vocoder, (4, 5, 6, 7))
    batch = SimpleNamespace(audio_latents=torch.randn(1, 2, 7, 4))
    vocoder.compile_audio(upstream_audio, stage, batch, AUDIO_ARGS)
    assert isinstance(stage.vocoder, vocoder.AudioChain)
    stage.vocoder.vocoder.threads.clear()
    return stage


def audio_request(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return SimpleNamespace(audio_latents=torch.randn(1, 2, 7, 4, generator=generator))


def test_the_audio_runs_beside_the_video(av_stage):
    batch = audio_request()
    expected = upstream_audio(av_stage, batch, AUDIO_ARGS)
    av_stage.vocoder.vocoder.threads.clear()
    with vocoder.audio_beside_video(av_stage, batch, AUDIO_ARGS):
        waveform = upstream_audio(av_stage, batch, AUDIO_ARGS)
    assert torch.equal(waveform, expected)
    assert av_stage.vocoder.vocoder.threads == ["rbln-audio_0"]


def test_other_latents_from_the_stage_run_in_place(av_stage):
    started, handed = audio_request(0), audio_request(1)
    with vocoder.audio_beside_video(av_stage, started, AUDIO_ARGS):
        waveform = upstream_audio(av_stage, handed, AUDIO_ARGS)
    av_stage.vocoder.vocoder.threads.clear()
    assert torch.equal(waveform, upstream_audio(av_stage, handed, AUDIO_ARGS))


@pytest.mark.parametrize(
    "case", ["shares-an-npu", "on-the-dit-npus", "first-request", "no-audio"]
)
def test_the_audio_stays_in_place(av_stage, case):
    batch = audio_request()
    if case == "shares-an-npu":
        placement.place(av_stage.vocoder, (3, 4, 5, 6))
    elif case == "on-the-dit-npus":
        placement._placements.pop(av_stage.vocoder)
    elif case == "first-request":
        av_stage.vocoder.__dict__.pop("_rbln_ran")
    else:
        batch = SimpleNamespace(audio_latents=None)
    with vocoder.audio_beside_video(av_stage, batch, AUDIO_ARGS):
        assert "_rbln_pending" not in av_stage.vocoder.__dict__
    assert av_stage.vocoder.vocoder.threads == []


def test_an_audio_error_surfaces_where_the_stage_takes_the_audio(av_stage, monkeypatch):
    def fail(spectrogram):
        raise RuntimeError("audio graph failed")

    # the chain calls the vocoder's class forward
    monkeypatch.setattr(RecordingVocoder, "forward", lambda self, x: fail(x))
    batch = audio_request()
    with (
        pytest.raises(RuntimeError, match="audio graph failed"),
        vocoder.audio_beside_video(av_stage, batch, AUDIO_ARGS),
    ):
        upstream_audio(av_stage, batch, AUDIO_ARGS)


def test_an_untaken_audio_error_is_raised_on_leaving(av_stage, monkeypatch):
    def fail(spectrogram):
        raise RuntimeError("audio graph failed")

    # the chain calls the vocoder's class forward
    monkeypatch.setattr(RecordingVocoder, "forward", lambda self, x: fail(x))
    with (
        pytest.raises(RuntimeError, match="audio graph failed"),
        vocoder.audio_beside_video(av_stage, audio_request(), AUDIO_ARGS),
    ):
        pass


def test_a_stage_error_stays_the_stages_and_the_audio_finishes(av_stage, monkeypatch):
    import threading

    finished = threading.Event()

    def slow(self, spectrogram):
        threading.Event().wait(0.2)
        finished.set()
        return Vocoder.forward(self, spectrogram)

    monkeypatch.setattr(RecordingVocoder, "forward", slow)
    with (
        pytest.raises(ValueError, match="video decode failed"),
        vocoder.audio_beside_video(av_stage, audio_request(), AUDIO_ARGS),
    ):
        raise ValueError("video decode failed")
    assert finished.is_set()
    assert "_rbln_pending" not in av_stage.vocoder.__dict__


def test_the_stage_decodes_with_the_audio_beside_it(av_stage):
    batch = audio_request()
    batch.use_diffusion_decoder = False
    seen = []

    def forward(stage, batch, server_args):
        seen.append("_rbln_pending" in stage.vocoder.__dict__)
        return upstream_audio(stage, batch, server_args)

    av_stage.vae.decoder.__dict__[decoding_av.COMPILED] = True
    waveform = decoding_av.compile_decoders(forward, av_stage, batch, AUDIO_ARGS)
    assert seen == [True]
    assert av_stage.vocoder.vocoder.threads == ["rbln-audio_0"]
    av_stage.vocoder.vocoder.threads.clear()
    assert torch.equal(waveform, upstream_audio(av_stage, batch, AUDIO_ARGS))

"""Tests for stages/prosody.py — acoustic prosody extraction (Spec 7, Phase 7.1/7.2).

Real librosa + parselmouth run on small synthetic arrays (a short generated
sine + vibrato + jitter/shimmer "voice", ~3s) — keeps each test fast (<2s)
while exercising the actual library calls per Spec 7.
"""

import numpy as np
import pytest
import soundfile as sf

from shumary.models import AlignedSegment
from shumary.stages import prosody


# ── Synthetic fixture ────────────────────────────────────────────────────────────

def _make_voice_wav(path, duration=3.0, f0=120.0, sr=16000):
    """A short speech-like signal: f0 fundamental + vibrato + harmonics +
    jitter (freq) + shimmer (amplitude) perturbations."""
    n = int(sr * duration)
    t = np.arange(n) / sr

    vibrato = 6.0 * np.sin(2 * np.pi * 5.0 * t)
    rng = np.random.default_rng(0)
    jitter_noise = rng.normal(0, 1.0, n)
    f0_inst = f0 + vibrato + jitter_noise

    phase = 2 * np.pi * np.cumsum(f0_inst) / sr
    y = (
        1.0 * np.sin(phase)
        + 0.5 * np.sin(2 * phase)
        + 0.25 * np.sin(3 * phase)
    )

    shimmer_noise = 1.0 + rng.normal(0, 0.03, n)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t) ** 2
    y = y * shimmer_noise * envelope
    y = y / np.max(np.abs(y)) * 0.7

    sf.write(str(path), y.astype(np.float32), sr, subtype="PCM_16")
    return path


def _make_silence_wav(path, duration=2.0, sr=16000):
    n = int(sr * duration)
    sf.write(str(path), np.zeros(n, dtype=np.float32), sr, subtype="PCM_16")
    return path


# ── available() ──────────────────────────────────────────────────────────────────

def test_available_true_when_libs_installed():
    assert prosody.available() is True


def test_run_returns_none_when_librosa_missing(tmp_path, monkeypatch):
    wav = _make_voice_wav(tmp_path / "audio.wav")
    segments = [AlignedSegment(0.0, 3.0, "hello", -0.1, 0.05, "SPEAKER_00")]

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "librosa":
            raise ImportError("no librosa")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert prosody.run(wav, segments) is None


def test_run_returns_none_when_parselmouth_missing(tmp_path, monkeypatch):
    wav = _make_voice_wav(tmp_path / "audio.wav")
    segments = [AlignedSegment(0.0, 3.0, "hello", -0.1, 0.05, "SPEAKER_00")]

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "parselmouth":
            raise ImportError("no parselmouth")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert prosody.run(wav, segments) is None


# ── Per-segment extraction ────────────────────────────────────────────────────────

def test_run_extracts_f0_energy_and_voice_quality_for_voiced_segment(tmp_path):
    wav = _make_voice_wav(tmp_path / "audio.wav")
    segments = [AlignedSegment(0.0, 3.0, "hello there", -0.1, 0.05, "SPEAKER_00")]

    result = prosody.run(wav, segments)
    assert result is not None
    assert len(result.segments) == 1

    seg = result.segments[0]
    assert seg.f0_median_hz is not None
    assert 65 <= seg.f0_median_hz <= 400  # speech band
    assert seg.f0_range_st is not None
    assert seg.energy_mean_db is not None
    assert seg.energy_range_db is not None
    # jitter/shimmer should be small positive floats for a near-periodic signal
    assert seg.jitter_local is not None
    assert seg.shimmer_local is not None
    assert 0 < seg.jitter_local < 1.0
    assert 0 < seg.shimmer_local < 1.0


def test_segments_shorter_than_min_duration_are_skipped(tmp_path):
    wav = _make_voice_wav(tmp_path / "audio.wav")
    segments = [
        AlignedSegment(0.0, 0.5, "hi", -0.1, 0.05, "SPEAKER_00"),  # < 1.0s, skipped
        AlignedSegment(0.5, 3.0, "there", -0.1, 0.05, "SPEAKER_00"),
    ]

    result = prosody.run(wav, segments)
    assert result is not None
    assert len(result.segments) == 1
    assert result.segments[0].start == 0.5


def test_silent_segment_yields_none_f0_fields(tmp_path):
    wav = _make_silence_wav(tmp_path / "silence.wav")
    segments = [AlignedSegment(0.0, 2.0, "...", -0.1, 0.9, "SPEAKER_00")]

    result = prosody.run(wav, segments)
    assert result is not None
    seg = result.segments[0]
    assert seg.f0_median_hz is None
    assert seg.f0_range_st is None
    # speakers with no voiced-valid segments are omitted from aggregation
    assert result.speakers == ()


# ── Per-speaker aggregation (Spec 7.2) ────────────────────────────────────────────

def test_aggregation_single_speaker_label_from_speaker_00(tmp_path):
    wav = _make_voice_wav(tmp_path / "audio.wav")
    segments = [
        AlignedSegment(0.0, 1.5, "a", -0.1, 0.05, "SPEAKER_00"),
        AlignedSegment(1.5, 3.0, "b", -0.1, 0.05, "SPEAKER_00"),
    ]

    result = prosody.run(wav, segments)
    assert result is not None
    assert len(result.speakers) == 1
    sp = result.speakers[0]
    assert sp.speaker == "speaker"
    assert sp.f0_baseline_hz > 0
    assert sp.f0_variability_st >= 0
    assert isinstance(sp.f0_trend_by_third_st, tuple)
    assert len(sp.f0_trend_by_third_st) == 3


def test_aggregation_none_speaker_label_is_speaker(tmp_path):
    wav = _make_voice_wav(tmp_path / "audio.wav")
    segments = [AlignedSegment(0.0, 3.0, "solo memo", -0.1, 0.05, None)]

    result = prosody.run(wav, segments)
    assert result is not None
    assert len(result.speakers) == 1
    assert result.speakers[0].speaker == "speaker"


def test_aggregation_two_speakers_labelled_separately(tmp_path):
    wav = _make_voice_wav(tmp_path / "audio.wav")
    segments = [
        AlignedSegment(0.0, 1.5, "a", -0.1, 0.05, "SPEAKER_00"),
        AlignedSegment(1.5, 3.0, "b", -0.1, 0.05, "SPEAKER_01"),
    ]

    result = prosody.run(wav, segments)
    assert result is not None
    labels = {sp.speaker for sp in result.speakers}
    assert labels == {"SPEAKER_00", "SPEAKER_01"}


def test_speakers_with_no_valid_segments_are_omitted(tmp_path):
    voice = _make_voice_wav(tmp_path / "voice.wav")
    silence = _make_silence_wav(tmp_path / "silence.wav")

    # Build a combined wav: voiced for SPEAKER_00, silent for SPEAKER_01
    y_voice, sr = sf.read(str(voice))
    y_silence, _ = sf.read(str(silence))
    combined = np.concatenate([y_voice[: sr * 3], y_silence[: sr * 2]])
    combined_path = tmp_path / "combined.wav"
    sf.write(str(combined_path), combined.astype(np.float32), sr, subtype="PCM_16")

    segments = [
        AlignedSegment(0.0, 3.0, "voiced", -0.1, 0.05, "SPEAKER_00"),
        AlignedSegment(3.0, 5.0, "silent", -0.1, 0.9, "SPEAKER_01"),
    ]

    result = prosody.run(combined_path, segments)
    assert result is not None
    labels = {sp.speaker for sp in result.speakers}
    assert labels == {"SPEAKER_00"}


# ── summarize() ──────────────────────────────────────────────────────────────────

def test_summarize_one_line_per_speaker(tmp_path):
    wav = _make_voice_wav(tmp_path / "audio.wav")
    segments = [AlignedSegment(0.0, 3.0, "hello", -0.1, 0.05, "SPEAKER_00")]

    result = prosody.run(wav, segments)
    summary = prosody.summarize(result)
    assert "speaker" in summary
    assert "Hz" in summary


def test_summarize_no_voiced_segments():
    from shumary.models import ProsodyResult
    result = ProsodyResult(segments=(), speakers=())
    assert "unavailable" in prosody.summarize(result)


# ── YAML serialization regression (pilot finding 1) ──────────────────────────────

def test_aggregation_output_is_yaml_safe_even_with_numpy_inputs():
    """The pilot crashed on np.float64 leaking into yaml.safe_dump. Aggregation
    must emit pure-Python floats even when segment fields are numpy scalars."""
    import numpy as np
    import yaml
    from shumary.models import ProsodySegment
    from shumary.stages.prosody import _aggregate_per_speaker

    def seg(start, speaker, f0):
        return ProsodySegment(
            start=start, end=start + 2.0, speaker=speaker,
            f0_median_hz=np.float64(f0), f0_range_st=np.float64(2.0),
            energy_mean_db=np.float64(-30.0), energy_range_db=np.float64(10.0),
            jitter_local=np.float64(0.01), shimmer_local=np.float64(0.05),
        )

    speakers = _aggregate_per_speaker([
        seg(0.0, "SPEAKER_00", 110.0), seg(2.0, "SPEAKER_00", 130.0),
        seg(4.0, "SPEAKER_01", 200.0), seg(6.0, "SPEAKER_01", 210.0),
    ])

    from shumary.stages.render import _speaker_prosody_block
    blocks = [_speaker_prosody_block(sp) for sp in speakers]
    dumped = yaml.safe_dump({"prosody": blocks}, allow_unicode=True)  # must not raise
    assert "f0_baseline_hz" in dumped

    for block in blocks:
        for key, value in block.items():
            leaves = value if isinstance(value, list) else [value]
            for leaf in leaves:
                assert not isinstance(leaf, np.generic), f"{key} leaked numpy type"

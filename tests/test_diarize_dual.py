"""Tests for the dual-source (mic + system audio) diarization path."""

import wave

import numpy as np
import pytest

from shumary.models import DiarizationResult, Turn
from shumary.stages import diarize

RATE = 16_000


def write_wav(path, spans, *, duration=10.0, amplitude=0.5, rate=RATE, seed=0):
    """Write a mono 16-bit WAV with noise bursts over `spans` (seconds)."""
    rng = np.random.default_rng(seed)
    samples = np.zeros(int(duration * rate), dtype=np.float32)
    for start, end in spans:
        a, b = int(start * rate), int(end * rate)
        samples[a:b] = rng.normal(0, amplitude, b - a)
    pcm = np.clip(samples, -1, 1) * 32767
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.astype(np.int16).tobytes())
    return path


def spans_of(turns, speaker=None):
    return [
        (round(t.start, 1), round(t.end, 1))
        for t in turns
        if speaker is None or t.speaker == speaker
    ]


# ── mask smoothing ───────────────────────────────────────────────────────────

def test_spans_from_mask_bridges_short_gaps():
    # Two runs separated by 0.1s — below DUAL_MIN_GAP_SEC, so they merge.
    mask = np.zeros(200, dtype=bool)
    mask[0:50] = True
    mask[60:150] = True

    spans = diarize._spans_from_mask(mask, hop_sec=0.01, frame_sec=0.03)
    assert len(spans) == 1


def test_spans_from_mask_drops_runs_below_minimum_speech():
    mask = np.zeros(200, dtype=bool)
    mask[0:5] = True  # 0.05s — shorter than DUAL_MIN_SPEECH_SEC

    assert diarize._spans_from_mask(mask, hop_sec=0.01, frame_sec=0.03) == []


def test_spans_from_mask_returns_nothing_for_silence():
    assert diarize._spans_from_mask(np.zeros(100, dtype=bool), hop_sec=0.01, frame_sec=0.03) == []


def test_spans_from_mask_handles_speech_running_to_the_end():
    mask = np.zeros(100, dtype=bool)
    mask[50:] = True

    spans = diarize._spans_from_mask(mask, hop_sec=0.01, frame_sec=0.03)
    assert len(spans) == 1
    assert spans[0][1] == pytest.approx(1.03, abs=0.01)


# ── mic gating ───────────────────────────────────────────────────────────────

def test_mic_turns_finds_speech_on_the_left_channel(tmp_path):
    left = write_wav(tmp_path / "left.wav", [(1.0, 3.0), (6.0, 8.0)])
    right = write_wav(tmp_path / "right.wav", [])

    turns = diarize._mic_turns(left, right, label="You")

    assert [t.speaker for t in turns] == ["You", "You"]
    starts = [round(t.start) for t in turns]
    assert starts == [1, 6]


def test_mic_turns_rejects_speaker_bleed_from_the_system_channel(tmp_path):
    """Audio played out loud and re-captured by the mic must not become a turn.

    The bleed is an attenuated copy of what the system channel already holds,
    so the echo margin rules it out.
    """
    span = [(2.0, 5.0)]
    right = write_wav(tmp_path / "right.wav", span, amplitude=0.5, seed=1)
    # Same burst on the mic, 20 dB down — plausible speaker-to-mic attenuation.
    left = write_wav(tmp_path / "left.wav", span, amplitude=0.05, seed=1)

    assert diarize._mic_turns(left, right, label="You") == []


def test_mic_turns_keeps_speech_that_overlaps_system_audio(tmp_path):
    """Talking over the other side is real speech, not bleed."""
    right = write_wav(tmp_path / "right.wav", [(0.0, 10.0)], amplitude=0.05, seed=2)
    left = write_wav(tmp_path / "left.wav", [(3.0, 6.0)], amplitude=0.6, seed=3)

    turns = diarize._mic_turns(left, right, label="You")
    assert turns and turns[0].speaker == "You"


def test_mic_turns_returns_nothing_for_a_silent_mic(tmp_path):
    left = write_wav(tmp_path / "left.wav", [])
    right = write_wav(tmp_path / "right.wav", [(1.0, 4.0)])

    assert diarize._mic_turns(left, right, label="You") == []


def test_mic_turns_survives_unreadable_channels(tmp_path, capsys):
    bad = tmp_path / "left.wav"
    bad.write_bytes(b"not a wav")
    right = write_wav(tmp_path / "right.wav", [])

    assert diarize._mic_turns(bad, right, label="You") == []
    assert "could not read" in capsys.readouterr().out


# ── signal detection ─────────────────────────────────────────────────────────

def test_has_signal_is_false_for_a_silent_channel(tmp_path):
    assert diarize._has_signal(write_wav(tmp_path / "s.wav", [])) is False


def test_has_signal_is_true_when_something_played(tmp_path):
    assert diarize._has_signal(write_wav(tmp_path / "s.wav", [(1.0, 2.0)])) is True


# ── run_dual ─────────────────────────────────────────────────────────────────

def test_run_dual_merges_mic_and_remote_turns_in_time_order(tmp_path, monkeypatch):
    left = write_wav(tmp_path / "left.wav", [(0.0, 2.0)])
    right = write_wav(tmp_path / "right.wav", [(4.0, 6.0)])

    monkeypatch.setattr(
        diarize, "run",
        lambda path, progress=None: DiarizationResult(
            turns=(Turn(start=4.0, end=6.0, speaker="SPEAKER_00"),), backend="pyannote"
        ),
    )

    result = diarize.run_dual(left, right, mic_label="You")

    assert result.backend == "dual-channel+pyannote"
    assert [t.speaker for t in result.turns] == ["You", "SPEAKER_00"]
    assert list(result.turns) == sorted(result.turns, key=lambda t: t.start)


def test_run_dual_skips_diarization_when_nothing_played(tmp_path, monkeypatch):
    left = write_wav(tmp_path / "left.wav", [(1.0, 3.0)])
    right = write_wav(tmp_path / "right.wav", [])

    def _explode(path):  # pragma: no cover - must not run
        raise AssertionError("a silent system channel must not be diarized")

    monkeypatch.setattr(diarize, "run", _explode)

    result = diarize.run_dual(left, right, mic_label="You")
    assert result.backend == "dual-channel"
    assert {t.speaker for t in result.turns} == {"You"}


def test_run_dual_returns_none_when_both_channels_are_empty(tmp_path, monkeypatch):
    left = write_wav(tmp_path / "left.wav", [])
    right = write_wav(tmp_path / "right.wav", [])
    monkeypatch.setattr(diarize, "run", lambda path, progress=None: None)

    assert diarize.run_dual(left, right, mic_label="You") is None


def test_run_dual_keeps_remote_turns_when_the_mic_is_silent(tmp_path, monkeypatch):
    left = write_wav(tmp_path / "left.wav", [])
    right = write_wav(tmp_path / "right.wav", [(1.0, 5.0)])

    monkeypatch.setattr(
        diarize, "run",
        lambda path, progress=None: DiarizationResult(
            turns=(Turn(start=1.0, end=5.0, speaker="SPEAKER_00"),), backend="resemblyzer"
        ),
    )

    result = diarize.run_dual(left, right, mic_label="You")
    assert [t.speaker for t in result.turns] == ["SPEAKER_00"]
    assert result.backend == "dual-channel+resemblyzer"


def test_run_dual_labels_the_mic_with_the_requested_name(tmp_path, monkeypatch):
    left = write_wav(tmp_path / "left.wav", [(1.0, 3.0)])
    right = write_wav(tmp_path / "right.wav", [])
    monkeypatch.setattr(diarize, "run", lambda path, progress=None: None)

    result = diarize.run_dual(left, right, mic_label="Denis")
    assert {t.speaker for t in result.turns} == {"Denis"}


# ── onset gate: sustained speech vs transients ────────────────────────────────

def test_mic_gate_admits_sustained_speech(tmp_path):
    left = write_wav(tmp_path / "left.wav", [(1.0, 3.0)], amplitude=0.5)
    right = write_wav(tmp_path / "right.wav", [])

    turns = diarize._mic_turns(left, right, label="Me")
    assert spans_of(turns) == [(1.0, 3.0)]


def test_mic_gate_rejects_transient_clicks(tmp_path):
    """A keypress clears the release threshold but never holds the onset
    threshold for DUAL_MIN_ONSET_SEC, so it is not a turn. Two 20 ms spikes
    inside one smoothing window would otherwise merge into a ~0.3 s 'turn'."""
    left = write_wav(
        tmp_path / "left.wav",
        [(1.00, 1.02), (1.20, 1.22), (1.40, 1.42)],
        amplitude=0.5,
    )
    right = write_wav(tmp_path / "right.wav", [])

    assert diarize._mic_turns(left, right, label="Me") == []


def test_mic_gate_keeps_short_backchannel(tmp_path):
    """A 300 ms voiced utterance is short but sustained — it must survive."""
    left = write_wav(tmp_path / "left.wav", [(2.0, 2.3)], amplitude=0.5)
    right = write_wav(tmp_path / "right.wav", [])

    turns = diarize._mic_turns(left, right, label="Me")
    assert len(turns) == 1
    # Spans carry one frame of padding at the end by construction (see
    # _spans_from_mask), so the turn is the utterance plus DUAL_FRAME_SEC.
    assert turns[0].end - turns[0].start == pytest.approx(0.3, abs=0.08)


def test_mic_gate_still_rejects_speaker_bleed(tmp_path):
    """The echo margin must keep working: an attenuated copy of the system
    channel never beats its own source, even though it is sustained."""
    left = write_wav(tmp_path / "left.wav", [(1.0, 4.0)], amplitude=0.02)
    right = write_wav(tmp_path / "right.wav", [(1.0, 4.0)], amplitude=0.5)

    assert diarize._mic_turns(left, right, label="Me") == []

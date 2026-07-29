"""Tests for stages/audio.py — validation raises ShumaryError, not sys.exit (A9)."""

import subprocess
from pathlib import Path

import pytest

from shumary.errors import ShumaryError
from shumary.stages import audio


def test_validate_missing_file_raises_shumary_error(tmp_path):
    missing = tmp_path / "nope.m4a"
    with pytest.raises(ShumaryError, match="not found"):
        audio.validate(missing)


def test_validate_unsupported_format_raises_shumary_error(tmp_path):
    bad = tmp_path / "memo.txt"
    bad.write_text("not audio")
    with pytest.raises(ShumaryError, match="Unsupported format"):
        audio.validate(bad)


def test_validate_ok_for_supported_file(tmp_path):
    good = tmp_path / "memo.m4a"
    good.write_text("fake-audio")
    audio.validate(good)  # should not raise


# ── decode_to_wav (3.1) ──────────────────────────────────────────────────────────

def test_decode_to_wav_missing_ffmpeg_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("shumary.paths.cache_dir_for", lambda slug: tmp_path / slug)
    monkeypatch.setattr(audio, "cache_dir_for", lambda slug: tmp_path / slug)
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)

    src = tmp_path / "memo.m4a"
    src.write_text("fake-audio")

    with pytest.raises(ShumaryError, match="ffmpeg"):
        audio.decode_to_wav(src, "memo")


def test_decode_to_wav_invokes_ffmpeg_with_expected_args(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "cache_dir_for", lambda slug: tmp_path / slug)
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    src = tmp_path / "memo.m4a"
    src.write_text("fake-audio")

    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        # Simulate ffmpeg writing the output file.
        out = Path(cmd[-1])
        out.write_bytes(b"RIFF....WAVEfmt ")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", _fake_run)

    out = audio.decode_to_wav(src, "memo")

    assert out == tmp_path / "memo" / "audio.wav"
    assert out.exists()
    cmd = captured["cmd"]
    assert cmd[0] == "ffmpeg"
    assert "-i" in cmd and str(src) in cmd
    assert "-ar" in cmd and "16000" in cmd
    assert "-ac" in cmd and "1" in cmd
    assert "-c:a" in cmd and "pcm_s16le" in cmd


def test_decode_to_wav_reuses_existing_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "cache_dir_for", lambda slug: tmp_path / slug)
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    cache_dir = tmp_path / "memo"
    cache_dir.mkdir()
    existing = cache_dir / "audio.wav"
    existing.write_bytes(b"already-decoded")

    def _fake_run(cmd, **kwargs):
        raise AssertionError("ffmpeg should not be invoked when the WAV already exists")

    monkeypatch.setattr(audio.subprocess, "run", _fake_run)

    src = tmp_path / "memo.m4a"
    src.write_text("fake-audio")

    out = audio.decode_to_wav(src, "memo")

    assert out == existing
    assert out.read_bytes() == b"already-decoded"


def test_decode_to_wav_ffmpeg_failure_raises_shumary_error(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "cache_dir_for", lambda slug: tmp_path / slug)
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    src = tmp_path / "memo.m4a"
    src.write_text("fake-audio")

    def _fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="invalid data")

    monkeypatch.setattr(audio.subprocess, "run", _fake_run)

    with pytest.raises(ShumaryError, match="ffmpeg failed"):
        audio.decode_to_wav(src, "memo")


# ── split_channels / channel_count (dual-source recordings) ─────────────────────

def test_channel_count_reads_stream_channels(tmp_path, monkeypatch):
    monkeypatch.setattr(
        audio.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="2\n", stderr=""),
    )
    assert audio.channel_count(tmp_path / "rec.wav") == 2


def test_channel_count_returns_none_when_ffprobe_says_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        audio.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom"),
    )
    assert audio.channel_count(tmp_path / "rec.wav") is None


def test_split_channels_missing_ffmpeg_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "cache_dir_for", lambda slug: tmp_path / slug)
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)

    src = tmp_path / "rec.wav"
    src.write_text("fake-audio")

    with pytest.raises(ShumaryError, match="ffmpeg"):
        audio.split_channels(src, "rec")


def test_split_channels_writes_left_and_right_mono_wavs(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "cache_dir_for", lambda slug: tmp_path / slug)
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    src = tmp_path / "rec.wav"
    src.write_text("fake-audio")

    captured = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        for arg in cmd:
            if str(arg).endswith(".tmp.wav"):
                Path(arg).parent.mkdir(parents=True, exist_ok=True)
                Path(arg).write_bytes(b"RIFF....WAVEfmt ")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", _fake_run)

    left, right = audio.split_channels(src, "rec")

    assert left.name == "left.wav" and right.name == "right.wav"
    assert left.exists() and right.exists()
    assert "channelsplit=channel_layout=stereo[l][r]" in " ".join(captured["cmd"])
    # Both channels are downmixed to 16 kHz mono, like the main decode path.
    assert captured["cmd"].count("16000") == 2


def test_split_channels_reuses_existing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "cache_dir_for", lambda slug: tmp_path / slug)

    cache = tmp_path / "rec"
    cache.mkdir()
    (cache / "left.wav").write_bytes(b"cached-left")
    (cache / "right.wav").write_bytes(b"cached-right")

    def _explode(*args, **kwargs):
        raise AssertionError("ffmpeg should not run when both channels are cached")

    monkeypatch.setattr(audio.subprocess, "run", _explode)

    src = tmp_path / "rec.wav"
    src.write_text("fake-audio")

    left, right = audio.split_channels(src, "rec")
    assert left.read_bytes() == b"cached-left"


def test_split_channels_ffmpeg_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "cache_dir_for", lambda slug: tmp_path / slug)
    monkeypatch.setattr(audio.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    src = tmp_path / "rec.wav"
    src.write_text("fake-audio")

    def _fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="bad layout")

    monkeypatch.setattr(audio.subprocess, "run", _fake_run)

    with pytest.raises(ShumaryError, match="split channels"):
        audio.split_channels(src, "rec")

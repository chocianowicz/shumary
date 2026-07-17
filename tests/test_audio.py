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

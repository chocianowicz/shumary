"""Tests for utils.py — time/duration formatting (A6) and recording date (A7)."""

import os
import time
import wave

import pytest

from shumary.utils import fmt_time, fmt_duration, resolve_recording_date, wav_duration_seconds


def test_fmt_time_under_a_minute():
    assert fmt_time(7) == "00:07"


def test_fmt_time_under_an_hour():
    assert fmt_time(125) == "02:05"


def test_fmt_time_at_exactly_one_hour():
    assert fmt_time(3600) == "1:00:00"


def test_fmt_time_past_an_hour():
    # 226:12 (the A6 bug example) is 3h 46m 12s == 13572 seconds
    assert fmt_time(13572) == "3:46:12"


def test_fmt_duration_none():
    assert fmt_duration(None) == "unknown"


def test_fmt_duration_under_a_minute():
    assert fmt_duration(7) == "0m 07s"


def test_fmt_duration_under_an_hour():
    assert fmt_duration(8 * 60 + 12) == "8m 12s"


def test_fmt_duration_at_exactly_one_hour():
    assert fmt_duration(3600) == "1h 0m 00s"


def test_fmt_duration_past_an_hour():
    # 226m 12s == 3h 46m 12s
    assert fmt_duration(226 * 60 + 12) == "3h 46m 12s"


# ── resolve_recording_date (A7) ──────────────────────────────────────────────────

def test_date_override_wins(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_text("x")
    date_str, source = resolve_recording_date(audio, override="2025-01-02")
    assert date_str == "2025-01-02"
    assert source == "--date override"


def test_date_from_filename(tmp_path):
    audio = tmp_path / "2025-11-12 session.m4a"
    audio.write_text("x")
    date_str, source = resolve_recording_date(audio)
    assert date_str == "2025-11-12"
    assert source == "filename"


def test_invalid_date_in_filename_falls_back_to_mtime(tmp_path):
    # 2025-13-40 is not a real date — should be ignored, falling back to mtime.
    audio = tmp_path / "2025-13-40 session.m4a"
    audio.write_text("x")
    date_str, source = resolve_recording_date(audio)
    assert source == "file mtime"


def test_date_from_mtime(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_text("x")
    # Set mtime to a known date.
    target = time.mktime((2024, 6, 15, 12, 0, 0, 0, 0, -1))
    os.utime(audio, (target, target))

    date_str, source = resolve_recording_date(audio)
    assert date_str == "2024-06-15"
    assert source == "file mtime"


def test_date_falls_back_to_today_for_missing_file(tmp_path):
    missing = tmp_path / "ghost.m4a"
    date_str, source = resolve_recording_date(missing)
    assert source == "today"


# ── wav_duration_seconds (3.1) ───────────────────────────────────────────────────

def test_wav_duration_seconds_reads_header(tmp_path):
    wav_path = tmp_path / "audio.wav"
    sr = 16000
    n_frames = sr * 2  # 2 seconds
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * n_frames)

    assert wav_duration_seconds(wav_path) == pytest.approx(2.0)


def test_wav_duration_seconds_missing_file_returns_none(tmp_path):
    missing = tmp_path / "nope.wav"
    assert wav_duration_seconds(missing) is None


def test_wav_duration_seconds_invalid_file_returns_none(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_text("not a wav file")
    assert wav_duration_seconds(bad) is None

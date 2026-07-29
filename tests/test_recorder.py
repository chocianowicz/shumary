"""Tests for recorder.py — sidecar handling and helper build plumbing.

The Swift helper itself is not exercised here: it needs real audio hardware and
TCC permission, so these tests cover everything around it.
"""

import json
import subprocess
from pathlib import Path

import pytest

from shumary import recorder
from shumary.errors import ShumaryError


# ── Sidecar ──────────────────────────────────────────────────────────────────

def test_sidecar_path_keeps_audio_extension(tmp_path):
    wav = tmp_path / "meeting.wav"
    assert recorder.sidecar_path(wav).name == "meeting.wav.shumary.json"


def test_write_sidecar_marks_recording_as_dual_source(tmp_path):
    wav = tmp_path / "meeting.wav"
    wav.write_bytes(b"RIFF")

    path = recorder.write_sidecar(wav)
    payload = json.loads(path.read_text())

    assert payload["dual_source"] is True
    assert payload["channels"] == {"left": "mic", "right": "system"}
    assert payload["mic_label"] == recorder.MIC_LABEL


def test_write_sidecar_honours_custom_mic_label(tmp_path):
    wav = tmp_path / "meeting.wav"
    recorder.write_sidecar(wav, mic_label="Denis")
    assert recorder.read_sidecar(wav)["mic_label"] == "Denis"


def test_read_sidecar_returns_none_when_absent(tmp_path):
    assert recorder.read_sidecar(tmp_path / "meeting.wav") is None


def test_read_sidecar_returns_none_for_corrupt_json(tmp_path):
    wav = tmp_path / "meeting.wav"
    recorder.sidecar_path(wav).write_text("{not json")
    assert recorder.read_sidecar(wav) is None


def test_dual_source_info_ignores_sidecar_that_opts_out(tmp_path):
    wav = tmp_path / "meeting.wav"
    recorder.sidecar_path(wav).write_text(json.dumps({"dual_source": False}))
    assert recorder.dual_source_info(wav) is None


def test_dual_source_info_returns_payload_for_recording(tmp_path):
    wav = tmp_path / "meeting.wav"
    recorder.write_sidecar(wav)
    assert recorder.dual_source_info(wav)["mic_label"] == recorder.MIC_LABEL


# ── Output naming ────────────────────────────────────────────────────────────

def test_default_output_path_is_timestamped_wav_in_directory(tmp_path):
    out = recorder.default_output_path(tmp_path)
    assert out.parent == tmp_path
    assert out.name.startswith("recording-")
    assert out.suffix == ".wav"


# ── Helper build ─────────────────────────────────────────────────────────────

def test_helper_binary_path_changes_with_source(tmp_path, monkeypatch):
    monkeypatch.setattr("shumary.recorder.CACHE", tmp_path)

    source = tmp_path / "helper.swift"
    source.write_text("// v1")
    monkeypatch.setattr(recorder, "helper_sources", lambda: [source])
    first = recorder._helper_binary_path()

    source.write_text("// v2")
    assert recorder._helper_binary_path() != first


def test_build_helper_without_swiftc_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("shumary.recorder.CACHE", tmp_path)
    monkeypatch.setattr(recorder.shutil, "which", lambda name: None)

    with pytest.raises(ShumaryError, match="xcode-select"):
        recorder.build_helper()


def test_build_helper_reuses_existing_binary(tmp_path, monkeypatch):
    monkeypatch.setattr("shumary.recorder.CACHE", tmp_path)
    binary = recorder._helper_binary_path()
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"already-built")

    def _explode(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("swiftc should not be invoked for a cached binary")

    monkeypatch.setattr(recorder.subprocess, "run", _explode)
    assert recorder.build_helper() == binary


def test_build_helper_surfaces_compiler_errors(tmp_path, monkeypatch):
    monkeypatch.setattr("shumary.recorder.CACHE", tmp_path)
    monkeypatch.setattr(recorder.shutil, "which", lambda name: "/usr/bin/swiftc")
    monkeypatch.setattr(
        recorder.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom"),
    )

    with pytest.raises(ShumaryError, match="boom"):
        recorder.build_helper()


# ── record() ─────────────────────────────────────────────────────────────────

class _FakeProcess:
    def __init__(self, on_wait):
        self._on_wait = on_wait
        self.signals: list[int] = []

    def wait(self, timeout=None):
        return self._on_wait()

    def send_signal(self, sig):
        self.signals.append(sig)

    def kill(self):  # pragma: no cover - only for the timeout path
        pass


def test_record_writes_sidecar_and_returns_path(tmp_path, monkeypatch):
    out = tmp_path / "meeting.wav"

    def _fake_popen(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"\0" * 4096)
        return _FakeProcess(lambda: 0)

    monkeypatch.setattr(recorder, "build_helper", lambda: Path("/fake/shumary-record"))
    monkeypatch.setattr(recorder.subprocess, "Popen", _fake_popen)

    assert recorder.record(out) == out
    assert recorder.dual_source_info(out) is not None


def test_record_rejects_header_only_output(tmp_path, monkeypatch):
    out = tmp_path / "meeting.wav"

    def _fake_popen(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"\0" * 44)  # WAV header, no frames
        return _FakeProcess(lambda: 0)

    monkeypatch.setattr(recorder, "build_helper", lambda: Path("/fake/shumary-record"))
    monkeypatch.setattr(recorder.subprocess, "Popen", _fake_popen)

    with pytest.raises(ShumaryError, match="permission"):
        recorder.record(out)


def test_record_forwards_interrupt_to_the_helper(tmp_path, monkeypatch):
    import signal as signal_mod

    out = tmp_path / "meeting.wav"
    process = None

    def _fake_popen(cmd, **kwargs):
        nonlocal process
        Path(cmd[-1]).write_bytes(b"\0" * 4096)
        calls = {"n": 0}

        def _wait():
            calls["n"] += 1
            if calls["n"] == 1:
                raise KeyboardInterrupt
            return 0

        process = _FakeProcess(_wait)
        return process

    monkeypatch.setattr(recorder, "build_helper", lambda: Path("/fake/shumary-record"))
    monkeypatch.setattr(recorder.subprocess, "Popen", _fake_popen)

    recorder.record(out)
    assert process.signals == [signal_mod.SIGINT]

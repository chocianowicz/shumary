"""CLI smoke tests using Typer's test runner."""

from typer.testing import CliRunner
from shumary.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "transcribe" in result.output
    assert "doctor" in result.output


def test_transcribe_help():
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.output
    assert "--no-diarize" in result.output
    assert "--model" in result.output
    assert "--output" in result.output


def test_cache_list_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("shumary.paths.CACHE", tmp_path / ".cache")
    result = runner.invoke(app, ["cache", "list"])
    assert result.exit_code == 0


def test_cache_list_shows_fingerprint_column(tmp_path, monkeypatch):
    cache_dir = tmp_path / ".cache"
    monkeypatch.setattr("shumary.paths.CACHE", cache_dir)
    monkeypatch.setattr("shumary.cache.cache_dir_for", lambda slug: cache_dir / slug)

    from shumary import cache as cache_mod

    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"audio-bytes")
    cache_mod.ensure_manifest("memo", audio, "turbo", True)
    cache_mod.write("memo", "whisper", {})

    # Legacy slug: stage file but no manifest.
    cache_mod.write("legacy-slug", "whisper", {})

    result = runner.invoke(app, ["cache", "list"])

    assert result.exit_code == 0
    assert "turbo --fast" in result.output
    assert "v1/legacy" in result.output


def test_cache_clear_stage_clears_descendants(tmp_path, monkeypatch):
    cache_dir = tmp_path / ".cache"
    monkeypatch.setattr("shumary.paths.CACHE", cache_dir)
    monkeypatch.setattr("shumary.cache.cache_dir_for", lambda slug: cache_dir / slug)

    from shumary import cache as cache_mod
    cache_mod.write("memo", "whisper", {})
    cache_mod.write("memo", "align", {})
    cache_mod.write("memo", "mood", {})
    cache_mod.mark_done("memo")

    result = runner.invoke(app, ["cache", "clear", "memo", "--stage", "whisper"])

    assert result.exit_code == 0
    assert not cache_mod.is_cached("memo", "whisper")
    assert not cache_mod.is_cached("memo", "align")
    assert not cache_mod.is_cached("memo", "mood")
    assert not cache_mod.is_done("memo")


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ffprobe" in result.output


def test_transcribe_missing_file_exits_nonzero(tmp_path):
    missing = tmp_path / "does-not-exist.m4a"
    result = runner.invoke(app, ["transcribe", str(missing)])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_transcribe_invalid_date_exits_nonzero(tmp_path):
    missing = tmp_path / "does-not-exist.m4a"
    result = runner.invoke(app, ["transcribe", str(missing), "--date", "not-a-date"])
    assert result.exit_code == 1
    assert "Invalid --date" in result.output


def test_transcribe_help_lists_date_option():
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--date" in result.output


# ── record ─────────────────────────────────────────────────────────────────────

def test_record_help():
    result = runner.invoke(app, ["record", "--help"])
    assert result.exit_code == 0
    assert "system audio" in result.output
    assert "--transcribe" in result.output


def test_transcribe_help_lists_dual_channel_flag():
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--dual-channel" in result.output
    assert "--no-dual-channel" in result.output


def test_record_writes_to_the_given_path(tmp_path, monkeypatch):
    from shumary import recorder

    target = tmp_path / "call.wav"
    captured = {}

    def _fake_record(out):
        captured["out"] = out
        out.write_bytes(b"pcm")
        return out

    monkeypatch.setattr(recorder, "record", _fake_record)

    result = runner.invoke(app, ["record", str(target)])

    assert result.exit_code == 0
    assert captured["out"] == target


def test_record_into_a_directory_gets_a_timestamped_name(tmp_path, monkeypatch):
    from shumary import recorder

    captured = {}
    monkeypatch.setattr(recorder, "record", lambda out: captured.setdefault("out", out))

    result = runner.invoke(app, ["record", str(tmp_path)])

    assert result.exit_code == 0
    assert captured["out"].parent == tmp_path
    assert captured["out"].name.startswith("recording-")


def test_record_adds_a_wav_extension_when_missing(tmp_path, monkeypatch):
    from shumary import recorder

    captured = {}
    monkeypatch.setattr(recorder, "record", lambda out: captured.setdefault("out", out))

    runner.invoke(app, ["record", str(tmp_path / "call")])

    assert captured["out"].suffix == ".wav"


def test_record_reports_failure_as_exit_code_one(tmp_path, monkeypatch):
    from shumary import recorder
    from shumary.errors import ShumaryError

    def _fail(out):
        raise ShumaryError("no permission")

    monkeypatch.setattr(recorder, "record", _fail)

    result = runner.invoke(app, ["record", str(tmp_path / "call.wav")])

    assert result.exit_code == 1
    assert "no permission" in result.output


def test_record_can_chain_into_transcription(tmp_path, monkeypatch):
    from shumary import pipeline, recorder

    target = tmp_path / "call.wav"
    transcribed = {}

    monkeypatch.setattr(recorder, "record", lambda out: out)
    monkeypatch.setattr(
        pipeline, "process_file",
        lambda wav, **kw: transcribed.setdefault("wav", wav) or wav.with_suffix(".md"),
    )

    result = runner.invoke(app, ["record", str(target), "--transcribe"])

    assert result.exit_code == 0
    assert transcribed["wav"] == target

"""CLI smoke tests using Typer's test runner."""

from typer.testing import CliRunner
from shumary.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "status" in result.output
    assert "lint" in result.output
    assert "doctor" in result.output


def test_ingest_help():
    result = runner.invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.output
    assert "--no-diarize" in result.output
    assert "--model" in result.output


def test_status_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("shumary.paths.INBOX", tmp_path / "Inbox")
    monkeypatch.setattr("shumary.paths.RAW_TRANSCRIPTS", tmp_path / "transcripts")
    monkeypatch.setattr("shumary.paths.WIKI", tmp_path / "wiki")
    (tmp_path / "Inbox").mkdir()
    (tmp_path / "transcripts").mkdir()
    (tmp_path / "wiki").mkdir()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0


def test_lint_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("shumary.paths.WIKI", tmp_path / "wiki")
    monkeypatch.setattr("shumary.paths.RAW_TRANSCRIPTS", tmp_path / "transcripts")
    (tmp_path / "wiki").mkdir()
    (tmp_path / "transcripts").mkdir()
    result = runner.invoke(app, ["lint"])
    assert result.exit_code == 0


def test_lint_json_output(tmp_path, monkeypatch):
    monkeypatch.setattr("shumary.paths.WIKI", tmp_path / "wiki")
    monkeypatch.setattr("shumary.paths.RAW_TRANSCRIPTS", tmp_path / "transcripts")
    (tmp_path / "wiki").mkdir()
    (tmp_path / "transcripts").mkdir()
    result = runner.invoke(app, ["lint", "--json"])
    assert result.exit_code == 0
    import json
    parsed = json.loads(result.output)
    assert isinstance(parsed, list)


def test_cache_list_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("shumary.paths.CACHE", tmp_path / ".cache")
    result = runner.invoke(app, ["cache", "list"])
    assert result.exit_code == 0


def test_doctor_runs():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ffprobe" in result.output

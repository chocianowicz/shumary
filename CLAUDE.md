# Shumary — CLAUDE.md

Local audio-to-transcript pipeline. No LLM calls in the pipeline itself — every stage (transcription,
diarization, mood, interaction, prosody) is deterministic/heuristic Python. Output is a single
self-contained markdown file per recording; see `README.md` for the frontmatter schema.

## Architecture

`src/shumary/pipeline.py` orchestrates a fixed stage sequence, each cached independently on disk
(`~/.cache/shumary/<slug>/`, keyed by an audio content hash + params manifest):

```
decode → transcribe → diarize → align → mood → interaction (multi-speaker only)
       → prosody (optional, needs librosa/praat-parselmouth) → render
```

- `stages/` — one module per pipeline stage, each with a `run()` entry point.
- `backends.py` — Whisper backend auto-detection (mlx → faster-whisper → openai-whisper).
- `cache.py` — per-stage JSON cache with a dependency graph (`STAGE_DEPENDENTS`) that invalidates
  downstream stages when an upstream one is recomputed.
- `models.py` — frozen dataclasses for every stage's result, each with `to_dict`/`from_dict` for
  cache (de)serialization.
- `paths.py` — cache directory + output path resolution only. No project-relative layout — this
  tool is not tied to running inside a particular directory.
- `cli.py` — Typer app: `transcribe`, `cache list/clear`, `doctor`.

## Conventions

- Every stage function takes plain data in, returns a plain dataclass out — no stage reaches into
  another stage's internals.
- `ShumaryError` (not `sys.exit`) for user-facing failures, so `process_directory` can continue past
  per-file errors in batch mode.
- Cache writes are atomic (`tmp` file + `os.replace`).
- Tests monkeypatch `shumary.paths.CACHE` per-test to avoid touching the real cache directory.

## Testing

```bash
python3 -m pytest -q
```

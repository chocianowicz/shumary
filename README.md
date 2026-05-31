# Shumary

> A personal LLM-maintained wiki fed by audio recordings.

Drop an audio file into `Inbox/`, run `shumary ingest`, and watch your knowledge base grow.

---

## Quick Start

### 1. Install system dependencies

```bash
brew install ffmpeg
```

### 2. Create a virtual environment and install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Set up speaker diarization (one-time, free)

pyannote.audio requires a free Hugging Face account:

1. Create a free account at **[huggingface.co](https://huggingface.co)**
2. Accept the model license at **[hf.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)**
3. Create a token at **[hf.co/settings/tokens](https://huggingface.co/settings/tokens)**
4. Add it to your shell:
   ```bash
   echo 'export HF_TOKEN="hf_your_token_here"' >> ~/.zshrc
   source ~/.zshrc
   ```

> **No token?** Transcription and mood analysis still work. Only speaker detection is skipped.

### 4. Check your setup

```bash
shumary doctor
```

### 5. Ingest an audio file

```bash
cp ~/Desktop/my-recording.m4a Inbox/
shumary ingest
```

This transcribes the audio (Whisper), detects speakers (pyannote), analyzes behavioral signals
(pauses, speaking rate, fillers), and saves a rich markdown transcript to `raw/transcripts/`.

### 6. Ingest the transcript into the wiki

Open this directory in a new Claude Code session and say:

> *"Please ingest `raw/transcripts/2026-05-31-my-recording.md` into the wiki."*

Claude reads `CLAUDE.md`, then reads the transcript and updates the wiki — creating or updating
pages, cross-references, the index, and the log. The wiki (`wiki/`) is the compounding artifact.

### 7. Browse in Obsidian

```bash
shumary open
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `shumary ingest [PATH]` | Transcribe one audio file or all of `Inbox/` |
| `shumary status` | Show pending transcripts, wiki stats, recent log entries |
| `shumary lint` | Run structural health checks on the wiki |
| `shumary open [TARGET]` | Open `wiki/` in Obsidian (default) |
| `shumary cache list [SLUG]` | Show pipeline cache state |
| `shumary cache clear [SLUG]` | Clear pipeline cache (all or one file, all or one stage) |
| `shumary doctor` | Check system deps, models, tokens |

### Useful flags

```bash
shumary ingest --force          # re-run all stages, ignoring cache
shumary ingest --no-diarize     # skip speaker diarization
shumary ingest --model medium   # use a different Whisper model
shumary cache clear <slug> --stage mood   # re-run only the mood stage
shumary lint --json             # machine-readable output
```

---

## Architecture

Three layers (Karpathy's LLM Wiki pattern):

**1. Raw sources** — immutable originals in `raw/`. The pipeline reads audio and writes transcripts
here. Nothing is ever deleted or modified after it lands in `raw/`.

**2. The wiki** — `wiki/` is LLM-owned markdown. Claude Code writes and maintains all pages based
on ingested transcripts. The wiki compounds over time: cross-references, entity pages, concept
summaries, an evolving overview. You read it; the LLM writes it. Browse with Obsidian.

**3. The schema** — `CLAUDE.md` is Claude's operating manual. It defines conventions, workflows
(ingest, query, lint), mood signal interpretation, and the emergent structure policy (no
predefined subdirs — Claude decides page layout as the wiki grows).

---

## Directory Structure

```
shumary/
├── CLAUDE.md                  ← Wiki agent operating manual
├── Inbox/                     ← Drop audio files here
├── raw/
│   ├── audio/                 ← Archived originals (gitignored)
│   ├── transcripts/           ← Rich markdown transcripts (committed)
│   └── assets/                ← Images and attachments
├── wiki/                      ← LLM-maintained knowledge base (Obsidian vault)
│   ├── index.md               ← Master catalog
│   ├── log.md                 ← Append-only activity log
│   └── overview.md            ← High-level synthesis
└── src/shumary/               ← Python package (the transcription pipeline)
```

---

## Supported Audio Formats

`.m4a` · `.mp3` · `.wav` · `.ogg` · `.flac` · `.mp4` · `.webm` · `.aac`

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | No | — | Hugging Face token for speaker diarization |
| `SHUMARY_WHISPER_MODEL` | No | `large` | Whisper model size (`tiny`/`base`/`small`/`medium`/`large`) |
| `SHUMARY_WHISPER_DEVICE` | No | auto | Force device: `cpu`, `mps`, or `cuda` |

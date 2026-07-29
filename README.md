# Shumary

> Audio → detailed markdown transcripts. Speaker diarization, behavioral mood signals, and
> acoustic prosody — fast, accurate, fully local.

```bash
shumary record            # capture a call: your mic + the laptop's audio
shumary transcribe my-recording.m4a
```

produces `my-recording.md`: a timestamped, per-speaker transcript with a structured frontmatter
block covering speaking rate, pauses, fillers, turn-taking dynamics, and voice/pitch signals.

There are two front ends over the same engine: this CLI, and a
[menu bar app](#the-menu-bar-app) for everyday use. Everything runs on your machine — no LLM
calls, no accounts, and nothing is ever uploaded.

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

> **No token?** Transcription and mood analysis still work. Only speaker detection is skipped
> (a tokenless `resemblyzer` fallback handles diarization instead — see below).

> **Privacy note:** your audio never leaves the machine. The HF token is only used to download
> the pyannote model weights once; all diarization runs locally afterwards. Set `HF_HUB_OFFLINE=1`
> (see [Environment Variables](#environment-variables)) once the weights are cached to guarantee
> no network calls are made on subsequent runs.

> **If pyannote fails at runtime** (e.g. unsupported audio, model load issues), shumary
> automatically falls back to the tokenless `resemblyzer` backend rather than skipping
> diarization entirely. The transcript frontmatter records which backend actually ran.

### 4. Check your setup

```bash
shumary doctor
```

### 5. Transcribe

```bash
shumary transcribe my-recording.m4a
```

Writes `my-recording.md` next to the input by default. Use `-o` to choose a different path or
directory, or point at a whole directory to batch-process every audio file in it:

```bash
shumary transcribe my-recording.m4a -o transcripts/session.md
shumary transcribe ./recordings/ -o ./transcripts/
```

---

## The menu bar app

For everyday use there's an ultraminimalist macOS menu bar app — the shumary mark in the menu bar,
a steady red dot while recording, and nothing else:

```bash
./app/scripts/build-app.sh && open dist/shumary.app
```

| | |
|---|---|
| **Start / Stop recording** | `⌘R` in the menu, or **`⌃⌥⌘R` from anywhere** |
| **Import recording…** | `⌘U` — pick existing audio to transcribe (opens at the Inbox) |
| **Open recordings folder** | `⌘O` |
| **Quit shumary** | `⌘Q` |

Sessions land in `~/Documents/shumary/Recordings/<date-time>/` with the two audio tracks, a
`metadata.json`, and transcripts in both Markdown and JSON. Recording always takes priority:
starting one pauses any transcription in progress and resumes it afterwards, and the queue is
rebuilt from disk at launch so nothing is lost to a crash.

`⌃⌥⌘R` is a separate global hotkey because a menu bar app never becomes frontmost — plain `⌘R`
only fires while the menu is open, which is no use mid-call.

The app makes **no network connections and links no networking frameworks**. `Import recording…`
reads a local file; nothing is ever uploaded.

---

## Recording a call

`shumary record` captures **both sides of a conversation** — your microphone and whatever the
laptop is playing (Zoom, Meet, Teams, a browser tab) — into a single stereo WAV:

```bash
shumary record                      # → recording-2026-07-29-1432.wav
shumary record meeting.wav          # explicit name
shumary record inbox/               # into a directory
shumary record -t                   # …and transcribe as soon as it stops
```

Press **Ctrl-C** to stop. Your voice lands on the left channel, the remote participants on the
right, and `shumary transcribe` picks this up automatically.

**Why two channels matter:** normally a transcript's speaker labels are a clustering guess.
Here your side is not a guess at all — anything arriving on the mic is you, by construction. So
you get a real name instead of `SPEAKER_XX`, interruptions are measured exactly (both channels
loud at once), and pyannote only has to separate the *remote* voices from each other, which it
does far more reliably than separating them from you.

| | |
|---|---|
| **macOS 14.4+** | Uses Core Audio process taps — no virtual audio driver, no admin rights |
| **Audio keeps playing** | The tap is passive; you still hear the call normally |
| **First run** | Compiles a small Swift helper (needs `xcode-select --install`), then reuses it |
| **Permissions** | macOS asks for microphone and system-audio recording access on first use |

> **Headphones or speakers?** Both work. Speaker output bleeds into the mic, but shumary rejects
> it: a bleed is an attenuated copy of what the system channel already holds, so it never wins
> against its own source. Headphones simply remove the question.

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `shumary record [PATH]` | Record your mic + the laptop's system audio (Ctrl-C to stop) |
| `shumary transcribe PATH` | Transcribe one audio file, or every audio file in a directory |
| `shumary cache list [SLUG]` | Show pipeline cache state |
| `shumary cache clear [SLUG]` | Clear pipeline cache (all or one file, all or one stage) |
| `shumary doctor` | Check system deps, models, tokens |

### Useful flags

```bash
shumary transcribe rec.m4a --force          # re-run all stages, ignoring cache
shumary transcribe rec.m4a --no-diarize     # skip speaker diarization
shumary transcribe rec.m4a --solo           # single-speaker recording, skip diarization
shumary transcribe rec.m4a --model medium   # use a different Whisper model
shumary transcribe rec.m4a --date 2026-05-31   # override recording date detection
shumary cache clear <slug> --stage mood     # re-run only the mood stage

shumary transcribe rec.wav --dual-channel      # force channel-aware diarization
shumary transcribe rec.wav --no-dual-channel   # diarize the mixdown instead

shumary transcribe rec.m4a --json           # also write rec.json alongside rec.md
shumary transcribe rec.m4a --progress-json  # NDJSON progress on stderr, for tooling
```

`--dual-channel` is auto-enabled for anything made by `shumary record` (via a small
`<name>.wav.shumary.json` sidecar written next to the audio). Use the flags to force it on for a
stereo file recorded elsewhere with the same left=you / right=them layout, or to turn it off.

### Speed tuning

By default, shumary uses `whisper large` — maximum quality. The per-stage timing breakdown
printed at the end of every run (e.g. `transcribe 13m 02s · diarize 5s · align <1s · …`)
shows exactly where time goes so you can make an informed trade-off.

| Flag / Env var | Speed-up | Quality impact |
|---|---|---|
| `--fast` | ~25–35% faster Whisper | Tiny — greedy decode on `large` is nearly identical for clean speech |
| `--model turbo` | ~5–8× faster Whisper | Minor — may miss tricky phrasing on noisy audio |
| `--model medium` | ~3× faster Whisper | Moderate — noticeable on dense technical content |
| `SHUMARY_WHISPER_MODEL=turbo` | same as `--model turbo` | same |

Recommended fast path for casual recordings:
```bash
shumary transcribe my-recording.m4a --fast --model turbo
```

Keep `--model large` (the default) for anything where word-level accuracy matters.

---

## What's in the output

Every transcript is one self-contained markdown file:

- **YAML frontmatter** — title, date, duration, language, word count, per-speaker breakdown,
  behavioral mood signals (speaking rate, pauses, fillers, inferred state), turn-taking
  interaction metrics (for multi-speaker recordings), and acoustic prosody (pitch, energy, voice
  quality) when `librosa`/`praat-parselmouth` are installed.
- **Speakers** — detected speaker count and talk-time breakdown.
- **Behavioral / Mood Signals** — speaking rate, pause patterns, filler words, repeated phrases;
  always hedged as `(uncertain)` hypotheses, never diagnoses.
- **Interaction Dynamics** — turn counts, response latency, interruptions, talk share over time
  (multi-speaker only).
- **Voice / Prosody** — pitch baseline/variability, energy dynamics, jitter/shimmer (when
  available).
- **Full Transcript** — timestamped, speaker-attributed prose with inline pause markers and
  low-confidence/near-silence annotations.
- **Plain Text** — the raw transcript text (optional, `--no-plain-text` to omit).

With `--json` you also get a structured twin of the same analysis — the frontmatter fields plus
every segment with its speaker, timings and confidence. The Markdown is for reading five minutes
after a meeting; the JSON is for whatever you build on top of it. (The menu bar app always writes
both.)

All stages run locally: Whisper (mlx/faster/openai backend, auto-detected) for transcription,
pyannote.audio (or a tokenless resemblyzer fallback) for diarization. No LLM calls, no network
access beyond the one-time model download.

For a `shumary record` capture the frontmatter reports `dual-channel+pyannote` as the backend,
and your side of the conversation is labelled directly instead of `SPEAKER_XX`.

---

## Supported Audio Formats

`.m4a` · `.mp3` · `.wav` · `.ogg` · `.flac` · `.mp4` · `.webm` · `.aac`

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | No | — | Hugging Face token for speaker diarization (used only to download model weights once) |
| `HF_HUB_OFFLINE` | No | — | Set to `1` to force Hugging Face Hub into offline mode. Once the pyannote weights are cached locally (`~/.cache/huggingface/hub`), no network access is needed at all — set this for guaranteed-offline, privacy-sensitive runs. `shumary doctor` suggests this once it detects the weights are already cached. |
| `SHUMARY_WHISPER_MODEL` | No | `large` | Whisper model size (`tiny`/`base`/`small`/`medium`/`large`/`turbo`) |
| `SHUMARY_WHISPER_DEVICE` | No | auto | Force device: `cpu`, `mps`, or `cuda` |
| `SHUMARY_CACHE_DIR` | No | `~/.cache/shumary` | Override the pipeline's per-stage cache directory |

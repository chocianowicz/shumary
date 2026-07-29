<div align="center">

<img src="app/Resources/logo.png" width="130" alt="shumary">

# shumary

**A local meetings assistant for macOS.**

It records both sides of your call, transcribes them on your own machine,
and never sends a single byte anywhere.

</div>

---

## The name

**shumary** — 书玛理

| | | |
|---|---|---|
| **书** | *shū* | a book — a written record |
| **玛** | *mǎ* | the syllable in the middle, as in 玛丽 (Mary) |
| **理** | *lǐ* | reason, logic, to put things in order |

*A written record, put in order.* That is the entire product. A meeting goes in as sound and comes
out as something you can read, search, and reason about — without a stranger's server in the middle.

---

## What it does

You join a call. You press **⌃⌥⌘R**. A red dot appears next to the mark in your menu bar, and you
forget about it. When the call ends you press it again, and a few minutes later a transcript is
waiting in a folder.

In between, shumary:

1. **Records two sources at once** — your microphone *and* whatever your Mac is playing (Zoom,
   Meet, Teams, a browser tab), captured as two separate channels of one file.
2. **Transcribes it locally** with Whisper, on your CPU and GPU. No queue, no upload, works offline.
3. **Separates the speakers** — you are `Me`, because anything on the mic channel is you by
   construction. The remote participants are split from each other with pyannote.
4. **Reads the conversation, not just the words** — speaking rate, pauses, filler words,
   turn-taking, interruptions, response latency, talk share, pitch and voice quality.
5. **Writes it down twice** — Markdown to read, JSON to build on.

No LLM is involved anywhere in the pipeline. Every stage is deterministic Python, so the same
audio always produces the same transcript, and nothing is invented.

---

## What it looks like

Nothing. That is the point.

There is no window, no dock icon, and no preferences pane — just the mark sitting in your menu bar,
with a **steady red dot** while it is recording:

```
  ⌘  ✳  ☾   SM ●   Fri 14:32
                ▲
                └── steady red dot: audio is being captured right now
```

Click it and you get four things and a status line:

```
┌────────────────────────────────────┐
│  recording · 04:12                 │   ← what it is doing, right now
├────────────────────────────────────┤
│  Stop recording               ⌘R   │
│  Import recording…            ⌘U   │
│  Open recordings folder       ⌘O   │
├────────────────────────────────────┤
│  Quit shumary                 ⌘Q   │
└────────────────────────────────────┘
```

The status line is the only display the app has, and it tells you everything:

| | |
|---|---|
| `idle` | nothing happening |
| `recording · 04:12` | capturing, with elapsed time |
| `transcribing · 40%` | working through a recording |
| `recording · 04:12 · 1 queued` | capturing, with a transcript waiting its turn |

**⌃⌥⌘R** starts and stops from anywhere, even with Zoom in front of you. (Plain `⌘R` only works
while the menu is open — a menu bar app never becomes the active app, so it needs a global
shortcut of its own.)

The dot never blinks. A pulsing menu bar item reads as an alarm; this is just a state.

---

## Why two channels changes everything

Most transcription tools record one mixed track and then ask a model to guess who spoke when. The
guess is wrong often enough to matter — speakers get merged, swapped, or split into people who were
never in the room.

shumary doesn't guess about you. Your microphone is one channel and the call is the other, so your
turns are a fact of how the audio was captured, not a clustering result:

```
meeting.wav
  ├── left   mic     → "Me"                            certain, by construction
  └── right  system  → "Speaker 1", "Speaker 2", …     pyannote splits the remote side
```

Three things follow:

- **You are never mislabelled**, and you get a real name instead of `SPEAKER_03`.
- **Interruptions become measurable.** Both channels loud at once is genuine overlapping speech,
  not mush.
- **Diarization gets easier.** The model only has to tell the *remote* voices apart from each
  other, which is far more reliable than telling them apart from you.

Speakers rather than headphones? Still fine. Your laptop's output bleeds into the mic, but a bleed
is an attenuated copy of what the system channel already holds, so it never beats its own source
and never becomes a false turn. Measured on a real capture: bleed at −34 dB against system at
−0.4 dB, zero false turns.

---

## Privacy

Not a policy — a property of how it is built.

- Audio is captured to your disk.
- Transcription runs on your hardware.
- Transcripts are written to your disk.
- **There is no step that transmits anything.**

The app links no networking frameworks and opens no sockets — verifiable with
`lsof -a -p <pid> -i` (zero network file descriptors) and `otool -L`. There is no account, no
sign-in, and no subscription that can lapse and lock you out of your own archive.

**Import recording…** is an *import*: a file picker on your own Mac for audio recorded elsewhere.
Nothing is ever uploaded.

The one network access in the whole project is Hugging Face downloading the pyannote model weights,
once. Set `HF_HUB_OFFLINE=1` afterwards and even that disappears.

---

## Where things land

Every session is a plain folder of plain files — no database, no proprietary container, no export
step:

```
~/Documents/shumary/
├── Inbox/                          ← drop audio here for ⌘U
└── Recordings/
    └── 2026-07-29-1432/
        ├── mic.opus                ← your side
        ├── system.opus             ← their side
        ├── metadata.json           ← duration, speakers, state
        ├── transcript.md           ← for reading
        └── transcript.json         ← for building on
```

Raw audio is kept — it is the one thing that cannot be regenerated, and a better model next year
means better transcripts from the same recording. It is compressed to Opus only *after* a
transcript exists, and the original is deleted only after both tracks are verified against it:
about **690 MB/hour becomes about 23 MB/hour**.

Recording always takes priority over transcribing. Starting a capture pauses whatever is being
transcribed and resumes it afterwards — a meeting you failed to record is gone forever, while a
transcript that arrives twenty minutes late is merely late. The queue is rebuilt from disk at
launch, so a crash or a restart costs nothing.

---

## Install

```bash
brew install ffmpeg

python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

./app/scripts/build-app.sh && open dist/shumary.app
```

macOS will ask for microphone and system-audio permission the first time you record. Building the
app needs the Xcode command line tools (`xcode-select --install`); running it does not.

**Speaker diarization** (splitting the remote voices) uses pyannote, which wants a free Hugging
Face account:

1. Sign up at [huggingface.co](https://huggingface.co)
2. Accept the licence at [hf.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. Create a token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens)
4. `echo 'export HF_TOKEN="hf_…"' >> ~/.zshrc`

Without a token everything still works; shumary falls back to a tokenless `resemblyzer` backend for
diarization, and the transcript records which one actually ran.

Check the setup any time with `shumary doctor`.

---

## The command line

The app is a front end over a CLI that does the same work, and the CLI is a fine way to use
shumary on its own:

```bash
shumary record                       # Ctrl-C to stop
shumary record inbox/ -t             # record into a folder, transcribe immediately
shumary transcribe my-recording.m4a  # → my-recording.md
shumary transcribe ./recordings/ -o ./transcripts/
```

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
shumary transcribe rec.m4a --solo           # single speaker, skip diarization
shumary transcribe rec.m4a --no-diarize     # skip diarization entirely
shumary transcribe rec.m4a --model turbo    # faster Whisper model
shumary transcribe rec.m4a --date 2026-05-31   # override date detection
shumary transcribe rec.m4a --json           # also write rec.json
shumary cache clear <slug> --stage mood     # re-run only the mood stage

shumary transcribe rec.wav --dual-channel      # force channel-aware diarization
shumary transcribe rec.wav --no-dual-channel   # diarize the mixdown instead
```

`--dual-channel` is automatic for anything `shumary record` produced, via a small
`<name>.wav.shumary.json` sidecar written beside the audio. Use the flags to force it on for a
stereo file recorded elsewhere with the same left=you / right=them layout, or to turn it off.

### Speed

The default is `whisper large` — maximum quality. Every run prints a per-stage timing breakdown
(`transcribe 13m 02s · diarize 5s · align <1s · …`) so you can see exactly what a trade-off buys.

| Flag | Speed-up | Quality impact |
|---|---|---|
| `--fast` | ~25–35% | Tiny — greedy decode is near-identical on clean speech |
| `--model turbo` | ~5–8× | Minor — may miss tricky phrasing on noisy audio |
| `--model medium` | ~3× | Moderate — noticeable on dense technical content |

```bash
shumary transcribe my-recording.m4a --fast --model turbo   # casual recordings
```

Keep `--model large` for anything where word-level accuracy matters.

---

## What's in a transcript

**`transcript.md`** — one self-contained file:

- **YAML frontmatter** — title, date, duration, language, word count, per-speaker breakdown, mood
  signals, interaction metrics, and acoustic prosody.
- **Speakers** — who was detected, and how the talk time split.
- **Behavioral / Mood Signals** — speaking rate, pause patterns, filler words, repeated phrases.
  Always hedged as `(uncertain)` hypotheses; these are observations, never diagnoses.
- **Interaction Dynamics** — turn counts, response latency, interruptions, talk share over time.
- **Voice / Prosody** — pitch baseline and variability, energy, jitter/shimmer (needs
  `librosa`/`praat-parselmouth`).
- **Full Transcript** — timestamped and speaker-attributed, with inline pause markers and
  low-confidence annotations.

**`transcript.json`** — the same analysis, structured: every frontmatter field plus each segment
with its speaker, timings and confidence. The Markdown is for you, five minutes after the meeting.
The JSON is for whatever you build on top.

For a `shumary record` capture the backend reads `dual-channel+pyannote`, and your side is labelled
`Me` instead of `SPEAKER_XX`.

---

## Supported audio formats

`.m4a` · `.mp3` · `.wav` · `.ogg` · `.flac` · `.mp4` · `.webm` · `.aac`

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | No | — | Hugging Face token for speaker diarization (used only to download model weights once) |
| `HF_HUB_OFFLINE` | No | — | Set to `1` to force Hugging Face Hub offline. Once the pyannote weights are cached (`~/.cache/huggingface/hub`) no network access is needed at all — set this for guaranteed-offline runs. `shumary doctor` suggests it once it sees the weights are cached. |
| `SHUMARY_WHISPER_MODEL` | No | `large` | Whisper model size (`tiny`/`base`/`small`/`medium`/`large`/`turbo`) |
| `SHUMARY_WHISPER_DEVICE` | No | auto | Force device: `cpu`, `mps`, or `cuda` |
| `SHUMARY_CACHE_DIR` | No | `~/.cache/shumary` | Override the per-stage pipeline cache directory |
| `SHUMARY_CLI` | No | auto | Path to the `shumary` CLI, for the menu bar app to drive |

---

## Requirements

macOS 14.4 or newer (system audio capture uses Core Audio process taps), Python 3.11–3.12, and
ffmpeg. Apple Silicon is preferred — shumary picks `mlx-whisper` automatically and it is several
times faster than the alternatives.

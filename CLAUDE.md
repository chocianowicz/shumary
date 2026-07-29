# Shumary — CLAUDE.md

Local audio-to-transcript pipeline. No LLM calls in the pipeline itself — every stage (transcription,
diarization, mood, interaction, prosody) is deterministic/heuristic Python. Output is a single
self-contained markdown file per recording; see `README.md` for the frontmatter schema.

Two front ends share one engine:

- **CLI** (`shumary record` / `shumary transcribe`) — the Python package in `src/shumary/`.
- **Menu bar app** (`app/`) — a Swift `LSUIElement` app that drives the same CLI. Build with
  `./app/scripts/build-app.sh` → `dist/shumary.app`.

The app contains **no networking code and links no networking frameworks** — verified with
`lsof -a -p <pid> -i` (zero network file descriptors) and `otool -L`. Keep it that way.

## Architecture

`src/shumary/pipeline.py` orchestrates a fixed stage sequence, each cached independently on disk
(`~/.cache/shumary/<slug>/`, keyed by an audio content hash + params manifest):

```
decode → transcribe → diarize → align → mood → interaction (multi-speaker only)
       → prosody (optional, needs librosa/praat-parselmouth) → render
```

- `stages/` — one module per pipeline stage, each with a `run()` entry point.
- `backends.py` — Whisper backend auto-detection (mlx → faster-whisper → openai-whisper).
- `recorder.py` + `native/Recorder.swift` + `native/main.swift` — mic + system-audio capture
  (see below). `Recorder.swift` is shared verbatim with the menu bar app.
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

## Recording (`shumary record`)

`native/main.swift` is a thin CLI over `native/Recorder.swift`, compiled on demand by
`recorder.build_helper()` into `<cache>/bin/shumary-record-<sources-hash>` — nothing is built at
install time, and editing either source yields a new path so a stale binary is never reused.
`Recorder.swift` must stay free of `exit()` and stderr writes: the app links the same file.

It creates a Core Audio process tap (macOS 14.4+) and puts it in one **private aggregate device**
together with the default input, so mic and system audio share a clock and cannot drift apart
over a long meeting. Output is a stereo WAV: **left = mic, right = system**.

Three behaviours exist because of hard-won failure modes — do not remove them:

- **Silent-output keep-alive.** A tap only produces frames while the output device is running,
  and an aggregate delivers nothing until *every* member stream does. On an idle machine the
  IOProc never fires and even the mic is lost. The helper plays inaudible silence through
  `AVAudioEngine` for the whole recording to hold the output device open.
- **Parent-death watchdog.** `DispatchSource.makeProcessSource(.exit)` on the parent PID. Without
  it an interrupted `shumary record` leaves the helper recording forever, holding a tap nobody
  is watching. This lives in `main.swift`, *not* the shared class — in-process it would kill the app.
- **Periodic header checkpoint.** The RIFF sizes are unknown until the recording stops, so they
  are rewritten every ~10s. A crash or SIGKILL then still leaves a valid WAV up to the last
  checkpoint instead of a file with a zero-length data chunk.

A recording is paired with a `<name>.wav.shumary.json` sidecar marking it dual-source. That is
the *only* signal the pipeline uses to auto-enable channel-aware diarization; `--dual-channel` /
`--no-dual-channel` override it.

## Menu bar app (`app/`)

`Package.swift` sits at the **repo root**, not in `app/`: SwiftPM requires every target to live
inside the package root, and the app shares `src/shumary/native/Recorder.swift` verbatim with the
CLI helper. One copy, no drift.

Sessions live in `~/Documents/shumary/Recordings/<yyyy-MM-dd-HHmm>/` and hold the audio, a
`metadata.json`, and `transcript.md` + `transcript.json`.

The menu bar mark is generated from `icon.png` by `app/scripts/make-icon.py`, which keys out the
background by projecting each pixel onto the line between the two dominant colours (so anti-aliased
edges stay smooth) and emits a **template** image — macOS re-tints it from alpha, so it follows
light/dark on its own. The recording dot stays title text, because a template image cannot hold a
colour. Re-run that script if `icon.png` changes.

Four things here are load-bearing:

- **Audio is named after its session** (`2026-07-29-1432.wav`), never `audio.wav`. The engine keys
  its stage cache on the filename stem, so a shared name makes every session collide with the last
  one and the fingerprint check rejects it.
- **`metadata.json` is the queue.** Outstanding work is rebuilt from disk at launch, so a crash
  mid-transcription costs nothing. A session found in state `transcribing` at launch means the
  previous run died — retry it, the raw audio is intact.
- **Permission is settled before Core Audio is touched** (`Permissions.ensureMicrophone`). With the
  TCC question still open, `AudioDeviceCreateIOProcIDWithBlock` blocks in `mach_msg` on coreaudiod
  *forever* rather than returning an error — no timeout, nothing to report. Capture also starts off
  the main thread so device setup can never freeze the menu.
- **Recording pauses transcription** via `SIGSTOP`/`SIGCONT` on the engine process, and resumes it
  on stop. Recording must never wait for Whisper.

## Dual-source diarization

When a recording's channels are separate sources, "who is speaking" stops being a clustering
problem for one side of the conversation:

- The mic channel is gated by RMS energy (`diarize._mic_turns`) and labelled directly — anything
  on it is the person at the keyboard, so it needs no clustering and gets a real name.
- Only the system channel goes through pyannote/resemblyzer, to split the remote participants
  from each other.
- Turns from the two channels are **not** made mutually exclusive: genuine simultaneous speech is
  exactly what makes the interaction stage's interruption metrics meaningful here.

Speaker bleed (audio played on speakers and re-captured by the mic) is rejected by
`DUAL_ECHO_MARGIN_DB`: a bleed is an attenuated copy of what the system channel already holds, so
requiring the mic to *beat* the system channel rules it out. Verified against a real capture with
bleed at −34 dB against system at −0.4 dB — zero false mic turns.

Transcription still runs on the mono mixdown; the split channels only feed diarization.

## Inbox workflow

Three tracked folders (via `.gitkeep`), all with gitignored *contents*:

- `inbox/` — drop-folder for raw audio waiting to be transcribed (inputs).
- `transcripts/` — generated markdown output (regenerable).
- `processed/` — audio archived here after a successful transcription (inbox-zero).

When the user drops files in `inbox/` and asks to **"extract from inbox"** (or "process
inbox", "transcribe the inbox", etc.), the agent should:

1. List the audio files currently in `inbox/`.
2. Run the pipeline over them, writing markdown into `transcripts/`:
   ```bash
   shumary transcribe inbox/ -o transcripts/
   ```
   (or per-file if only some are requested). Use `--solo` for single-speaker recordings,
   `--date YYYY-MM-DD` when the filename has no date, and `--fast --model turbo` when the
   user wants speed over word-level accuracy.
3. **On success, move each transcribed audio file from `inbox/` into `processed/`** so
   `inbox/` only ever holds pending work and re-runs don't reprocess done files. Only move
   files whose transcript was written successfully — leave failures in `inbox/`.
4. Report what was produced (one `transcripts/<name>.md` per input, audio now in
   `processed/`) and surface any per-file failures — `process_directory` continues past
   errors, so check its return value before moving anything.

Do not commit inbox/processed audio or generated transcripts; only the folder `.gitkeep`s
are tracked.

## Testing

```bash
python3 -m pytest -q
```

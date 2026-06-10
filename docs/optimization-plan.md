# Shumary Pipeline Optimization Plan

Status: approved 2026-06-10 · Source: full-codebase review (~2,300 lines, 13 modules)

Goal: audio → high-quality markdown reports with accurate speaker attribution,
diarization, and per-speaker mood/behavioral signals suitable for psychological
analysis.

User decisions baked into this plan:

- **Pilot before mass reprocessing** — regenerate exactly one Anton session with the
  upgraded pipeline and produce an old-vs-new quality comparison report before
  deciding about the other 16 recordings.
- **Pyannote via HF token, offline-hardened** — token is used only to download model
  weights; all inference is local. Set `HF_HUB_OFFLINE=1` after first download.
  Resemblyzer fallback stays fully supported (solo memos, tokenless machines).
- **Prosody analysis is in scope** (Phase 7) — acoustic features, fully local.
- **Solo voice memos are a real use case** — single-speaker detection is a priority
  fix, not an edge case.

---

## Review findings (reference)

### A. Correctness bugs

| ID | Bug | Where |
|----|-----|-------|
| A1 | `archived_to` frontmatter wrong on filename collision (render computes path before archive renames) | `render.py:64`, `pipeline.py:133`, `render.py:46-47` |
| A2 | `--no-diarize` writes a "skipped" marker into the diarize cache; later normal runs silently never diarize | `pipeline.py:61-64` |
| A3 | Cache keyed only by slug+stage: no invalidation on model/flag change, stale downstream stages after partial clear, slug collisions overwrite transcripts | `cache.py` |
| A4 | Resemblyzer fallback can never return 1 speaker (`_MIN_SPEAKERS=2`) — solo memos falsely split | `stages/diarize.py:46,100-138` |
| A5 | Multi-word fillers unreachable (alphabetical alternation: `"no"` shadows `"no to"`) | `mood.py:86` |
| A6 | `fmt_time` breaks past 99 min (`[226:12]`) — user has 130–226 min sessions | `utils.py:18-21` |
| A7 | Transcript date = ingest date, not recording date (wiki already shows the damage) | `pipeline.py:39` |
| A8 | `source_file: Inbox/...` hardcoded; `archive_audio` *moves* files even from outside the repo | `render.py:134` |
| A9 | `sys.exit(1)` in library code; one bad file aborts the whole inbox batch | `stages/audio.py:16-17`, `pipeline.py:143-163` |
| A10 | Speaker section claims "detected by pyannote.audio" even when resemblyzer ran (all 17 existing transcripts were resemblyzer — no HF token has ever been configured) | `render.py:173` |

### B. Quality gaps vs. the psychological goal

- B1 Mood signals are global; therapy dialogues blend therapist+patient into one number.
- B2 "Pauses" conflate intra-speaker hesitation with inter-speaker response latency.
- B3 No turn-taking metrics (interruptions/overlaps, turn lengths, talk-share over time).
- B4 Whisper silence-hallucination unmitigated (no VAD); `condition_on_previous_text=True` in fast path is a no-op (already default).
- B5 Char-proportional text splitting at speaker boundaries → misattributed quotes; word timestamps fix this.
- B6 Filler lexicon overcounts in RU/PL (`а`, `вот`, `это`, `no`, `tak`, `bo` are ordinary words).
- B7 Mixed-language recordings forced into one detected language; filler matcher should always merge EN+RU+PL.

### C. Performance

- C1 Backend: `openai-whisper` is the slowest option on Apple Silicon (M1 Pro). `mlx-whisper` ≈5–10× faster; `faster-whisper` ≈4× + built-in VAD + cheap word timestamps.
- C2 Whisper/pyannote/resemblyzer each independently decode the same `.m4a` — decode once to 16 kHz mono WAV.
- C3 `status` parses every transcript's frontmatter 4×.
- C4 Full text duplicated in every report (speaker body + Plain Text section).

### D. Duplication

`models.py` to/from_dict boilerplate (~80 lines) · `pipeline.py` 5× copy-pasted stage
pattern · `align.py` 4× AlignedSegment copying · `checks.py` duplicated regex + wiki
walks · inbox scan duplicated CLI/pipeline · `modelhub.py` duplicated warnings blocks ·
table-drivable `doctor` · `render.py:196` hardcoded `3.0` · CLI `open` shadows builtin ·
no tests for diarize/pipeline/cache-staleness.

---

## Phase 1 — Correctness fixes

1.1 **Archive before render** (`pipeline.py`, `render.py`): archive becomes step 5,
    render step 6; real archive destination flows into frontmatter. `source_file`
    records the true original path; `archive_audio` *copies* when source is outside
    `Inbox/`, moves only from Inbox.

1.2 **Honest backend attribution** (fixes A10): diarize stage returns which backend ran
    (`pyannote` / `resemblyzer` / none); renderer prints it truthfully.

1.3 **Stop poisoning the diarize cache** (A2): `--no-diarize` skips without writing;
    cache entries only represent real computation.

1.4 **Exceptions, resilient batch** (A9): `ShumaryError` instead of `sys.exit`;
    `process_inbox` continues past failures, ends with `N succeeded / M failed`,
    exit code 1 if any failed.

1.5 **Time formatting** (A6): `H:MM:SS` past an hour in `fmt_time`; hours in
    `fmt_duration`.

1.6 **Recording date** (A7): filename `YYYY-MM-DD` → file mtime → today, with the
    chosen source printed and a `--date` override.

1.7 **Filler regex ordering** (A5): sort alternatives by length descending; test that
    `"no to"` beats `"no"`.

Verification: new unit tests per item; existing suite green.

## Phase 2 — Cache integrity

2.1 **Fingerprint manifest** per slug: `{audio_hash, model_name, fast, pipeline_version}`.
    Hash mismatch → hard error (same name, different file); param mismatch →
    invalidate whisper + downstream.

2.2 **Dependency-aware invalidation**: declared DAG `whisper → align → mood`,
    `diarize → align`, `diarize → interaction`. Clearing or rewriting a stage drops all
    descendants + `DONE`.

2.3 `cache list` shows model + fingerprint.

Verification: staleness test matrix (clear whisper ⇒ align/mood gone; model change ⇒
auto-invalidate; same slug, different audio ⇒ error).

## Phase 3 — Speed: backend abstraction + decode-once

3.1 **ffmpeg pre-decode stage** (step 0): 16 kHz mono WAV into the slug cache dir,
    consumed by all stages, deleted on `mark_done`. Duration read from WAV header.

3.2 **Transcription backend protocol** (`backends.py`):
    `mlx-whisper` (preferred on this M1 Pro) · `faster-whisper` (elsewhere; VAD) ·
    `openai-whisper` (fallback). Auto-selected, `--backend` / `SHUMARY_BACKEND`
    override. All backends emit **word timestamps** stored on `Segment.words` and
    serialized to cache. `--fast` maps per-backend; openai path sets
    `condition_on_previous_text=False`. `doctor` reports installed/chosen backends.

3.3 **Hallucination guard**: faster-whisper VAD on by default; other backends get a
    post-filter dropping repeated-text + high `no_speech_prob` segments, with removals
    counted in the report's quality notes.

Expected: 13 min transcribe → ~2–4 min at equal quality on the M1 Pro.

## Phase 4 — Alignment + diarization quality

4.1 **Word-level alignment** (`align.py`): each word assigned to the max-overlap turn;
    consecutive same-speaker words regrouped; piece times from actual word times.
    Char-proportional split remains only as the no-word-data fallback.

4.2 **Pyannote setup, offline-hardened**: HF token in `~/.zshrc`; first run downloads
    weights; then `HF_HUB_OFFLINE=1` documented/set so no network is ever needed
    again. README privacy note: audio never leaves the machine; the token only
    authenticates the one-time weight download. Runtime pyannote failure falls back to
    resemblyzer (today it returns None).

4.3 **Single-speaker detection — priority** (solo memos are real): allow k=1 in the
    fallback via cluster-cohesion check (tight k=1 cohesion + weak k=2 silhouette →
    one speaker); pyannote handles k=1 natively. Optional `--solo` flag to skip
    diarization entirely. Tuned against one known-solo and one known-duo recording.

Verification: golden-file alignment test (synthetic two-speaker segment splits exactly
at the turn boundary); solo memo yields `count: 1`.

## Phase 5 — Mood signals v2 (per-speaker)

5.1 **Per-speaker signals**: new `SpeakerMood`; `MoodSignals.per_speaker` added,
    global fields kept (schema versioned via Phase 2 manifest). Per speaker:
    duration-weighted speaking rate, variability, intra-speaker pauses, fillers split
    into **hesitations** vs **discourse markers** (B6; EN+RU+PL lexicons always
    merged — B7), repeated phrases, low-confidence count.

5.2 **Interaction metrics** (new `stages/interaction.py`, multi-speaker only): turn
    counts/lengths per speaker, response-latency distribution per pair,
    overlap/interruption counts (pyannote overlapping turns; omitted under
    resemblyzer), talk-share per session third. Own cache stage after `diarize`.

5.3 **Inference text** per speaker + global; hedged `(uncertain)` wording preserved;
    `render` reads `LONG_PAUSE_SEC` instead of hardcoded `3.0`.

5.4 **CLAUDE.md schema update**: document `mood_signals.per_speaker` and
    `interaction` frontmatter blocks + interpretation guidance (latency, overlaps).

## Phase 6 — Report quality + hygiene

6.1 Renderer: per-speaker mood table, interaction section, inline `*(pause 4.2s)*`
    markers (implements the dead comment at `render.py:237`), consecutive same-speaker
    segments merged into readable paragraphs, `--no-plain-text` flag, H:MM:SS
    timestamps.

6.2 Deduplication sweep: generic dataclass serde; stage-descriptor loop in
    `pipeline.py` (shares the Phase 2 DAG); `list_inbox_audio()`; single-pass wiki
    scanner in `checks.py`; table-driven `doctor`; one warnings context manager;
    CLI `open` → `open_` with explicit command name.

6.3 Tests for the gaps: pipeline orchestration with fake stages, diarize merging,
    per-speaker mood, interaction metrics, batch-failure continuation.

## Phase 7 — Acoustic prosody analysis (new)

7.1 **Feature extraction** (new `stages/prosody.py`, runs on the Phase 3 WAV, fully
    local): per aligned segment — F0 mean/range (pitch), RMS energy dynamics,
    jitter/shimmer voice quality. Libraries: `librosa` (pitch/energy) +
    `praat-parselmouth` (voice quality). Cached stage after `align` in the DAG.

7.2 **Per-speaker aggregation**: pitch baseline + variability, energy dynamics,
    voice-quality summary per speaker; deltas across session thirds (e.g., flattening
    pitch toward the end). Added to frontmatter under `prosody:`; same epistemic-
    humility rules as mood signals.

7.3 **CLAUDE.md interpretation table** for prosody signals (monotone delivery,
    raised-pitch stress markers, etc.), all hedged `(uncertain)`.

## Phase 8 — Pilot validation (gate before mass reprocessing)

8.1 Re-run **one Anton session** (suggest `12 listopada 2025.m4a` — session 8, most
    recent) through the full upgraded pipeline into a **separate output**
    (`raw/transcripts/pilot/`), leaving the original transcript untouched.

8.2 Produce `docs/pilot-comparison.md`: old vs new side by side — transcription
    accuracy spot-checks, speaker-attribution accuracy on sampled quotes, signal
    richness (global-only vs per-speaker + interaction + prosody), wall-clock time.

8.3 **Decision gate**: based on the comparison, decide whether to reprocess the
    remaining 16 recordings and how the wiki refresh is handled.

---

## Execution order

1. Phases 1–2 (one session, pure fixes)
2. Phase 3 (one session — biggest speed win)
3. Phases 4–5 (the quality core; HF token already configured 2026-06-10)
4. Phases 6–7
5. Phase 8 pilot → user decision on the archive

---

# Implementation Specs (appendix — makes Phases 3, 4.3, 5, 7 executable without design judgment)

## Spec 3 — Transcription backend protocol

### Normalized data model (`models.py`)

```python
@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str          # stripped — openai/mlx words carry a leading space, strip it
    prob: float        # word-level probability; 1.0 if backend doesn't provide it

@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    words: tuple[Word, ...] = ()       # empty = word timestamps unavailable
```

`TranscriptionResult` unchanged except segments now carry words. Cache schema bump:
`pipeline_version` in the Phase 2 manifest increments to invalidate old whisper caches.

### Protocol (`backends.py`)

```python
class TranscribeBackend(Protocol):
    name: str  # "mlx" | "faster" | "openai"
    def available(self) -> bool: ...
    def transcribe(self, wav: Path, model_name: str, *, fast: bool) -> TranscriptionResult: ...
```

Selection order in `pick_backend()`:
1. `--backend` flag / `SHUMARY_BACKEND` env (error if named backend unavailable)
2. `mlx` if `sys.platform == "darwin"` and `platform.machine() == "arm64"` and `mlx_whisper` imports
3. `faster` if `faster_whisper` imports
4. `openai` (current behavior, fallback)

`doctor` prints all three with availability and which one `pick_backend()` chose.

### Per-backend mapping

| Concept | openai-whisper | faster-whisper | mlx-whisper |
|---|---|---|---|
| call | `model.transcribe(str(wav), word_timestamps=True, verbose=False, condition_on_previous_text=False, **fastopts)` | `model.transcribe(str(wav), word_timestamps=True, vad_filter=True, condition_on_previous_text=False, **fastopts)` | `mlx_whisper.transcribe(str(wav), path_or_hf_repo=repo, word_timestamps=True, condition_on_previous_text=False, **fastopts)` |
| model load | `whisper.load_model(name, device=...)` (modelhub lru_cache, as today) | `WhisperModel(mapped_name, device="cpu", compute_type="int8")` (lru_cache) | none — repo string passed per call; weights cached by HF hub |
| `--fast` opts | `temperature=0.0` (kills the temperature-fallback retries — main saving) | `beam_size=1, temperature=0.0` | `temperature=0.0` |
| result shape | dict: `segments[i].words = [{"word"," foo","start","end","probability"}]` | generator of `Segment(..., words=[Word(start,end,word,probability)])` — **must be consumed inside the stage** (it streams); `info.language` | dict, same shape as openai |
| `no_speech_prob` / `avg_logprob` | per segment dict | on `Segment` | per segment dict |
| VAD | no — use post-filter (Spec 3-VAD) | built-in `vad_filter=True` | no — use post-filter |

Model name mapping:

| shumary name | openai | faster-whisper | mlx repo |
|---|---|---|---|
| tiny/base/small/medium | same | same | `mlx-community/whisper-<name>-mlx` |
| large | `large` | `large-v3` | `mlx-community/whisper-large-v3-mlx` |
| turbo | `turbo` | `turbo` (≥1.1) or `deepdml/faster-whisper-large-v3-turbo-ct2` | `mlx-community/whisper-large-v3-turbo` |

> Executor note: HF repo ids and the faster-whisper `turbo` alias must be smoke-tested
> at implementation time (one 10-second WAV through each installed backend) before
> wiring into the pipeline — do not trust this table blindly. Mark unavailable repos
> in a `KNOWN_REPOS` dict with the verified ids.

### Spec 3-VAD — hallucination post-filter (openai + mlx paths)

Drop segment `i` when **all** hold:
- `no_speech_prob >= 0.5`
- normalized text equals normalized text of segment `i-1` (casefold, strip punctuation)
- segment duration ≥ 2.0 s

Count drops; pipeline prints `filtered N hallucination-suspect segment(s)` and the count
goes into the report's quality notes. New config: `HALLUCINATION_NO_SPEECH = 0.5`.

## Spec 4.1 — Word-level alignment rule (`align.py`)

For segments with non-empty `words` and turns present:

1. For each word, candidate turns = turns with temporal overlap > 0 against
   `[word.start, word.end]`. Assign to max-overlap turn.
2. Zero-overlap word (falls in a diarization gap): inherit the previous word's speaker;
   if it's the first word, assign to the turn whose midpoint is nearest.
3. Group consecutive same-speaker words into one `AlignedSegment`:
   `start = first word.start`, `end = last word.end`, `text = " ".join(word.text)`,
   `avg_logprob`/`no_speech_prob` copied from the parent Whisper segment.
4. Segments with empty `words`: existing char-proportional fallback, unchanged.

Golden test: segment 0.0–10.0 s, words at 1 s intervals, turn boundary at 5.0 s →
exactly two pieces, words 0–4 / 5–9, no text lost (assert concatenation equality).

## Spec 4.3 — Single-speaker decision rule (`stages/diarize.py`)

Resemblyzer path, before `_estimate_speakers`:

```python
SOLO_SILHOUETTE = 0.10   # config.py — tunable
```

1. Compute `sil2 = _mean_silhouette(E_sub, fcluster(Z, 2))` (existing machinery).
2. If `sil2 < SOLO_SILHOUETTE`: return a single turn list — one `SPEAKER_00` turn
   spanning first-to-last window — and print
   `resemblyzer: weak 2-way split (silhouette 0.07) → treating as single speaker`.
3. Else proceed to k=2..8 estimation exactly as today.

Pyannote path: no change (handles k=1 natively).

**Tuning procedure (must be executed, not skipped):** run the rule against
(a) one known-solo memo and (b) `raw/audio/12 listopada 2025.m4a` (known duo).
Requirements: solo `sil2 < SOLO_SILHOUETTE` and duo `sil2 ≥ 2 × SOLO_SILHOUETTE`.
If the margin fails, adjust `SOLO_SILHOUETTE` to the geometric mean of the two observed
scores and record both observed values in a comment next to the constant.
This task needs human-visible output — print both scores in the test run summary.

`--solo` CLI flag: skips diarization, marks `speakers.count: 1`,
`backend: "user-flag"`.

## Spec 5 — Interaction metric definitions (`stages/interaction.py`)

Preprocessing: merge consecutive turns of the **same** speaker separated by < 1.0 s
into one *utterance*. All metrics below operate on utterances except overlap detection,
which uses raw turns.

| Metric | Definition |
|---|---|
| `turn_count` (per speaker) | number of utterances |
| `avg_turn_sec` | mean utterance duration |
| Response latency | for each consecutive utterance pair (A then B, different speakers, `B.start >= A.end`): `B.start − A.end`. Latencies > 30 s are excluded (topic break / silence, counted separately as `long_gaps`). Report median and p90 **per responding speaker** |
| Overlap | raw-turn intersection > 0.1 s between different speakers. `overlap_count`, `overlap_total_sec`. **Pyannote backend only** — resemblyzer turns cannot overlap by construction → fields are `None` |
| Interruption | overlap where B starts strictly inside A's turn AND A's turn ends within 2.0 s after `B.start` (A yields). Counted as `interruptions_made[B]` / `interruptions_received[A]`. Overlaps where A keeps talking ≥ 2.0 s past B's end and B's turn < 1.0 s are back-channels — counted in `overlap_count` but not as interruptions |
| `talk_share_by_third` | recording split into 3 equal time windows; per window, each speaker's speech seconds ÷ total speech seconds in that window (not wall-clock) |

```python
@dataclass(frozen=True)
class SpeakerInteraction:
    speaker: str
    turn_count: int
    avg_turn_sec: float
    median_response_latency_sec: float | None   # None if never responds
    p90_response_latency_sec: float | None
    interruptions_made: int
    interruptions_received: int
    talk_share_by_third: tuple[float, float, float]

@dataclass(frozen=True)
class InteractionMetrics:
    backend: str                    # "pyannote" | "resemblyzer"
    overlap_count: int | None
    overlap_total_sec: float | None
    long_gaps: int                  # inter-speaker gaps > 30 s
    speakers: tuple[SpeakerInteraction, ...]
```

Stage runs only when speaker count ≥ 2. Cache stage name: `interaction`,
DAG: `diarize → interaction`. New config constants: `UTTERANCE_MERGE_GAP = 1.0`,
`OVERLAP_MIN = 0.1`, `INTERRUPT_YIELD_SEC = 2.0`, `LATENCY_CAP_SEC = 30.0`.

## Spec 7 — Prosody extraction (`stages/prosody.py`)

Dependencies: `librosa`, `praat-parselmouth` (add to `pyproject.toml` as an
optional extra `[prosody]`; stage degrades gracefully when missing, like diarization).

Input: the Phase 3 WAV (16 kHz mono) + aligned segments. Skip segments < 1.0 s.

Per segment:

```python
y, sr = librosa.load(wav, sr=16000)                     # loaded ONCE per file
seg_y = y[int(seg.start*sr):int(seg.end*sr)]

# Pitch — pyin, speech band
f0, voiced_flag, _ = librosa.pyin(seg_y, fmin=65, fmax=400, sr=sr, frame_length=2048)
voiced = f0[voiced_flag & ~numpy.isnan(f0)]
# require ≥ 10 voiced frames, else f0 fields = None
f0_median = float(numpy.median(voiced))
f0_range_st = 12 * numpy.log2(numpy.percentile(voiced, 90) / numpy.percentile(voiced, 10))

# Energy
rms = librosa.feature.rms(y=seg_y)[0]
rms_db = 20 * numpy.log10(rms + 1e-10)
energy_mean_db = float(rms_db.mean())
energy_range_db = float(numpy.percentile(rms_db, 90) - numpy.percentile(rms_db, 10))

# Voice quality — parselmouth/Praat (wrap in try/except; None on failure)
snd = parselmouth.Sound(str(wav)).extract_part(seg.start, seg.end)
pp  = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
jitter  = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
shimmer = parselmouth.praat.call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
```

Per-speaker aggregation (only voiced-valid segments):
- `f0_baseline_hz` = median of segment medians
- `f0_variability_st` = IQR of per-segment medians expressed in semitones vs baseline
  (`12*log2(x/baseline)`)
- `energy_mean_db`, `energy_dynamics_db` = medians of per-segment values
- `jitter_local`, `shimmer_local` = medians (drop NaN)
- `trend_by_third`: per session third, median segment-F0 delta vs baseline in
  semitones — surfaces pitch flattening/rising over the session

Frontmatter block:

```yaml
prosody:
  - speaker: SPEAKER_00
    f0_baseline_hz: 142.3
    f0_variability_st: 2.1
    energy_mean_db: -23.4
    energy_dynamics_db: 8.7
    jitter_local: 0.012
    shimmer_local: 0.061
    f0_trend_by_third_st: [0.0, -0.4, -1.1]
```

Same epistemic rules as mood: values are descriptive; interpretation lives in
CLAUDE.md's table and is always `(uncertain)`. Cache stage `prosody`,
DAG: `align → prosody`.

## Executor (Sonnet) ground rules

- One phase per session; within a phase, one numbered task per commit; run
  `python -m pytest` after every task — a failing suite blocks the next task.
- Anything marked **Executor note** or **tuning procedure** requires running real
  commands and pasting observed output into the commit message — no assumed values.
- External API shapes (backend result formats, HF repo ids, Praat call signatures)
  must be smoke-tested with a throwaway script before integration.
- Never modify files under `wiki/` or `raw/transcripts/` (pilot output goes to
  `raw/transcripts/pilot/`).
- If a spec conflicts with observed library behavior, stop and report — do not
  improvise a workaround silently.

"""Pipeline orchestrator — coordinates all stages with per-stage caching."""

import time
from datetime import datetime
from pathlib import Path

from shumary import cache as cache_mod
from shumary.config import WHISPER_MODEL
from shumary.models import (
    TranscriptionResult,
    Turn,
    AlignedSegment,
    SpeakerStats,
    MoodSignals,
)
from shumary.paths import INBOX, transcript_path_for, relative_to_repo
from shumary.config import SUPPORTED_FORMATS
from shumary.utils import slugify, fmt_duration
from shumary.stages import audio as audio_stage
from shumary.stages import transcribe as transcribe_stage
from shumary.stages import diarize as diarize_stage
from shumary.stages import align as align_stage
from shumary.stages import mood as mood_stage
from shumary.stages import render as render_stage


def process_file(
    audio: Path,
    *,
    force: bool = False,
    no_diarize: bool = False,
    fast: bool = False,
    model_name: str = WHISPER_MODEL,
) -> Path:
    audio_stage.validate(audio)
    _start = time.monotonic()
    timings: dict[str, float | str] = {}

    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(audio.stem)
    duration_sec = audio_stage.duration_seconds(audio)

    _header(audio, date_str, slug, duration_sec)

    # ── 1. Transcribe ──────────────────────────────────────────────────────────
    _step(1, "Whisper Transcription")
    t0 = time.monotonic()
    transcription, cached = _run_cached(
        slug, "whisper", force,
        compute=lambda: transcribe_stage.run(audio, model_name, fast=fast),
        serialize=lambda r: r.to_dict(),
        deserialize=TranscriptionResult.from_dict,
    )
    timings["transcribe"] = "cached" if cached else time.monotonic() - t0
    print(f"  language: {transcription.language}  |  words: {transcription.word_count}")

    # ── 2. Diarize ─────────────────────────────────────────────────────────────
    _step(2, "Speaker Diarization")
    if no_diarize:
        print("  skipped (--no-diarize)")
        turns_raw: dict | None = {"skipped": "no_diarize_flag"}
        if not cache_mod.is_cached(slug, "diarize") or force:
            cache_mod.write(slug, "diarize", turns_raw)  # type: ignore[arg-type]
        turns = None
        timings["diarize"] = "skipped"
    else:
        t0 = time.monotonic()
        turns, cached = _run_cached(
            slug, "diarize", force,
            compute=lambda: diarize_stage.run(audio),
            serialize=lambda t: {"turns": [x.to_dict() for x in t]} if t else {"skipped": "no_token"},
            deserialize=lambda d: None if "skipped" in d else [Turn.from_dict(t) for t in d["turns"]],
        )
        timings["diarize"] = "cached" if cached else time.monotonic() - t0

    speaker_stats: SpeakerStats | None = diarize_stage.speaker_stats(turns)
    if speaker_stats:
        for s in speaker_stats.speakers:
            print(f"  {s.label}: {fmt_duration(s.duration_sec)} ({s.pct}%)")
    else:
        print("  (no speaker data)")

    # ── 3. Align ───────────────────────────────────────────────────────────────
    _step(3, "Align Speakers → Segments")

    def _load_aligned(d: dict) -> list[AlignedSegment]:
        return [AlignedSegment.from_dict(s) for s in d["segments"]]

    t0 = time.monotonic()
    aligned, cached = _run_cached(
        slug, "align", force,
        compute=lambda: align_stage.run(transcription.segments, turns),
        serialize=lambda segs: {"segments": [s.to_dict() for s in segs]},
        deserialize=_load_aligned,
    )
    timings["align"] = "cached" if cached else time.monotonic() - t0
    print(f"  {len(aligned)} segments")

    # ── 4. Mood analysis ───────────────────────────────────────────────────────
    _step(4, "Behavioral / Mood Analysis")
    t0 = time.monotonic()
    mood, cached = _run_cached(
        slug, "mood", force,
        compute=lambda: mood_stage.run(aligned, transcription.language),
        serialize=lambda m: m.to_dict(),
        deserialize=MoodSignals.from_dict,
    )
    timings["mood"] = "cached" if cached else time.monotonic() - t0
    print(f"  rate: {mood.avg_speaking_rate_wps} wps  |  pauses: {mood.long_pauses}  |  fillers: {mood.filler_word_count}")
    print(f"  state: {mood.inferred_state}")

    # ── 5. Render transcript ───────────────────────────────────────────────────
    _step(5, "Render Transcript")
    t0 = time.monotonic()
    out_path = transcript_path_for(date_str, slug)
    render_stage.save_transcript(
        audio=audio,
        transcription=transcription,
        segments=aligned,
        mood=mood,
        speaker_stats=speaker_stats,
        date_str=date_str,
        slug=slug,
        duration_sec=duration_sec,
        out_path=out_path,
    )
    timings["render"] = time.monotonic() - t0
    print(f"  📝  {relative_to_repo(out_path)}")

    # ── 6. Archive audio ───────────────────────────────────────────────────────
    _step(6, "Archive Audio")
    t0 = time.monotonic()
    archived = render_stage.archive_audio(audio)
    timings["archive"] = time.monotonic() - t0
    print(f"  📦  {relative_to_repo(archived)}")

    cache_mod.mark_done(slug)
    elapsed = time.monotonic() - _start
    _footer(out_path, speaker_stats, elapsed, timings)
    return out_path


def process_inbox(
    *,
    force: bool = False,
    no_diarize: bool = False,
    fast: bool = False,
    model_name: str = WHISPER_MODEL,
) -> list[Path]:
    INBOX.mkdir(exist_ok=True)
    files = sorted(
        f for f in INBOX.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    )
    if not files:
        print("📭  Inbox is empty — no audio files to process.")
        return []

    print(f"📬  Found {len(files)} file(s) in Inbox/\n")
    return [
        process_file(f, force=force, no_diarize=no_diarize, fast=fast, model_name=model_name)
        for f in files
    ]


# ── Cache wrapper ──────────────────────────────────────────────────────────────

def _run_cached(slug, stage, force, *, compute, serialize, deserialize):
    if not force and cache_mod.is_cached(slug, stage):
        raw = cache_mod.read(slug, stage)
        return deserialize(raw), True  # type: ignore[arg-type]
    result = compute()
    cache_mod.write(slug, stage, serialize(result))
    return result, False


# ── Print helpers ──────────────────────────────────────────────────────────────

def _header(audio: Path, date_str: str, slug: str, duration_sec: float | None) -> None:
    dur = f"  |  ⏱  {fmt_duration(duration_sec)}" if duration_sec else ""
    print(f"\n{'═'*60}")
    print(f"  📁  {audio.name}")
    print(f"  📅  {date_str}  |  🔤  {slug}{dur}")
    print(f"{'═'*60}\n")


def _step(n: int, label: str) -> None:
    print(f"── Step {n}/6  {label} {'─'*(44 - len(label))}")


def _fmt_stage_timing(v: float | str) -> str:
    if isinstance(v, str):
        return f"({v})"
    if v < 1.0:
        return "<1s"
    mins, secs = divmod(int(v), 60)
    return f"{mins}m {secs:02d}s" if mins else f"{secs}s"


def _footer(
    transcript_path: Path,
    speaker_stats: SpeakerStats | None,
    elapsed_sec: float,
    timings: dict[str, float | str],
) -> None:
    rel = relative_to_repo(transcript_path)
    mins, secs = divmod(int(elapsed_sec), 60)
    elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    stage_order = ["transcribe", "diarize", "align", "mood", "render", "archive"]
    timing_line = "  ·  ".join(
        f"{k} {_fmt_stage_timing(timings.get(k, 0.0))}" for k in stage_order
    )

    print(f"\n{'═'*60}")
    print(f"  ✅  Done in {elapsed_str} — ready for wiki ingest.")
    print(f"  ⏱   {timing_line}")
    print(f"\n  → Open Claude Code and say:")
    print(f'     "Please ingest {rel}"')
    if speaker_stats and speaker_stats.count > 1:
        labels = [s.label for s in speaker_stats.speakers]
        print(f"\n  → Name the speakers:")
        print(f'     "{", ".join(labels)} — who are they?"')
    print(f"{'═'*60}\n")

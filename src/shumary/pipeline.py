"""Pipeline orchestrator — coordinates all stages with per-stage caching."""

import time
from pathlib import Path

from shumary import cache as cache_mod
from shumary.backends import pick_backend
from shumary.config import WHISPER_MODEL
from shumary.errors import ShumaryError
from shumary.models import (
    TranscriptionResult,
    DiarizationResult,
    AlignedSegment,
    InteractionMetrics,
    ProsodyResult,
    SpeakerStats,
    MoodSignals,
    Turn,
)
from shumary.paths import output_path_for
from shumary.utils import slugify, fmt_duration, resolve_recording_date
from shumary.stages import audio as audio_stage
from shumary.stages import transcribe as transcribe_stage
from shumary.stages import diarize as diarize_stage
from shumary.stages import align as align_stage
from shumary.stages import mood as mood_stage
from shumary.stages import interaction as interaction_stage
from shumary.stages import prosody as prosody_stage
from shumary.stages import render as render_stage


def process_file(
    audio: Path,
    *,
    force: bool = False,
    no_diarize: bool = False,
    solo: bool = False,
    fast: bool = False,
    model_name: str = WHISPER_MODEL,
    date_override: str | None = None,
    backend_override: str | None = None,
    plain_text: bool = True,
    output: Path | None = None,
) -> Path:
    audio_stage.validate(audio)
    backend = pick_backend(backend_override)
    _start = time.monotonic()
    timings: dict[str, float | str] = {}

    date_str, date_source = resolve_recording_date(audio, date_override)
    slug = slugify(audio.stem)

    _header(audio, date_str, slug, None, date_source)

    prev_manifest = cache_mod.read_manifest(slug)
    invalidated = cache_mod.ensure_manifest(slug, audio, model_name, fast)
    if invalidated:
        print(f"  ♻️  cache invalidated: {_describe_invalidation(prev_manifest, model_name, fast, invalidated)}\n")

    # ── 0. Pre-decode to 16 kHz mono WAV ──────────────────────────────────────────
    _step(0, "Pre-decode Audio")
    t0 = time.monotonic()
    wav = audio_stage.decode_to_wav(audio, slug)
    timings["decode"] = time.monotonic() - t0
    duration_sec = audio_stage.wav_duration_seconds(wav)
    print(f"  🎚  {wav.name}  |  ⏱  {fmt_duration(duration_sec)}")

    # ── 1. Transcribe ──────────────────────────────────────────────────────────
    _step(1, "Whisper Transcription")
    transcription = _run_timed_stage(
        timings, "transcribe", slug, "whisper", force,
        compute=lambda: transcribe_stage.run(wav, model_name, fast=fast, backend=backend),
        serialize=lambda r: r.to_dict(),
        deserialize=TranscriptionResult.from_dict,
    )
    print(f"  language: {transcription.language}  |  words: {transcription.word_count}")

    # ── 2. Diarize ─────────────────────────────────────────────────────────────
    _step(2, "Speaker Diarization")
    if solo:
        print("  skipped (--solo) — single SPEAKER_00 turn")
        turns = [Turn(start=0.0, end=duration_sec, speaker="SPEAKER_00")]
        diarize_backend = "user-flag"
        timings["diarize"] = "skipped"
    elif no_diarize:
        print("  skipped (--no-diarize)")
        turns = None
        diarize_backend: str | None = None
        timings["diarize"] = "skipped"
    else:
        diarization = _run_timed_stage(
            timings, "diarize", slug, "diarize", force,
            compute=lambda: diarize_stage.run(wav),
            serialize=lambda d: d.to_dict() if d else {"skipped": "no_token"},
            deserialize=lambda d: None if "skipped" in d else DiarizationResult.from_dict(d),
        )
        turns = list(diarization.turns) if diarization else None
        diarize_backend = diarization.backend if diarization else None

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

    aligned = _run_timed_stage(
        timings, "align", slug, "align", force,
        compute=lambda: align_stage.run(transcription.segments, turns),
        serialize=lambda segs: {"segments": [s.to_dict() for s in segs]},
        deserialize=_load_aligned,
    )
    print(f"  {len(aligned)} segments")

    # ── 4. Mood analysis ───────────────────────────────────────────────────────
    _step(4, "Behavioral / Mood Analysis")
    mood = _run_timed_stage(
        timings, "mood", slug, "mood", force,
        compute=lambda: mood_stage.run(aligned, transcription.language),
        serialize=lambda m: m.to_dict(),
        deserialize=MoodSignals.from_dict,
    )
    print(f"  rate: {mood.avg_speaking_rate_wps} wps  |  pauses: {mood.long_pauses}  |  fillers: {mood.filler_word_count}")
    print(f"  state: {mood.inferred_state}")

    # ── 5. Interaction metrics ─────────────────────────────────────────────────
    interaction: InteractionMetrics | None = None
    if turns and speaker_stats and speaker_stats.count >= 2:
        _step(5, "Interaction Dynamics")
        interaction = _run_timed_stage(
            timings, "interaction", slug, "interaction", force,
            compute=lambda: interaction_stage.run(turns, diarize_backend or "resemblyzer"),
            serialize=lambda i: i.to_dict(),
            deserialize=InteractionMetrics.from_dict,
        )
        turn_summary = ", ".join(
            f"{s.speaker}: {s.turn_count} turns" for s in interaction.speakers
        )
        overlap_summary = (
            f"{interaction.overlap_count} overlaps"
            if interaction.overlap_count is not None
            else "overlap: n/a"
        )
        print(f"  {turn_summary}  |  {overlap_summary}")
    else:
        timings["interaction"] = "skipped"

    # ── 6. Prosody analysis ─────────────────────────────────────────────────────
    _step(6, "Acoustic Prosody Analysis")
    if not prosody_stage.available():
        print("  ⚠  librosa/praat-parselmouth not installed — skipping prosody extraction")
        prosody = None
        timings["prosody"] = "skipped"
    else:
        prosody = _run_timed_stage(
            timings, "prosody", slug, "prosody", force,
            compute=lambda: prosody_stage.run(wav, aligned),
            serialize=lambda p: p.to_dict(),
            deserialize=ProsodyResult.from_dict,
        )
        if prosody is not None:
            print(f"  {prosody_stage.summarize(prosody)}")
        else:
            print("  (no prosody data)")

    # ── 7. Render transcript ───────────────────────────────────────────────────
    _step(7, "Render Transcript")
    t0 = time.monotonic()
    out_path = output_path_for(audio, output)
    render_stage.save_transcript(
        audio=audio,
        transcription=transcription,
        segments=aligned,
        mood=mood,
        speaker_stats=speaker_stats,
        diarize_backend=diarize_backend,
        interaction=interaction,
        prosody=prosody,
        date_str=date_str,
        slug=slug,
        duration_sec=duration_sec,
        out_path=out_path,
        plain_text=plain_text,
    )
    timings["render"] = time.monotonic() - t0
    print(f"  📝  {out_path}")

    cache_mod.mark_done(slug)
    elapsed = time.monotonic() - _start
    _footer(out_path, speaker_stats, elapsed, timings)
    return out_path


def process_directory(
    directory: Path,
    *,
    force: bool = False,
    no_diarize: bool = False,
    solo: bool = False,
    fast: bool = False,
    model_name: str = WHISPER_MODEL,
    date_override: str | None = None,
    backend_override: str | None = None,
    plain_text: bool = True,
    output_dir: Path | None = None,
) -> tuple[list[Path], list[Path]]:
    """Process every audio file in `directory`, continuing past per-file failures.

    Returns `(succeeded, failed)` — `succeeded` holds the output transcript
    paths, `failed` holds the audio paths that raised a `ShumaryError`.
    """
    files = audio_stage.list_audio_files(directory)
    if not files:
        print(f"📭  No supported audio files in {directory}.")
        return [], []

    print(f"📬  Found {len(files)} audio file(s) in {directory}\n")

    succeeded: list[Path] = []
    failed: list[Path] = []
    for f in files:
        try:
            succeeded.append(
                process_file(
                    f, force=force, no_diarize=no_diarize, solo=solo, fast=fast,
                    model_name=model_name, date_override=date_override,
                    backend_override=backend_override, plain_text=plain_text,
                    output=output_dir,
                )
            )
        except ShumaryError as exc:
            print(f"\n❌  {f.name}: {exc}\n")
            failed.append(f)

    print(f"\n{len(succeeded)} succeeded / {len(failed)} failed")
    return succeeded, failed


# ── Cache wrapper ──────────────────────────────────────────────────────────────

def _run_cached(slug, stage, force, *, compute, serialize, deserialize):
    if not force and cache_mod.is_cached(slug, stage):
        raw = cache_mod.read(slug, stage)
        return deserialize(raw), True  # type: ignore[arg-type]
    result = compute()
    cache_mod.write(slug, stage, serialize(result))
    return result, False


def _run_timed_stage(timings: dict[str, float | str], key: str, slug, stage, force, *, compute, serialize, deserialize):
    """`_run_cached` plus the repeated `t0 = time.monotonic(); ...;
    timings[key] = "cached" if cached else elapsed` bookkeeping shared by
    the whisper/align/mood/interaction stages.
    """
    t0 = time.monotonic()
    result, cached = _run_cached(slug, stage, force, compute=compute, serialize=serialize, deserialize=deserialize)
    timings[key] = "cached" if cached else time.monotonic() - t0
    return result


# ── Print helpers ──────────────────────────────────────────────────────────────

def _header(audio: Path, date_str: str, slug: str, duration_sec: float | None = None, date_source: str = "today") -> None:
    dur = f"  |  ⏱  {fmt_duration(duration_sec)}" if duration_sec else ""
    print(f"\n{'═'*60}")
    print(f"  📁  {audio.name}")
    print(f"  📅  {date_str} (from {date_source})  |  🔤  {slug}{dur}")
    print(f"{'═'*60}\n")


def _describe_invalidation(
    prev_manifest: dict | None,
    model_name: str,
    fast: bool,
    invalidated: list[str],
) -> str:
    """Human-readable summary of why the cache was invalidated."""
    if prev_manifest is None:
        return f"legacy cache (no manifest) — recomputing {', '.join(invalidated) or 'nothing'}"

    reasons = []
    if prev_manifest.get("model_name") != model_name:
        reasons.append(f"model changed {prev_manifest.get('model_name')} → {model_name}")
    if prev_manifest.get("fast") != fast:
        reasons.append(f"--fast changed {prev_manifest.get('fast')} → {fast}")
    if prev_manifest.get("pipeline_version") != cache_mod.PIPELINE_VERSION:
        reasons.append(
            f"pipeline version {prev_manifest.get('pipeline_version')} → {cache_mod.PIPELINE_VERSION}"
        )
    reason = "; ".join(reasons) or "parameters changed"
    return f"{reason} — recomputing {', '.join(invalidated) or 'nothing'}"


def _step(n: int, label: str) -> None:
    suffix = f"{n}/7" if n else "0"
    print(f"── Step {suffix}  {label} {'─'*(44 - len(label))}")


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
    mins, secs = divmod(int(elapsed_sec), 60)
    elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"

    stage_order = ["decode", "transcribe", "diarize", "align", "mood", "interaction", "prosody", "render"]
    timing_line = "  ·  ".join(
        f"{k} {_fmt_stage_timing(timings.get(k, 0.0))}" for k in stage_order
    )

    print(f"\n{'═'*60}")
    print(f"  ✅  Done in {elapsed_str}")
    print(f"  ⏱   {timing_line}")
    print(f"  📝  {transcript_path}")
    if speaker_stats and speaker_stats.count > 1:
        labels = ", ".join(s.label for s in speaker_stats.speakers)
        print(f"  🗣   {labels}")
    print(f"{'═'*60}\n")

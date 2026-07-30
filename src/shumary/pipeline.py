"""Pipeline orchestrator — coordinates all stages with per-stage caching."""

import time
from pathlib import Path
from typing import Callable

from shumary import cache as cache_mod
from shumary import recorder
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


# Share of total runtime each stage typically takes, used to turn discrete
# stage transitions into a percentage. Whisper dominates; the rest are noise by
# comparison. Deliberately coarse — sub-stage Whisper progress is not available
# without reaching into every backend, and a monotonic estimate is worth more
# than a precise one that stalls.
STAGE_WEIGHTS: dict[str, float] = {
    "decode": 0.02,
    "transcribe": 0.70,
    "diarize": 0.14,
    "align": 0.02,
    "mood": 0.02,
    "interaction": 0.02,
    "prosody": 0.06,
    "render": 0.02,
}

ProgressSink = Callable[[dict], None]


class _Progress:
    """Turns stage transitions into monotonic progress events.

    Skipped and cache-hit stages still report `done`, so the percentage always
    advances and never stalls on work that didn't need doing.

    Stages that take real time also report *within* themselves via `substage`,
    because two stages carry 84% of the weight: without it the percentage sits
    at 2% for the whole transcription and 72% for the whole diarization, which
    is indistinguishable from a hang. Sub-stage events are throttled to whole
    percent changes and can never reach the next stage's boundary, so the number
    only ever moves forward.
    """

    def __init__(self, sink: ProgressSink | None) -> None:
        self._sink = sink
        self._completed = 0.0
        self._last_pct = -1

    def start(self, stage: str) -> None:
        self._emit({"event": "stage", "stage": stage, "status": "start"})

    def done(self, stage: str) -> None:
        self._completed = min(1.0, self._completed + STAGE_WEIGHTS.get(stage, 0.0))
        self._emit({"event": "stage", "stage": stage, "status": "done"})

    def substage(self, stage: str, fraction: float) -> None:
        """Report `fraction` (0..1) of the way through `stage`."""
        if self._sink is None:
            return
        weight = STAGE_WEIGHTS.get(stage, 0.0)
        if weight <= 0:
            return
        fraction = max(0.0, min(1.0, fraction))
        pct = self._completed + weight * fraction
        if int(pct * 100) <= self._last_pct:
            return  # throttle: nothing new to say at whole-percent resolution
        self._emit({"event": "stage", "stage": stage, "status": "progress"}, pct=pct)

    def finished(self, output: Path) -> None:
        self._completed = 1.0
        self._emit({"event": "done", "output": str(output)})

    def failed(self, message: str) -> None:
        self._emit({"event": "error", "message": message})

    def _emit(self, payload: dict, pct: float | None = None) -> None:
        if self._sink is None:
            return
        value = self._completed if pct is None else pct
        self._last_pct = max(self._last_pct, int(value * 100))
        self._sink(payload | {"pct": round(value, 4)})


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
    dual: bool | None = None,
    progress: ProgressSink | None = None,
    write_json: bool = False,
) -> Path:
    audio_stage.validate(audio)
    prog = _Progress(progress)
    backend = pick_backend(backend_override)
    _start = time.monotonic()
    timings: dict[str, float | str] = {}

    date_str, date_source = resolve_recording_date(audio, date_override)
    slug = slugify(audio.stem)

    _header(audio, date_str, slug, None, date_source)

    mic_label = _resolve_mic_label(audio, dual)
    dual_source = mic_label is not None and not solo and not no_diarize
    if dual_source and audio_stage.channel_count(audio) != 2:
        print("  ⚠  not a 2-channel recording — falling back to normal diarization\n")
        dual_source = False

    prev_manifest = cache_mod.read_manifest(slug)
    invalidated = cache_mod.ensure_manifest(slug, audio, model_name, fast, dual_source)
    if invalidated:
        print(f"  ♻️  cache invalidated: {_describe_invalidation(prev_manifest, model_name, fast, dual_source, invalidated)}\n")

    # ── 0. Pre-decode to 16 kHz mono WAV ──────────────────────────────────────────
    _step(0, "Pre-decode Audio")
    prog.start("decode")
    t0 = time.monotonic()
    wav = audio_stage.decode_to_wav(audio, slug)
    channels: tuple[Path, Path] | None = None
    if dual_source:
        # Transcription still runs on the mixed-down mono WAV; the split
        # channels only feed diarization.
        channels = audio_stage.split_channels(audio, slug)
    timings["decode"] = time.monotonic() - t0
    duration_sec = audio_stage.wav_duration_seconds(wav)
    print(f"  🎚  {wav.name}  |  ⏱  {fmt_duration(duration_sec)}")
    if channels:
        print(f"  🎙  dual-source: left = {mic_label} · right = system audio")
    prog.done("decode")

    # ── 1. Transcribe ──────────────────────────────────────────────────────────
    _step(1, "Whisper Transcription")
    prog.start("transcribe")
    transcription = _run_timed_stage(
        timings, "transcribe", slug, "whisper", force,
        compute=lambda: transcribe_stage.run(
            wav, model_name, fast=fast, backend=backend,
            progress=lambda f: prog.substage("transcribe", f),
        ),
        serialize=lambda r: r.to_dict(),
        deserialize=TranscriptionResult.from_dict,
    )
    prog.done("transcribe")
    print(f"  language: {transcription.language}  |  words: {transcription.word_count}")

    # ── 2. Diarize ─────────────────────────────────────────────────────────────
    _step(2, "Speaker Diarization")
    prog.start("diarize")
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
        if channels:
            left, right = channels
            compute_diarization = lambda: diarize_stage.run_dual(  # noqa: E731
                left, right, mic_label=mic_label or recorder.MIC_LABEL,
                progress=lambda f: prog.substage("diarize", f),
            )
        else:
            compute_diarization = lambda: diarize_stage.run(  # noqa: E731
                wav, progress=lambda f: prog.substage("diarize", f)
            )

        diarization = _run_timed_stage(
            timings, "diarize", slug, "diarize", force,
            compute=compute_diarization,
            serialize=lambda d: d.to_dict() if d else {"skipped": "no_token"},
            deserialize=lambda d: None if "skipped" in d else DiarizationResult.from_dict(d),
        )
        turns = list(diarization.turns) if diarization else None
        diarize_backend = diarization.backend if diarization else None

    prog.done("diarize")
    turn_stats: SpeakerStats | None = diarize_stage.speaker_stats(turns)
    if turn_stats:
        for s in turn_stats.speakers:
            print(f"  {s.label}: {fmt_duration(s.duration_sec)} audible")
    else:
        print("  (no speaker data)")

    # ── 3. Align ───────────────────────────────────────────────────────────────
    _step(3, "Align Speakers → Segments")
    prog.start("align")

    def _load_aligned(d: dict) -> list[AlignedSegment]:
        return [AlignedSegment.from_dict(s) for s in d["segments"]]

    aligned = _run_timed_stage(
        timings, "align", slug, "align", force,
        compute=lambda: align_stage.run(transcription.segments, turns),
        serialize=lambda segs: {"segments": [s.to_dict() for s in segs]},
        deserialize=_load_aligned,
    )
    prog.done("align")
    print(f"  {len(aligned)} segments")

    # The speaker table and every turn-derived metric are both computed from
    # *speech*, not from raw turns: an energy-gated mic turn that caught a
    # keyboard press is only distinguishable from a real one once alignment has
    # decided where the words went.
    speaker_stats: SpeakerStats | None = (
        diarize_stage.speaker_stats_from_speech(aligned) or turn_stats
    )
    if speaker_stats:
        for s in speaker_stats.speakers:
            print(f"  {s.label}: {fmt_duration(s.duration_sec)} ({s.pct}%)")
        if turn_stats:
            kept = {s.label for s in speaker_stats.speakers}
            for s in sorted(turn_stats.speakers, key=lambda s: s.label):
                if s.label not in kept:
                    print(f"  {s.label}: too little transcribed speech — not a speaker")

    # The single set of labels that count as participants. Every per-speaker
    # table — speakers, turn metrics, mood, prosody — is built from it, so they
    # cannot disagree about who was in the meeting.
    accepted_speakers = (
        {s.label for s in speaker_stats.speakers} if speaker_stats else None
    )

    pruned = diarize_stage.prune_unvoiced_turns(
        turns, aligned, keep_labels=accepted_speakers,
    )
    if turns and pruned is not None and len(pruned) < len(turns):
        print(f"  {len(turns) - len(pruned)} of {len(turns)} turns carried no speech — excluded from metrics")
    turns = pruned

    # ── 4. Mood analysis ───────────────────────────────────────────────────────
    _step(4, "Behavioral / Mood Analysis")
    prog.start("mood")
    mood = _run_timed_stage(
        timings, "mood", slug, "mood", force,
        compute=lambda: mood_stage.run(aligned, transcription.language, accepted_speakers),
        serialize=lambda m: m.to_dict(),
        deserialize=MoodSignals.from_dict,
    )
    print(f"  rate: {mood.avg_speaking_rate_wps} wps  |  pauses: {mood.long_pauses}  |  fillers: {mood.filler_word_count}")
    prog.done("mood")
    print(f"  state: {mood.inferred_state}")

    # ── 5. Interaction metrics ─────────────────────────────────────────────────
    interaction: InteractionMetrics | None = None
    if turns and speaker_stats and speaker_stats.count >= 2:
        _step(5, "Interaction Dynamics")
        prog.start("interaction")
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
    prog.done("interaction")
    if interaction is None:
        timings["interaction"] = "skipped"

    # ── 6. Prosody analysis ─────────────────────────────────────────────────────
    _step(6, "Acoustic Prosody Analysis")
    prog.start("prosody")
    if not prosody_stage.available():
        print("  ⚠  librosa/praat-parselmouth not installed — skipping prosody extraction")
        prosody = None
        timings["prosody"] = "skipped"
    else:
        prosody = _run_timed_stage(
            timings, "prosody", slug, "prosody", force,
            compute=lambda: prosody_stage.run(wav, aligned, accepted_speakers),
            serialize=lambda p: p.to_dict(),
            deserialize=ProsodyResult.from_dict,
        )
        if prosody is not None:
            print(f"  {prosody_stage.summarize(prosody)}")
        else:
            print("  (no prosody data)")

    prog.done("prosody")

    # ── 7. Render transcript ───────────────────────────────────────────────────
    _step(7, "Render Transcript")
    prog.start("render")
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
    if write_json:
        json_path = render_stage.save_transcript_json(
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
            out_path=out_path.with_suffix(".json"),
        )
        print(f"  🧾  {json_path}")
    timings["render"] = time.monotonic() - t0
    print(f"  📝  {out_path}")

    prog.done("render")
    prog.finished(out_path)

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
    dual: bool | None = None,
    progress: ProgressSink | None = None,
    write_json: bool = False,
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
                    output=output_dir, dual=dual, progress=progress, write_json=write_json,
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


def _resolve_mic_label(audio: Path, dual: bool | None) -> str | None:
    """The mic channel's speaker label, or None if this isn't a dual-source run.

    `dual` is a three-state switch: None auto-detects from the recording's
    sidecar, True forces channel-aware diarization on any stereo file, and
    False disables it even for a `shumary record` capture.
    """
    if dual is False:
        return None
    info = recorder.dual_source_info(audio)
    if info:
        return str(info.get("mic_label") or recorder.MIC_LABEL)
    return recorder.MIC_LABEL if dual else None


def _describe_invalidation(
    prev_manifest: dict | None,
    model_name: str,
    fast: bool,
    dual_source: bool,
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
    if bool(prev_manifest.get("dual_source", False)) != bool(dual_source):
        reasons.append(
            f"dual-source diarization changed "
            f"{bool(prev_manifest.get('dual_source', False))} → {dual_source}"
        )
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

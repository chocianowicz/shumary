"""Acoustic prosody extraction (Spec 7, Phase 7.1/7.2).

Runs on the Phase 3 pre-decoded WAV (16 kHz mono) + aligned segments, fully
local. Per segment >= `PROSODY_MIN_SEGMENT_SEC`:

  - Pitch (F0): librosa.pyin in the speech band, median + range in semitones.
  - Energy: librosa RMS, mean dB + dynamic range dB.
  - Voice quality: praat-parselmouth jitter/shimmer (local), wrapped in
    try/except -> None on failure.

Degrades gracefully (like diarization) when `librosa` / `parselmouth` are not
installed: prints a warning and returns `None`. Pipeline continues without a
prosody block.
"""

from pathlib import Path

import numpy as np

from shumary.config import (
    PROSODY_F0_FMAX,
    PROSODY_F0_FMIN,
    PROSODY_FRAME_LENGTH,
    PROSODY_MIN_SEGMENT_SEC,
    PROSODY_MIN_VOICED_FRAMES,
)
from shumary.models import AlignedSegment, ProsodyResult, ProsodySegment, SpeakerProsody


def available() -> bool:
    """Whether both `librosa` and `praat-parselmouth` import cleanly."""
    try:
        import librosa  # noqa: F401
        import parselmouth  # noqa: F401
    except ImportError:
        return False
    return True


def run(wav: Path, segments: list[AlignedSegment]) -> ProsodyResult | None:
    """Extract per-segment prosody features and per-speaker aggregates.

    Returns `None` (with a printed warning) if `librosa` or
    `praat-parselmouth` are not installed.
    """
    try:
        import librosa
    except ImportError as exc:
        print(f"  ⚠  librosa not installed — skipping prosody extraction: {exc}")
        return None
    try:
        import parselmouth  # noqa: F401
    except ImportError as exc:
        print(f"  ⚠  praat-parselmouth not installed — skipping prosody extraction: {exc}")
        return None

    y, sr = librosa.load(str(wav), sr=16000)

    prosody_segments = [
        _extract_segment(librosa, parselmouth, y, sr, wav, seg)
        for seg in segments
        if (seg.end - seg.start) >= PROSODY_MIN_SEGMENT_SEC
    ]

    speaker_aggs = _aggregate_per_speaker(prosody_segments)

    return ProsodyResult(segments=tuple(prosody_segments), speakers=speaker_aggs)


# ── Per-segment extraction ───────────────────────────────────────────────────────

def _extract_segment(librosa, parselmouth, y, sr: int, wav: Path, seg: AlignedSegment) -> ProsodySegment:
    seg_y = y[int(seg.start * sr):int(seg.end * sr)]

    f0_median, f0_range_st = _pitch_features(librosa, seg_y, sr)
    energy_mean_db, energy_range_db = _energy_features(librosa, seg_y)
    jitter_local, shimmer_local = _voice_quality_features(parselmouth, wav, seg)

    return ProsodySegment(
        start=seg.start,
        end=seg.end,
        speaker=seg.speaker,
        f0_median_hz=f0_median,
        f0_range_st=f0_range_st,
        energy_mean_db=energy_mean_db,
        energy_range_db=energy_range_db,
        jitter_local=jitter_local,
        shimmer_local=shimmer_local,
    )


def _pitch_features(librosa, seg_y, sr: int) -> tuple[float | None, float | None]:
    f0, voiced_flag, _ = librosa.pyin(
        seg_y, fmin=PROSODY_F0_FMIN, fmax=PROSODY_F0_FMAX, sr=sr,
        frame_length=PROSODY_FRAME_LENGTH,
    )
    voiced = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced) < PROSODY_MIN_VOICED_FRAMES:
        return None, None

    f0_median = float(np.median(voiced))
    p10, p90 = np.percentile(voiced, 10), np.percentile(voiced, 90)
    f0_range_st = float(12 * np.log2(p90 / p10)) if p10 > 0 else None
    return f0_median, f0_range_st


def _energy_features(librosa, seg_y) -> tuple[float, float]:
    rms = librosa.feature.rms(y=seg_y)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    energy_mean_db = float(rms_db.mean())
    energy_range_db = float(np.percentile(rms_db, 90) - np.percentile(rms_db, 10))
    return energy_mean_db, energy_range_db


def _voice_quality_features(parselmouth, wav: Path, seg: AlignedSegment) -> tuple[float | None, float | None]:
    try:
        snd = parselmouth.Sound(str(wav)).extract_part(seg.start, seg.end)
        pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jitter = parselmouth.praat.call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = parselmouth.praat.call(
            [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6,
        )
        jitter = float(jitter) if jitter == jitter else None  # NaN check
        shimmer = float(shimmer) if shimmer == shimmer else None
        return jitter, shimmer
    except Exception:
        return None, None


# ── Per-speaker aggregation (Spec 7.2) ───────────────────────────────────────────

def _aggregate_per_speaker(prosody_segments: list[ProsodySegment]) -> tuple[SpeakerProsody, ...]:
    """Aggregate voiced-valid segments per speaker (Spec 7.2).

    Speakers with no F0-valid segments are omitted entirely. Single-speaker
    recordings (only one distinct speaker label across all segments — `None`
    or "SPEAKER_00") are labelled "speaker"; multi-speaker recordings keep
    their original labels (SPEAKER_00, SPEAKER_01, ...).
    """
    distinct_speakers = {ps.speaker for ps in prosody_segments}
    single_speaker = len(distinct_speakers) <= 1

    by_speaker: dict[str, list[ProsodySegment]] = {}
    for ps in prosody_segments:
        if ps.f0_median_hz is None:
            continue
        label = "speaker" if single_speaker else (ps.speaker or "speaker")
        by_speaker.setdefault(label, []).append(ps)

    result: list[SpeakerProsody] = []
    for label in sorted(by_speaker):
        segs = by_speaker[label]
        result.append(_aggregate_one_speaker(label, segs))
    return tuple(result)


def _aggregate_one_speaker(label: str, segs: list[ProsodySegment]) -> SpeakerProsody:
    import statistics

    f0_medians = [s.f0_median_hz for s in segs if s.f0_median_hz is not None]
    f0_baseline_hz = statistics.median(f0_medians)

    # f0_variability_st: IQR of per-segment medians expressed in semitones
    # vs baseline (12*log2(x/baseline)).
    semitone_deltas = [float(12 * np.log2(m / f0_baseline_hz)) for m in f0_medians]
    f0_variability_st = _iqr(semitone_deltas)

    energy_means = [s.energy_mean_db for s in segs if s.energy_mean_db is not None]
    energy_ranges = [s.energy_range_db for s in segs if s.energy_range_db is not None]
    energy_mean_db = statistics.median(energy_means)
    energy_dynamics_db = statistics.median(energy_ranges)

    jitters = [s.jitter_local for s in segs if s.jitter_local is not None]
    shimmers = [s.shimmer_local for s in segs if s.shimmer_local is not None]
    jitter_local = statistics.median(jitters) if jitters else None
    shimmer_local = statistics.median(shimmers) if shimmers else None

    f0_trend_by_third_st = _trend_by_third(segs, f0_baseline_hz)

    # float() casts guard YAML serialization: statistics.median/round preserve
    # numpy scalar types, and yaml.safe_dump cannot represent np.float64.
    return SpeakerProsody(
        speaker=label,
        f0_baseline_hz=round(float(f0_baseline_hz), 2),
        f0_variability_st=round(float(f0_variability_st), 2),
        energy_mean_db=round(float(energy_mean_db), 2),
        energy_dynamics_db=round(float(energy_dynamics_db), 2),
        jitter_local=round(float(jitter_local), 4) if jitter_local is not None else None,
        shimmer_local=round(float(shimmer_local), 4) if shimmer_local is not None else None,
        f0_trend_by_third_st=f0_trend_by_third_st,
    )


def _iqr(values: list[float]) -> float:
    if len(values) == 1:
        return 0.0
    ordered = sorted(values)
    q1 = _percentile(ordered, 25)
    q3 = _percentile(ordered, 75)
    return q3 - q1


def _percentile(ordered: list[float], pct: float) -> float:
    """Nearest-rank-interpolated percentile on a pre-sorted list."""
    if len(ordered) == 1:
        return ordered[0]
    k = (pct / 100) * (len(ordered) - 1)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def _trend_by_third(segs: list[ProsodySegment], baseline_hz: float) -> tuple[float, float, float]:
    """Per session third, median segment-F0 delta vs baseline in semitones."""
    import statistics

    span_start = min(s.start for s in segs)
    span_end = max(s.end for s in segs)
    span = span_end - span_start
    if span <= 0:
        delta = float(12 * np.log2(segs[0].f0_median_hz / baseline_hz))
        return (round(delta, 2),) * 3

    third = span / 3
    bounds = [
        (span_start, span_start + third),
        (span_start + third, span_start + 2 * third),
        (span_start + 2 * third, span_end),
    ]

    deltas_by_third: list[list[float]] = [[], [], []]
    for s in segs:
        mid = (s.start + s.end) / 2
        for i, (lo, hi) in enumerate(bounds):
            if lo <= mid <= hi:
                deltas_by_third[i].append(float(12 * np.log2(s.f0_median_hz / baseline_hz)))
                break

    trend = []
    for deltas in deltas_by_third:
        trend.append(round(statistics.median(deltas), 2) if deltas else 0.0)
    return (trend[0], trend[1], trend[2])


# ── Summary print ────────────────────────────────────────────────────────────────

def summarize(result: ProsodyResult) -> str:
    """One-line per-speaker F0 baseline summary for the pipeline log."""
    if not result.speakers:
        return "no voiced segments — prosody unavailable"
    parts = [f"{sp.speaker}: F0 ~{sp.f0_baseline_hz:.0f}Hz" for sp in result.speakers]
    return "  |  ".join(parts)

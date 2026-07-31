"""Behavioral / mood signal extraction from aligned segments."""

import re
import statistics
from collections import Counter
from functools import lru_cache

from shumary.config import (
    DISCOURSE_MARKER_HIGH,
    HESITATION_HIGH,
    LONG_PAUSE_SEC,
    LOW_CONF_THRESHOLD,
    HIGH_SILENCE_PROB,
    SLOW_RATE,
    FAST_RATE,
    BASELINE_RATE,
    HESITATIONS,
    DISCOURSE_MARKERS,
)
from shumary.models import AlignedSegment, MoodSignals, PauseDetail, SpeakerMood
from shumary.utils import fmt_time

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of", "is", "it",
})


def run(
    segments: list[AlignedSegment],
    language: str,
    speakers: set[str] | None = None,
) -> MoodSignals:
    """Global and per-speaker mood signals.

    `speakers`, when given, restricts the *per-speaker* breakdown to labels that
    qualified as participants. Global figures still cover all transcribed text —
    a disqualified label's words are in the transcript either way, so excluding
    them from the totals would make the totals disagree with what you can read.
    """
    text_lower = " ".join(s.text for s in segments).lower()

    pauses, pause_details = _detect_pauses(segments)
    rate, variability = _speaking_rate(segments)
    hesitations, markers, top_fillers = _count_fillers_split(text_lower)
    repeated = _detect_repetitions(text_lower)
    low_conf = _low_confidence_segments(segments)
    near_sil = _near_silence_segments(segments)
    state = _infer_state(
        rate, variability, pauses, hesitations, markers, len(repeated), low_conf, near_sil
    )

    per_speaker = _per_speaker_moods(segments, speakers)

    return MoodSignals(
        avg_speaking_rate_wps=rate,
        rate_variability=variability,
        long_pauses=pauses,
        long_pause_details=tuple(pause_details[:5]),
        filler_word_count=hesitations + markers,
        hesitations=hesitations,
        discourse_markers=markers,
        top_fillers=tuple(top_fillers[:5]),
        repeated_phrases=tuple(repeated[:10]),
        low_confidence_segments=low_conf,
        near_silence_segments=near_sil,
        inferred_state=state,
        per_speaker=per_speaker,
    )


# ── Per-speaker signals (Spec 5 / 5.1) ──────────────────────────────────────────

def _per_speaker_moods(
    segments: list[AlignedSegment], accepted: set[str] | None = None
) -> tuple[SpeakerMood, ...]:
    """Compute per-speaker mood signals.

    Returns an empty tuple when speaker labels are absent (e.g. diarization
    skipped) or fewer than two distinct speakers are present — single-speaker
    recordings get their signal from the global fields only.
    """
    speakers = {s.speaker for s in segments if s.speaker is not None}
    if accepted is not None:
        speakers &= accepted
    if len(speakers) < 2:
        return ()

    result: list[SpeakerMood] = []
    for speaker in sorted(speakers):
        speaker_segs = [s for s in segments if s.speaker == speaker]
        text_lower = " ".join(s.text for s in speaker_segs).lower()

        rate, variability = _duration_weighted_rate(speaker_segs)
        intra_pauses, intra_details = _detect_intra_pauses(segments, speaker)
        hesitation_count, marker_count, top_fillers = _count_fillers_split(text_lower)
        repeated = _detect_repetitions(text_lower)
        low_conf = _low_confidence_segments(speaker_segs)
        state = _infer_speaker_state(
            rate, variability, intra_pauses, hesitation_count, marker_count,
            len(repeated), low_conf,
        )

        result.append(SpeakerMood(
            speaker=speaker,
            speaking_rate_wps=rate,
            rate_variability=variability,
            intra_pauses=intra_pauses,
            intra_pause_details=tuple(intra_details[:5]),
            hesitations=hesitation_count,
            discourse_markers=marker_count,
            top_fillers=tuple(top_fillers[:5]),
            repeated_phrases=tuple(repeated[:10]),
            low_confidence_segments=low_conf,
            inferred_state=state,
        ))

    return tuple(result)


def _duration_weighted_rate(
    segments: list[AlignedSegment],
) -> tuple[float | None, float | None]:
    """Speaking rate = total words / total speech seconds (NOT mean of
    per-segment rates). Variability is the stdev of per-segment rates,
    kept as a secondary descriptive signal."""
    total_words = 0
    total_dur = 0.0
    rates = []
    for s in segments:
        dur = s.end - s.start
        words = len(s.text.split())
        if dur > 0.1:
            total_words += words
            total_dur += dur
            rates.append(words / dur)

    if total_dur <= 0:
        return None, None

    rate = round(total_words / total_dur, 2)
    var = round(statistics.stdev(rates), 2) if len(rates) > 2 else None
    return rate, var


def _detect_intra_pauses(
    segments: list[AlignedSegment], speaker: str,
) -> tuple[int, list[PauseDetail]]:
    """Gaps >= LONG_PAUSE_SEC between consecutive segments of `speaker` with
    no other speaker's segment between them (i.e. consecutive in the overall
    timeline, both attributed to `speaker`)."""
    count = 0
    details: list[PauseDetail] = []
    for i in range(1, len(segments)):
        prev, cur = segments[i - 1], segments[i]
        if prev.speaker != speaker or cur.speaker != speaker:
            continue
        gap = cur.start - prev.end
        if gap >= LONG_PAUSE_SEC:
            count += 1
            details.append(PauseDetail(at=fmt_time(prev.end), duration_sec=round(gap, 1)))
    return count, details


def _infer_speaker_state(
    rate: float | None,
    variability: float | None,
    intra_pauses: int,
    hesitations: int,
    discourse_markers: int,
    repetitions: int,
    low_conf: int,
) -> str:
    signals: list[str] = []

    if rate is not None:
        if rate < SLOW_RATE:
            signals.append("slow delivery — fatigue or deliberation")
        elif rate > FAST_RATE:
            signals.append("fast delivery — excitement or anxiety")

    if intra_pauses >= 4:
        signals.append("frequent long pauses")

    if hesitations >= HESITATION_HIGH:
        signals.append("frequent hesitations — cognitive load or discomfort")
    if discourse_markers >= DISCOURSE_MARKER_HIGH:
        signals.append("frequent discourse markers — conversational style, not distress")

    if repetitions >= 5:
        signals.append("repeated phrases — thought loops or emphasis")

    if low_conf >= 3:
        signals.append("multiple low-confidence segments — mumbling or trailing off")

    if variability is not None and variability > 1.2:
        signals.append("shifting speaking pace — varying emotional states")

    if not signals:
        return "(uncertain) delivery near baseline — no strong signals"

    return "(uncertain) " + "; ".join(signals)


# ── Signal extractors ──────────────────────────────────────────────────────────

def _detect_pauses(
    segments: list[AlignedSegment],
) -> tuple[int, list[PauseDetail]]:
    count = 0
    details: list[PauseDetail] = []
    for i in range(1, len(segments)):
        gap = segments[i].start - segments[i - 1].end
        if gap >= LONG_PAUSE_SEC:
            count += 1
            details.append(PauseDetail(at=fmt_time(segments[i - 1].end), duration_sec=round(gap, 1)))
    return count, details


def _speaking_rate(
    segments: list[AlignedSegment],
) -> tuple[float | None, float | None]:
    rates = []
    for s in segments:
        dur = s.end - s.start
        if dur > 0.1:
            words = len(s.text.split())
            rates.append(words / dur)
    if not rates:
        return None, None
    avg = round(statistics.mean(rates), 2)
    var = round(statistics.stdev(rates), 2) if len(rates) > 2 else None
    return avg, var


def _compile_alternation(fillers: set[str]) -> re.Pattern:
    """Compile one alternation regex, sorted by length descending so
    multi-word fillers are tried before their single-word prefixes (regex
    alternation matches the first alternative that succeeds, not the
    longest — e.g. without this, `\\b(no|no to)\\b` would match "no" inside
    "no to" and "no to" would be unreachable)."""
    escaped = sorted((re.escape(f) for f in fillers), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


@lru_cache(maxsize=1)
def _hesitation_pattern() -> re.Pattern:
    fillers: set[str] = set()
    for lexicon in HESITATIONS.values():
        fillers |= lexicon
    return _compile_alternation(fillers)


@lru_cache(maxsize=1)
def _discourse_marker_pattern() -> re.Pattern:
    fillers: set[str] = set()
    for lexicon in DISCOURSE_MARKERS.values():
        fillers |= lexicon
    return _compile_alternation(fillers)


def _count_fillers_split(text_lower: str) -> tuple[int, int, list[str]]:
    """Return (hesitation_count, discourse_marker_count, top_fillers) where
    `top_fillers` ranks across both lexicons combined."""
    hes_counter: Counter[str] = Counter(_hesitation_pattern().findall(text_lower))
    marker_counter: Counter[str] = Counter(_discourse_marker_pattern().findall(text_lower))
    combined = hes_counter + marker_counter
    top = [f"{word} ×{count}" for word, count in combined.most_common(5)]
    return sum(hes_counter.values()), sum(marker_counter.values()), top


def _detect_repetitions(text_lower: str) -> list[str]:
    words = text_lower.split()
    trigrams: Counter[str] = Counter()
    for i in range(len(words) - 2):
        tri = " ".join(words[i : i + 3])
        if not _is_trivial(tri):
            trigrams[tri] += 1
    return [tri for tri, count in trigrams.most_common(10) if count > 1]


def _is_trivial(trigram: str) -> bool:
    return all(w in _STOPWORDS for w in trigram.split())


def _low_confidence_segments(segments: list[AlignedSegment]) -> int:
    return sum(1 for s in segments if s.avg_logprob <= LOW_CONF_THRESHOLD)


def _near_silence_segments(segments: list[AlignedSegment]) -> int:
    return sum(1 for s in segments if s.no_speech_prob >= HIGH_SILENCE_PROB)


# ── State inference ────────────────────────────────────────────────────────────

def _infer_state(
    rate: float | None,
    variability: float | None,
    pauses: int,
    hesitations: int,
    discourse_markers: int,
    repetitions: int,
    low_conf: int,
    near_sil: int,
) -> str:
    signals: list[str] = []

    if rate is not None:
        if rate < SLOW_RATE:
            signals.append("slow delivery — fatigue or deliberation")
        elif rate > FAST_RATE:
            signals.append("fast delivery — excitement or anxiety")

    if pauses >= 4:
        signals.append("frequent long pauses")

    if hesitations >= HESITATION_HIGH:
        signals.append("frequent hesitations — cognitive load or discomfort")
    if discourse_markers >= DISCOURSE_MARKER_HIGH:
        signals.append("frequent discourse markers — conversational style, not distress")

    if repetitions >= 5:
        signals.append("repeated phrases — thought loops or emphasis")

    if low_conf >= 3:
        signals.append("multiple low-confidence segments — mumbling or trailing off")

    if near_sil >= 3:
        signals.append("near-silence segments — sighs or heavy pauses mid-sentence")

    if variability is not None and variability > 1.2:
        signals.append("shifting speaking pace — varying emotional states")

    if not signals:
        return "(uncertain) delivery near baseline — no strong signals"

    return "(uncertain) " + "; ".join(signals)

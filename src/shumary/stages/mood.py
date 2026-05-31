"""Behavioral / mood signal extraction from aligned segments."""

import re
import statistics
from collections import Counter
from functools import lru_cache

from shumary.config import (
    LONG_PAUSE_SEC,
    LOW_CONF_THRESHOLD,
    HIGH_SILENCE_PROB,
    SLOW_RATE,
    FAST_RATE,
    BASELINE_RATE,
    FILLER_WORDS,
)
from shumary.models import AlignedSegment, MoodSignals, PauseDetail
from shumary.utils import fmt_time

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of", "is", "it",
})


def run(segments: list[AlignedSegment], language: str) -> MoodSignals:
    text_lower = " ".join(s.text for s in segments).lower()

    pauses, pause_details = _detect_pauses(segments)
    rate, variability = _speaking_rate(segments)
    filler_count, top_fillers = _count_fillers(text_lower, language)
    repeated = _detect_repetitions(text_lower)
    low_conf = _low_confidence_segments(segments)
    near_sil = _near_silence_segments(segments)
    state = _infer_state(rate, variability, pauses, filler_count, len(repeated), low_conf, near_sil)

    return MoodSignals(
        avg_speaking_rate_wps=rate,
        rate_variability=variability,
        long_pauses=pauses,
        long_pause_details=tuple(pause_details[:5]),
        filler_word_count=filler_count,
        top_fillers=tuple(top_fillers[:5]),
        repeated_phrases=tuple(repeated[:10]),
        low_confidence_segments=low_conf,
        near_silence_segments=near_sil,
        inferred_state=state,
    )


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


@lru_cache(maxsize=16)
def _filler_pattern(language: str) -> re.Pattern:
    """Compile one alternation regex for all fillers in English + given language."""
    fillers: set[str] = FILLER_WORDS.get("en", set()).copy()
    fillers |= FILLER_WORDS.get(language, set())
    escaped = sorted(re.escape(f) for f in fillers)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


def _count_fillers(text_lower: str, language: str) -> tuple[int, list[str]]:
    pattern = _filler_pattern(language)
    counter: Counter[str] = Counter(pattern.findall(text_lower))
    top = [f"{word} ×{count}" for word, count in counter.most_common(5)]
    return sum(counter.values()), top


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
    fillers: int,
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

    if fillers >= 15:
        signals.append("high filler word count — cognitive load or discomfort")

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

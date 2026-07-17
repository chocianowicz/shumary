"""Interaction / turn-taking metric extraction (Spec 5, Phase 5.2).

Operates on raw diarization `Turn`s. Preprocessing merges consecutive
same-speaker turns separated by less than `UTTERANCE_MERGE_GAP` into one
*utterance*. All metrics operate on utterances **except** overlap detection,
which uses the raw turns (overlaps cannot survive utterance merging since
merging only joins same-speaker turns).

Only meaningful with 2+ speakers — callers should gate on speaker count.
"""

import statistics
from collections import defaultdict

from shumary.config import (
    INTERRUPT_YIELD_SEC,
    LATENCY_CAP_SEC,
    OVERLAP_MIN,
    UTTERANCE_MERGE_GAP,
)
from shumary.models import InteractionMetrics, SpeakerInteraction, Turn


def run(turns: list[Turn], backend: str) -> InteractionMetrics:
    sorted_turns = sorted(turns, key=lambda t: t.start)
    utterances = _merge_utterances(sorted_turns)
    speakers = sorted({t.speaker for t in sorted_turns})

    turn_counts, avg_turn_secs = _turn_stats(utterances, speakers)
    median_latency, p90_latency, long_gaps = _response_latencies(utterances, speakers)
    overlap_count, overlap_total_sec, interruptions_made, interruptions_received = _overlaps_and_interruptions(
        sorted_turns, backend, speakers,
    )
    talk_share = _talk_share_by_third(utterances, speakers)

    speaker_metrics = tuple(
        SpeakerInteraction(
            speaker=sp,
            turn_count=turn_counts[sp],
            avg_turn_sec=avg_turn_secs[sp],
            median_response_latency_sec=median_latency.get(sp),
            p90_response_latency_sec=p90_latency.get(sp),
            interruptions_made=interruptions_made.get(sp, 0),
            interruptions_received=interruptions_received.get(sp, 0),
            talk_share_by_third=talk_share[sp],
        )
        for sp in speakers
    )

    return InteractionMetrics(
        backend=backend,
        overlap_count=overlap_count,
        overlap_total_sec=overlap_total_sec,
        long_gaps=long_gaps,
        speakers=speaker_metrics,
    )


# ── Utterance merging ────────────────────────────────────────────────────────────

def _merge_utterances(sorted_turns: list[Turn]) -> list[Turn]:
    """Merge consecutive turns of the SAME speaker separated by less than
    UTTERANCE_MERGE_GAP into one utterance."""
    if not sorted_turns:
        return []

    merged: list[Turn] = [sorted_turns[0]]
    for turn in sorted_turns[1:]:
        last = merged[-1]
        if turn.speaker == last.speaker and (turn.start - last.end) < UTTERANCE_MERGE_GAP:
            merged[-1] = Turn(start=last.start, end=max(last.end, turn.end), speaker=last.speaker)
        else:
            merged.append(turn)
    return merged


# ── Turn counts / lengths ──────────────────────────────────────────────────────────

def _turn_stats(
    utterances: list[Turn], speakers: list[str],
) -> tuple[dict[str, int], dict[str, float]]:
    durations: dict[str, list[float]] = defaultdict(list)
    for u in utterances:
        durations[u.speaker].append(u.end - u.start)

    counts = {sp: len(durations.get(sp, [])) for sp in speakers}
    avgs = {
        sp: round(statistics.mean(durations[sp]), 2) if durations.get(sp) else 0.0
        for sp in speakers
    }
    return counts, avgs


# ── Response latency ──────────────────────────────────────────────────────────────

def _response_latencies(
    utterances: list[Turn], speakers: list[str],
) -> tuple[dict[str, float], dict[str, float], int]:
    """For each consecutive utterance pair (A then B, different speakers,
    B.start >= A.end): latency = B.start - A.end. Latencies > LATENCY_CAP_SEC
    are excluded and counted as long_gaps. Returns (median_by_responder,
    p90_by_responder, long_gaps)."""
    latencies: dict[str, list[float]] = defaultdict(list)
    long_gaps = 0

    for prev, cur in zip(utterances, utterances[1:]):
        if cur.speaker == prev.speaker:
            continue
        latency = cur.start - prev.end
        if latency < 0:
            continue
        if latency > LATENCY_CAP_SEC:
            long_gaps += 1
            continue
        latencies[cur.speaker].append(latency)

    median: dict[str, float] = {}
    p90: dict[str, float] = {}
    for sp in speakers:
        vals = latencies.get(sp)
        if not vals:
            continue
        median[sp] = round(statistics.median(vals), 2)
        p90[sp] = round(_percentile(vals, 90), 2)

    return median, p90, long_gaps


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile — no extra dependency for a single-use stat."""
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    k = (pct / 100) * (len(ordered) - 1)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


# ── Overlap + interruptions (raw turns) ─────────────────────────────────────────────

def _overlaps_and_interruptions(
    sorted_turns: list[Turn], backend: str, speakers: list[str],
) -> tuple[int | None, float | None, dict[str, int], dict[str, int]]:
    interruptions_made: dict[str, int] = defaultdict(int)
    interruptions_received: dict[str, int] = defaultdict(int)

    if backend != "pyannote":
        # Resemblyzer turns are produced by partitioning a single timeline —
        # they cannot overlap by construction.
        return None, None, dict(interruptions_made), dict(interruptions_received)

    overlap_count = 0
    overlap_total_sec = 0.0

    n = len(sorted_turns)
    for i in range(n):
        a = sorted_turns[i]
        for j in range(i + 1, n):
            b = sorted_turns[j]
            if b.start >= a.end:
                break  # sorted by start — no further turns can overlap `a`
            if b.speaker == a.speaker:
                continue

            ov = min(a.end, b.end) - max(a.start, b.start)
            if ov <= OVERLAP_MIN:
                continue

            overlap_count += 1
            overlap_total_sec += ov

            # Interruption: B starts strictly inside A's turn AND A's turn
            # ends within INTERRUPT_YIELD_SEC after B.start (A yields).
            b_starts_inside_a = a.start < b.start < a.end
            a_yields = (a.end - b.start) <= INTERRUPT_YIELD_SEC

            if b_starts_inside_a and a_yields:
                interruptions_made[b.speaker] += 1
                interruptions_received[a.speaker] += 1
            # else: back-channel (A keeps talking >= INTERRUPT_YIELD_SEC past
            # B's start and B's turn < UTTERANCE_MERGE_GAP) or other overlap —
            # counted in overlap_count but not as an interruption.

    overlap_total_sec = round(overlap_total_sec, 2)
    return overlap_count, overlap_total_sec, dict(interruptions_made), dict(interruptions_received)


# ── Talk share by third ──────────────────────────────────────────────────────────

def _talk_share_by_third(
    utterances: list[Turn], speakers: list[str],
) -> dict[str, tuple[float, float, float]]:
    if not utterances:
        return {sp: (0.0, 0.0, 0.0) for sp in speakers}

    rec_start = min(u.start for u in utterances)
    rec_end = max(u.end for u in utterances)
    span = rec_end - rec_start
    if span <= 0:
        return {sp: (0.0, 0.0, 0.0) for sp in speakers}

    third = span / 3
    bounds = [
        (rec_start, rec_start + third),
        (rec_start + third, rec_start + 2 * third),
        (rec_start + 2 * third, rec_end),
    ]

    speech_per_window: list[dict[str, float]] = [defaultdict(float) for _ in range(3)]

    for u in utterances:
        for w, (w_start, w_end) in enumerate(bounds):
            ov = min(u.end, w_end) - max(u.start, w_start)
            if ov > 0:
                speech_per_window[w][u.speaker] += ov

    shares: dict[str, list[float]] = {sp: [0.0, 0.0, 0.0] for sp in speakers}
    for w in range(3):
        total = sum(speech_per_window[w].values())
        if total <= 0:
            continue
        for sp in speakers:
            shares[sp][w] = round(speech_per_window[w].get(sp, 0.0) / total, 3)

    return {sp: (vals[0], vals[1], vals[2]) for sp, vals in shares.items()}

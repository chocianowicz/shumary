"""Tests for stages/interaction.py — turn-taking / interaction metrics (Spec 5)."""

from shumary.config import LATENCY_CAP_SEC, UTTERANCE_MERGE_GAP
from shumary.models import InteractionMetrics, SpeakerInteraction, Turn
from shumary.stages import interaction


# ── Utterance merging ────────────────────────────────────────────────────────────

def test_consecutive_same_speaker_turns_merge_into_one_utterance():
    turns = [
        Turn(0.0, 2.0, "SPEAKER_00"),
        Turn(2.5, 4.0, "SPEAKER_00"),  # 0.5s gap < UTTERANCE_MERGE_GAP (1.0)
        Turn(4.0, 6.0, "SPEAKER_01"),
    ]
    merged = interaction._merge_utterances(turns)
    assert len(merged) == 2
    assert merged[0] == Turn(0.0, 4.0, "SPEAKER_00")
    assert merged[1] == Turn(4.0, 6.0, "SPEAKER_01")


def test_same_speaker_turns_with_large_gap_do_not_merge():
    turns = [
        Turn(0.0, 2.0, "SPEAKER_00"),
        Turn(3.5, 5.0, "SPEAKER_00"),  # 1.5s gap >= UTTERANCE_MERGE_GAP
    ]
    merged = interaction._merge_utterances(turns)
    assert len(merged) == 2


def test_merge_gap_boundary_is_exclusive():
    """A gap exactly == UTTERANCE_MERGE_GAP must NOT merge (< not <=)."""
    turns = [
        Turn(0.0, 2.0, "SPEAKER_00"),
        Turn(2.0 + UTTERANCE_MERGE_GAP, 4.0, "SPEAKER_00"),
    ]
    merged = interaction._merge_utterances(turns)
    assert len(merged) == 2


# ── Turn counts / lengths ──────────────────────────────────────────────────────────

def test_turn_count_and_avg_turn_sec_per_speaker():
    turns = [
        Turn(0.0, 2.0, "SPEAKER_00"),   # utterance, 2s
        Turn(2.0, 4.0, "SPEAKER_01"),   # utterance, 2s
        Turn(4.0, 10.0, "SPEAKER_00"),  # utterance, 6s
    ]
    m = interaction.run(turns, "pyannote")
    by_speaker = {s.speaker: s for s in m.speakers}

    assert by_speaker["SPEAKER_00"].turn_count == 2
    assert by_speaker["SPEAKER_00"].avg_turn_sec == 4.0  # (2 + 6) / 2
    assert by_speaker["SPEAKER_01"].turn_count == 1
    assert by_speaker["SPEAKER_01"].avg_turn_sec == 2.0


# ── Response latency ──────────────────────────────────────────────────────────────

def test_response_latency_median_and_p90_per_responder():
    turns = [
        Turn(0.0, 5.0, "SPEAKER_00"),
        Turn(6.0, 8.0, "SPEAKER_01"),    # latency 1.0 for SPEAKER_01
        Turn(8.5, 12.0, "SPEAKER_00"),   # latency 0.5 for SPEAKER_00
        Turn(13.0, 14.0, "SPEAKER_01"),  # latency 1.0 for SPEAKER_01
    ]
    m = interaction.run(turns, "pyannote")
    by_speaker = {s.speaker: s for s in m.speakers}

    assert by_speaker["SPEAKER_00"].median_response_latency_sec == 0.5
    assert by_speaker["SPEAKER_01"].median_response_latency_sec == 1.0


def test_latencies_above_cap_excluded_and_counted_as_long_gaps():
    turns = [
        Turn(0.0, 5.0, "SPEAKER_00"),
        Turn(5.0 + LATENCY_CAP_SEC + 1, 8.0, "SPEAKER_01"),  # > cap -> long_gap
    ]
    m = interaction.run(turns, "pyannote")
    by_speaker = {s.speaker: s for s in m.speakers}

    assert m.long_gaps == 1
    assert by_speaker["SPEAKER_01"].median_response_latency_sec is None
    assert by_speaker["SPEAKER_01"].p90_response_latency_sec is None


def test_speaker_who_never_responds_has_none_latency():
    """A speaker who only ever speaks first (never as a responder) gets
    None for both latency stats."""
    turns = [
        Turn(0.0, 5.0, "SPEAKER_00"),
        Turn(5.0, 10.0, "SPEAKER_01"),
    ]
    m = interaction.run(turns, "pyannote")
    by_speaker = {s.speaker: s for s in m.speakers}

    # SPEAKER_00 never responds to anyone (always speaks first)
    assert by_speaker["SPEAKER_00"].median_response_latency_sec is None
    assert by_speaker["SPEAKER_00"].p90_response_latency_sec is None
    # SPEAKER_01 responds once with latency 0
    assert by_speaker["SPEAKER_01"].median_response_latency_sec == 0.0


# ── Overlap (pyannote only) ──────────────────────────────────────────────────────

def test_overlap_detected_for_pyannote_backend():
    turns = [
        Turn(0.0, 10.0, "SPEAKER_00"),
        Turn(9.0, 15.0, "SPEAKER_01"),  # 1s overlap with SPEAKER_00
    ]
    m = interaction.run(turns, "pyannote")
    assert m.overlap_count == 1
    assert m.overlap_total_sec == 1.0


def test_no_overlap_below_overlap_min_not_counted():
    turns = [
        Turn(0.0, 10.0, "SPEAKER_00"),
        Turn(10.05, 15.0, "SPEAKER_01"),  # 0.05s overlap < OVERLAP_MIN (0.1)
    ]
    m = interaction.run(turns, "pyannote")
    assert m.overlap_count == 0
    assert m.overlap_total_sec == 0.0


def test_resemblyzer_backend_overlap_fields_are_none():
    turns = [
        Turn(0.0, 5.0, "SPEAKER_00"),
        Turn(5.0, 10.0, "SPEAKER_01"),
    ]
    m = interaction.run(turns, "resemblyzer")
    assert m.overlap_count is None
    assert m.overlap_total_sec is None
    assert m.backend == "resemblyzer"


# ── Interruption vs back-channel discrimination ───────────────────────────────────

def test_interruption_when_b_starts_inside_a_and_a_yields_quickly():
    """B starts inside A's turn, and A's turn ends within INTERRUPT_YIELD_SEC
    (2.0s) after B starts — A yields the floor: counted as an interruption."""
    turns = [
        Turn(0.0, 10.0, "SPEAKER_00"),   # A
        Turn(9.0, 15.0, "SPEAKER_01"),   # B starts at 9 (inside A), A ends at 10 (1s later <= 2.0)
    ]
    m = interaction.run(turns, "pyannote")
    by_speaker = {s.speaker: s for s in m.speakers}

    assert by_speaker["SPEAKER_01"].interruptions_made == 1
    assert by_speaker["SPEAKER_00"].interruptions_received == 1
    assert m.overlap_count == 1


def test_back_channel_not_counted_as_interruption():
    """A talks 0-10, B says a short "mhm" at 5.0-5.5 (< 1.0s, well before A
    finishes — A keeps talking >= INTERRUPT_YIELD_SEC past B's start).
    Counted in overlap_count but NOT as an interruption."""
    turns = [
        Turn(0.0, 10.0, "SPEAKER_00"),  # A
        Turn(5.0, 5.5, "SPEAKER_01"),   # B's brief back-channel
    ]
    m = interaction.run(turns, "pyannote")
    by_speaker = {s.speaker: s for s in m.speakers}

    assert m.overlap_count == 1
    assert by_speaker["SPEAKER_01"].interruptions_made == 0
    assert by_speaker["SPEAKER_00"].interruptions_received == 0


def test_long_overlap_where_a_does_not_yield_quickly_is_not_an_interruption():
    """B starts inside A's turn, but A keeps talking well beyond
    INTERRUPT_YIELD_SEC after B starts (A doesn't yield), and B's turn is
    long enough that it's not a back-channel either — counted as an overlap
    but not an interruption."""
    turns = [
        Turn(0.0, 20.0, "SPEAKER_00"),  # A — long turn
        Turn(5.0, 8.0, "SPEAKER_01"),   # B starts inside A; A ends 15s later (> 2.0); B's turn is 3s (>= 1.0)
    ]
    m = interaction.run(turns, "pyannote")
    by_speaker = {s.speaker: s for s in m.speakers}

    assert m.overlap_count == 1
    assert by_speaker["SPEAKER_01"].interruptions_made == 0
    assert by_speaker["SPEAKER_00"].interruptions_received == 0


# ── Talk share by third ────────────────────────────────────────────────────────────

def test_talk_share_by_third_reflects_speech_seconds_not_wall_clock():
    """SPEAKER_00 talks continuously for the first two-thirds; SPEAKER_01
    talks only in the last third. Shares are computed from speech seconds
    within each window, not wall-clock fractions."""
    turns = [
        Turn(0.0, 6.0, "SPEAKER_00"),
        Turn(6.0, 9.0, "SPEAKER_01"),
    ]
    m = interaction.run(turns, "pyannote")
    by_speaker = {s.speaker: s for s in m.speakers}

    # Recording spans 0-9, thirds are [0,3), [3,6), [6,9)
    s00 = by_speaker["SPEAKER_00"].talk_share_by_third
    s01 = by_speaker["SPEAKER_01"].talk_share_by_third

    assert s00[0] == 1.0  # SPEAKER_00 speaks the entire first third
    assert s00[1] == 1.0  # and the entire second third
    assert s00[2] == 0.0  # none of the third third
    assert s01[2] == 1.0  # SPEAKER_01 speaks the entire third third


def test_talk_share_thirds_sum_to_one_when_speech_present():
    turns = [
        Turn(0.0, 3.0, "SPEAKER_00"),
        Turn(3.0, 6.0, "SPEAKER_01"),
        Turn(6.0, 9.0, "SPEAKER_00"),
    ]
    m = interaction.run(turns, "pyannote")
    shares = {s.speaker: s.talk_share_by_third for s in m.speakers}

    for w in range(3):
        total = sum(shares[sp][w] for sp in shares)
        assert abs(total - 1.0) < 1e-6


# ── Serialisation ───────────────────────────────────────────────────────────────

def test_interaction_metrics_roundtrip():
    turns = [
        Turn(0.0, 5.0, "SPEAKER_00"),
        Turn(5.0, 10.0, "SPEAKER_01"),
    ]
    m = interaction.run(turns, "pyannote")
    d = m.to_dict()
    restored = InteractionMetrics.from_dict(d)
    assert restored == m


def test_speaker_interaction_roundtrip_with_none_latencies():
    si = SpeakerInteraction(
        speaker="SPEAKER_00",
        turn_count=1,
        avg_turn_sec=5.0,
        median_response_latency_sec=None,
        p90_response_latency_sec=None,
        interruptions_made=0,
        interruptions_received=0,
        talk_share_by_third=(1.0, 0.0, 0.0),
    )
    d = si.to_dict()
    restored = SpeakerInteraction.from_dict(d)
    assert restored == si

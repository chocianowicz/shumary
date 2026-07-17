"""Tests for stages/align.py — speaker-boundary segment splitting."""

from shumary.models import Segment, Turn, Word
from shumary.stages import align


def test_no_turns_returns_none_speakers():
    segs = (Segment(0.0, 5.0, "hello", -0.2, 0.1),)
    result = align.run(segs, turns=None)
    assert len(result) == 1
    assert result[0].speaker is None


def test_single_turn_assigns_speaker():
    segs = (Segment(0.0, 5.0, "hello world", -0.2, 0.1),)
    turns = [Turn(0.0, 5.0, "SPEAKER_00")]
    result = align.run(segs, turns)
    assert len(result) == 1
    assert result[0].speaker == "SPEAKER_00"


def test_single_turn_covers_all_segments(simple_segments, two_speaker_turns):
    _ = two_speaker_turns  # not used here
    turns = [Turn(0.0, 15.0, "SPEAKER_00")]
    result = align.run(simple_segments, turns)
    assert all(r.speaker == "SPEAKER_00" for r in result)


def test_two_speakers_split_segment():
    """A segment that spans a speaker boundary should be split."""
    segs = (Segment(0.0, 10.0, "speaker zero then speaker one", -0.2, 0.1),)
    turns = [Turn(0.0, 5.0, "SPEAKER_00"), Turn(5.0, 10.0, "SPEAKER_01")]
    result = align.run(segs, turns)
    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[1].speaker == "SPEAKER_01"


def test_no_overlapping_turn_gets_unknown_speaker():
    segs = (Segment(20.0, 25.0, "orphan segment", -0.2, 0.1),)
    turns = [Turn(0.0, 10.0, "SPEAKER_00")]
    result = align.run(segs, turns)
    assert len(result) == 1
    assert result[0].speaker == "SPEAKER_??"


def test_timing_preserved_across_split():
    segs = (Segment(2.0, 8.0, "a b c d e f", -0.2, 0.1),)
    turns = [Turn(2.0, 5.0, "SPEAKER_00"), Turn(5.0, 8.0, "SPEAKER_01")]
    result = align.run(segs, turns)
    assert result[0].start == 2.0
    assert result[-1].end == 8.0


def test_empty_segments():
    assert align.run((), turns=None) == []
    assert align.run((), turns=[Turn(0.0, 1.0, "SPEAKER_00")]) == []


# ── Spec 4.1 — word-level alignment ────────────────────────────────────────────

def _word_segment() -> Segment:
    """Segment 0.0–10.0s, ten words at 1s intervals: w0..w9."""
    words = tuple(
        Word(start=float(i), end=float(i) + 1.0, text=f"w{i}", prob=1.0)
        for i in range(10)
    )
    return Segment(start=0.0, end=10.0, text=" ".join(w.text for w in words),
                    avg_logprob=-0.2, no_speech_prob=0.1, words=words)


def test_golden_word_level_split_at_turn_boundary():
    """Spec 4.1 golden test: segment 0-10s, words at 1s intervals, turn boundary
    at 5.0s -> exactly two pieces, words 0-4 / 5-9, no text lost."""
    seg = _word_segment()
    turns = [Turn(0.0, 5.0, "SPEAKER_00"), Turn(5.0, 10.0, "SPEAKER_01")]

    result = align.run((seg,), turns)

    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[1].speaker == "SPEAKER_01"
    assert result[0].text == "w0 w1 w2 w3 w4"
    assert result[1].text == "w5 w6 w7 w8 w9"

    # No text lost — concatenation of pieces equals concatenation of words.
    assert result[0].text + " " + result[1].text == seg.text

    # Times come from actual word times.
    assert result[0].start == 0.0
    assert result[0].end == 5.0
    assert result[1].start == 5.0
    assert result[1].end == 10.0


def test_word_level_avg_logprob_and_no_speech_prob_copied_from_parent():
    seg = _word_segment()
    turns = [Turn(0.0, 5.0, "SPEAKER_00"), Turn(5.0, 10.0, "SPEAKER_01")]
    result = align.run((seg,), turns)
    for piece in result:
        assert piece.avg_logprob == seg.avg_logprob
        assert piece.no_speech_prob == seg.no_speech_prob


def test_word_level_zero_overlap_word_inherits_previous_speaker():
    """A word falling entirely in a diarization gap inherits the previous
    word's speaker."""
    words = (
        Word(start=0.0, end=1.0, text="a", prob=1.0),
        Word(start=1.0, end=2.0, text="b", prob=1.0),  # falls in the gap
        Word(start=3.0, end=4.0, text="c", prob=1.0),
    )
    seg = Segment(start=0.0, end=4.0, text="a b c", avg_logprob=0.0, no_speech_prob=0.0, words=words)
    # Gap between turns from 1.0 to 3.0 — word "b" overlaps neither turn.
    turns = [Turn(0.0, 1.0, "SPEAKER_00"), Turn(3.0, 4.0, "SPEAKER_01")]

    result = align.run((seg,), turns)

    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[0].text == "a b"
    assert result[1].speaker == "SPEAKER_01"
    assert result[1].text == "c"


def test_word_level_first_word_zero_overlap_uses_nearest_midpoint():
    """If the FIRST word has zero overlap, assign it to the turn whose
    midpoint is nearest."""
    words = (
        Word(start=0.0, end=0.5, text="a", prob=1.0),  # gap before first turn
        Word(start=2.0, end=3.0, text="b", prob=1.0),
    )
    seg = Segment(start=0.0, end=3.0, text="a b", avg_logprob=0.0, no_speech_prob=0.0, words=words)
    # Turn A midpoint = 2.5 (closer to word a's midpoint 0.25 than turn B's 5.5)
    turns = [Turn(2.0, 3.0, "SPEAKER_00"), Turn(5.0, 6.0, "SPEAKER_01")]

    result = align.run((seg,), turns)

    # Word "a" (zero overlap) -> nearest turn by midpoint is SPEAKER_00 (2.5 vs 5.5)
    # Word "b" overlaps SPEAKER_00 fully -> same speaker -> grouped together
    assert len(result) == 1
    assert result[0].speaker == "SPEAKER_00"
    assert result[0].text == "a b"


def test_word_level_segment_with_empty_words_uses_char_fallback():
    """Segments with empty `words` keep the char-proportional fallback."""
    seg = Segment(start=0.0, end=10.0, text="speaker zero then speaker one", avg_logprob=-0.2, no_speech_prob=0.1)
    turns = [Turn(0.0, 5.0, "SPEAKER_00"), Turn(5.0, 10.0, "SPEAKER_01")]
    result = align.run((seg,), turns)
    assert len(result) == 2
    assert result[0].speaker == "SPEAKER_00"
    assert result[1].speaker == "SPEAKER_01"

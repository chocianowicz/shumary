"""Tests for stages/align.py — speaker-boundary segment splitting."""

from shumary.models import Segment, Turn
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

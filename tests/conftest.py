"""Shared fixtures for all test modules."""

import pytest
from shumary.models import Segment, Turn, AlignedSegment, MoodSignals, PauseDetail


@pytest.fixture
def simple_segments() -> tuple[Segment, ...]:
    return (
        Segment(start=0.0, end=3.0, text="Hello world this is a test.", avg_logprob=-0.2, no_speech_prob=0.1),
        Segment(start=3.5, end=6.0, text="And this is another sentence.", avg_logprob=-0.3, no_speech_prob=0.05),
        Segment(start=10.0, end=13.0, text="After a long pause.", avg_logprob=-0.8, no_speech_prob=0.05),
    )


@pytest.fixture
def two_speaker_turns() -> list[Turn]:
    return [
        Turn(start=0.0, end=4.0, speaker="SPEAKER_00"),
        Turn(start=4.0, end=13.0, speaker="SPEAKER_01"),
    ]


@pytest.fixture
def sample_aligned(simple_segments) -> list[AlignedSegment]:
    return [
        AlignedSegment(
            start=s.start, end=s.end, text=s.text,
            avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob,
            speaker="SPEAKER_00",
        )
        for s in simple_segments
    ]


@pytest.fixture
def sample_mood() -> MoodSignals:
    return MoodSignals(
        avg_speaking_rate_wps=2.3,
        rate_variability=0.8,
        long_pauses=1,
        long_pause_details=(PauseDetail(at="00:06", duration_sec=4.0),),
        filler_word_count=3,
        top_fillers=("um ×2", "like ×1"),
        repeated_phrases=(),
        low_confidence_segments=1,
        near_silence_segments=0,
        inferred_state="(uncertain) delivery near baseline — no strong signals",
    )

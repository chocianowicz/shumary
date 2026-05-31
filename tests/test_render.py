"""Tests for stages/render.py — transcript markdown generation."""

import yaml
from pathlib import Path

from shumary.models import (
    TranscriptionResult, Segment, AlignedSegment,
    SpeakerStats, SpeakerStat, MoodSignals, PauseDetail,
)
from shumary.stages import render


def _make_transcription():
    return TranscriptionResult(
        language="en",
        word_count=10,
        text="Hello world. This is a test. After a pause.",
        segments=(
            Segment(0.0, 3.0, "Hello world.", -0.2, 0.1),
            Segment(3.5, 6.0, "This is a test.", -0.3, 0.05),
            Segment(10.0, 12.0, "After a pause.", -0.8, 0.05),
        ),
    )


def _make_aligned():
    return [
        AlignedSegment(0.0, 3.0, "Hello world.", -0.2, 0.1, "SPEAKER_00"),
        AlignedSegment(3.5, 6.0, "This is a test.", -0.3, 0.05, "SPEAKER_01"),
        AlignedSegment(10.0, 12.0, "After a pause.", -0.8, 0.05, "SPEAKER_01"),
    ]


def _make_mood():
    return MoodSignals(
        avg_speaking_rate_wps=2.1,
        rate_variability=0.5,
        long_pauses=1,
        long_pause_details=(PauseDetail(at="00:06", duration_sec=4.0),),
        filler_word_count=2,
        top_fillers=("um ×2",),
        repeated_phrases=(),
        low_confidence_segments=1,
        near_silence_segments=0,
        inferred_state="(uncertain) delivery near baseline",
    )


def _make_stats():
    return SpeakerStats(
        count=2,
        speakers=(
            SpeakerStat("SPEAKER_00", 3.0, 30),
            SpeakerStat("SPEAKER_01", 7.0, 70),
        ),
    )


def test_frontmatter_is_valid_yaml(tmp_path):
    fm = render._build_frontmatter(
        audio=Path("Inbox/test.m4a"),
        archived_path=Path("raw/audio/test.m4a"),
        transcription=_make_transcription(),
        mood=_make_mood(),
        speaker_stats=_make_stats(),
        date_str="2026-05-31",
        duration_sec=12.0,
        slug="test",
    )
    parsed = yaml.safe_load(fm)
    assert parsed["title"] == "test"
    assert parsed["language"] == "en"
    assert parsed["status"] == "pending-ingest"
    assert parsed["speakers"]["count"] == 2


def test_frontmatter_handles_special_chars(tmp_path):
    """yaml.safe_dump should not break on quotes or apostrophes in extracted text."""
    mood = MoodSignals(
        avg_speaking_rate_wps=2.0,
        rate_variability=None,
        long_pauses=0,
        long_pause_details=(),
        filler_word_count=1,
        top_fillers=("i don't know ×3",),
        repeated_phrases=('she said "hello"',),
        low_confidence_segments=0,
        near_silence_segments=0,
        inferred_state='(uncertain) "test" apostrophe\'s',
    )
    fm = render._build_frontmatter(
        audio=Path("Inbox/test.m4a"),
        archived_path=Path("raw/audio/test.m4a"),
        transcription=_make_transcription(),
        mood=mood,
        speaker_stats=None,
        date_str="2026-05-31",
        duration_sec=12.0,
        slug="test",
    )
    # Must parse cleanly
    parsed = yaml.safe_load(fm)
    assert parsed is not None


def test_save_transcript_creates_file(tmp_path, sample_mood):
    out = tmp_path / "test-transcript.md"
    result = render.save_transcript(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        segments=_make_aligned(),
        mood=_make_mood(),
        speaker_stats=_make_stats(),
        date_str="2026-05-31",
        slug="test",
        duration_sec=12.0,
        out_path=out,
    )
    assert result == out
    assert out.exists()
    content = out.read_text()
    assert "## Full Transcript" in content
    assert "## Speakers" in content
    assert "## Plain Text" in content
    assert "SPEAKER_00" in content


def test_transcript_body_contains_speaker_labels():
    body = render._render_transcript_body(_make_aligned())
    assert "**SPEAKER_00**" in body
    assert "**SPEAKER_01**" in body


def test_transcript_body_low_confidence_annotation():
    segs = [
        AlignedSegment(0.0, 2.0, "clear", -0.2, 0.1, "SPEAKER_00"),
        AlignedSegment(2.0, 4.0, "mumble", -1.5, 0.1, "SPEAKER_00"),
    ]
    body = render._render_transcript_body(segs)
    assert "low-confidence" in body


def test_mood_section_renders():
    section = render._render_mood_section(_make_mood())
    assert "Behavioral / Mood Signals" in section
    assert "2.1 wps" in section
    assert "(uncertain)" in section

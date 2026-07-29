"""Tests for stages/render.py — transcript markdown generation."""

import json
import yaml
from pathlib import Path

from shumary.config import LONG_PAUSE_SEC
from shumary.models import (
    TranscriptionResult, Segment, AlignedSegment,
    SpeakerStats, SpeakerStat, MoodSignals, PauseDetail, SpeakerMood,
    InteractionMetrics, SpeakerInteraction,
    ProsodyResult, SpeakerProsody,
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
    assert parsed["speakers"]["count"] == 2
    assert parsed["source_file"] == "Inbox/test.m4a"


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


# ── speaker section backend attribution (1.2 / A10) ──────────────────────────────

def test_speaker_section_reports_resemblyzer_backend():
    section = render._render_speaker_section(_make_stats(), diarize_backend="resemblyzer")
    assert "by resemblyzer" in section
    assert "pyannote" not in section


def test_speaker_section_reports_pyannote_backend():
    section = render._render_speaker_section(_make_stats(), diarize_backend="pyannote")
    assert "by pyannote.audio" in section


def test_speaker_section_unknown_backend_when_not_provided():
    section = render._render_speaker_section(_make_stats())
    assert "by an unknown backend" in section


def test_frontmatter_records_diarize_backend():
    fm = render._build_frontmatter(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        mood=_make_mood(),
        speaker_stats=_make_stats(),
        date_str="2026-05-31",
        duration_sec=12.0,
        slug="test",
        diarize_backend="resemblyzer",
    )
    parsed = yaml.safe_load(fm)
    assert parsed["speakers"]["backend"] == "resemblyzer"


# ── 5.3: per-speaker mood + interaction render integration ────────────────────────

def _make_per_speaker_mood():
    return (
        SpeakerMood(
            speaker="SPEAKER_00",
            speaking_rate_wps=1.5,
            rate_variability=0.3,
            intra_pauses=0,
            intra_pause_details=(),
            hesitations=1,
            discourse_markers=2,
            top_fillers=("uh ×1", "like ×2"),
            repeated_phrases=(),
            low_confidence_segments=0,
            inferred_state="(uncertain) baseline for SPEAKER_00",
        ),
        SpeakerMood(
            speaker="SPEAKER_01",
            speaking_rate_wps=3.8,
            rate_variability=1.4,
            intra_pauses=1,
            intra_pause_details=(PauseDetail(at="00:06", duration_sec=4.0),),
            hesitations=0,
            discourse_markers=1,
            top_fillers=("ну ×1",),
            repeated_phrases=("i don't know",),
            low_confidence_segments=2,
            inferred_state="(uncertain) fast delivery for SPEAKER_01",
        ),
    )


def _make_mood_with_per_speaker():
    base = _make_mood()
    return MoodSignals(
        avg_speaking_rate_wps=base.avg_speaking_rate_wps,
        rate_variability=base.rate_variability,
        long_pauses=base.long_pauses,
        long_pause_details=base.long_pause_details,
        filler_word_count=base.filler_word_count,
        top_fillers=base.top_fillers,
        repeated_phrases=base.repeated_phrases,
        low_confidence_segments=base.low_confidence_segments,
        near_silence_segments=base.near_silence_segments,
        inferred_state=base.inferred_state,
        per_speaker=_make_per_speaker_mood(),
    )


def _make_interaction():
    return InteractionMetrics(
        backend="pyannote",
        overlap_count=2,
        overlap_total_sec=1.5,
        long_gaps=1,
        speakers=(
            SpeakerInteraction(
                speaker="SPEAKER_00",
                turn_count=3,
                avg_turn_sec=2.0,
                median_response_latency_sec=0.8,
                p90_response_latency_sec=1.2,
                interruptions_made=1,
                interruptions_received=0,
                talk_share_by_third=(0.5, 0.4, 0.3),
            ),
            SpeakerInteraction(
                speaker="SPEAKER_01",
                turn_count=4,
                avg_turn_sec=3.0,
                median_response_latency_sec=None,
                p90_response_latency_sec=None,
                interruptions_made=0,
                interruptions_received=1,
                talk_share_by_third=(0.5, 0.6, 0.7),
            ),
        ),
    )


def test_frontmatter_includes_per_speaker_mood():
    fm = render._build_frontmatter(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        mood=_make_mood_with_per_speaker(),
        speaker_stats=_make_stats(),
        date_str="2026-05-31",
        duration_sec=12.0,
        slug="test",
    )
    parsed = yaml.safe_load(fm)
    per_speaker = parsed["mood_signals"]["per_speaker"]
    assert len(per_speaker) == 2
    assert per_speaker[0]["speaker"] == "SPEAKER_00"
    assert per_speaker[0]["speaking_rate_wps"] == 1.5
    assert per_speaker[1]["speaking_rate_wps"] == 3.8
    assert per_speaker[0]["hesitations"] == 1
    assert per_speaker[0]["discourse_markers"] == 2


def test_frontmatter_omits_per_speaker_when_empty():
    fm = render._build_frontmatter(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        mood=_make_mood(),  # per_speaker=() default
        speaker_stats=None,
        date_str="2026-05-31",
        duration_sec=12.0,
        slug="test",
    )
    parsed = yaml.safe_load(fm)
    assert parsed["mood_signals"]["per_speaker"] == []


def test_frontmatter_includes_interaction_block_when_provided():
    fm = render._build_frontmatter(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        mood=_make_mood_with_per_speaker(),
        speaker_stats=_make_stats(),
        date_str="2026-05-31",
        duration_sec=12.0,
        slug="test",
        diarize_backend="pyannote",
        interaction=_make_interaction(),
    )
    parsed = yaml.safe_load(fm)
    interaction_block = parsed["interaction"]
    assert interaction_block["backend"] == "pyannote"
    assert interaction_block["overlap_count"] == 2
    assert interaction_block["overlap_total_sec"] == 1.5
    assert interaction_block["long_gaps"] == 1
    assert len(interaction_block["speakers"]) == 2
    assert interaction_block["speakers"][0]["turn_count"] == 3
    assert interaction_block["speakers"][1]["median_response_latency_sec"] is None


def test_frontmatter_omits_interaction_block_for_single_speaker():
    fm = render._build_frontmatter(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        mood=_make_mood(),
        speaker_stats=None,
        date_str="2026-05-31",
        duration_sec=12.0,
        slug="test",
        interaction=None,
    )
    parsed = yaml.safe_load(fm)
    assert "interaction" not in parsed


def test_mood_section_renders_per_speaker_table():
    section = render._render_mood_section(_make_mood_with_per_speaker())
    assert "Per-Speaker Signals" in section
    assert "`SPEAKER_00`" in section
    assert "`SPEAKER_01`" in section
    assert "Hesitations" in section
    assert "Discourse markers" in section
    assert "(uncertain) baseline for SPEAKER_00" in section
    assert "(uncertain) fast delivery for SPEAKER_01" in section


def test_mood_section_omits_per_speaker_table_for_single_speaker():
    section = render._render_mood_section(_make_mood())
    assert "Per-Speaker Signals" not in section


def test_mood_section_uses_long_pause_sec_constant_not_hardcoded():
    section = render._render_mood_section(_make_mood())
    assert f"Long pauses (≥{LONG_PAUSE_SEC}s)" in section
    assert f"Intra-speaker pauses (≥{LONG_PAUSE_SEC}s)" in _render_per_speaker_table_for_test()


def _render_per_speaker_table_for_test() -> str:
    return render._render_mood_section(_make_mood_with_per_speaker())


def test_interaction_section_renders_metrics_table():
    section = render._render_interaction_section(_make_interaction())
    assert "Interaction Dynamics" in section
    assert "`SPEAKER_00`" in section
    assert "`SPEAKER_01`" in section
    assert "Turn count" in section
    assert "Median response latency" in section
    assert "Interruptions made" in section
    assert "Talk share by third" in section
    assert "(uncertain)" in section


def test_interaction_section_handles_none_latency_with_dash():
    section = render._render_interaction_section(_make_interaction())
    # SPEAKER_01 has None for both latency stats
    assert "—" in section


def test_interaction_section_overlap_none_for_resemblyzer():
    metrics = InteractionMetrics(
        backend="resemblyzer",
        overlap_count=None,
        overlap_total_sec=None,
        long_gaps=0,
        speakers=(
            SpeakerInteraction(
                speaker="SPEAKER_00", turn_count=1, avg_turn_sec=2.0,
                median_response_latency_sec=None, p90_response_latency_sec=None,
                interruptions_made=0, interruptions_received=0,
                talk_share_by_third=(1.0, 1.0, 1.0),
            ),
            SpeakerInteraction(
                speaker="SPEAKER_01", turn_count=1, avg_turn_sec=2.0,
                median_response_latency_sec=None, p90_response_latency_sec=None,
                interruptions_made=0, interruptions_received=0,
                talk_share_by_third=(0.0, 0.0, 0.0),
            ),
        ),
    )
    section = render._render_interaction_section(metrics)
    assert "not available" in section
    assert "resemblyzer turns cannot overlap" in section


def test_build_document_includes_interaction_section_only_when_provided():
    doc_with = render._build_document(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        segments=_make_aligned(),
        mood=_make_mood_with_per_speaker(),
        speaker_stats=_make_stats(),
        diarize_backend="pyannote",
        interaction=_make_interaction(),
        date_str="2026-05-31",
        slug="test",
        duration_sec=12.0,
    )
    assert "## Interaction Dynamics" in doc_with

    doc_without = render._build_document(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        segments=_make_aligned(),
        mood=_make_mood(),
        speaker_stats=None,
        date_str="2026-05-31",
        slug="test",
        duration_sec=12.0,
    )
    assert "## Interaction Dynamics" not in doc_without


# ── 6.1: paragraph merging, inline pause markers, --no-plain-text, quality notes ──

def test_consecutive_same_speaker_segments_merge_into_one_paragraph():
    segs = [
        AlignedSegment(0.0, 2.0, "Hello there.", -0.2, 0.1, "SPEAKER_00"),
        AlignedSegment(2.2, 4.0, "How are you doing today?", -0.2, 0.1, "SPEAKER_00"),
    ]
    body = render._render_transcript_body(segs)
    # One speaker header, one timestamp, both sentences in the same paragraph.
    assert body.count("**SPEAKER_00**") == 1
    assert body.count("[00:00]") == 1
    assert "Hello there. How are you doing today?" in body


def test_speaker_change_starts_new_paragraph_without_pause_marker():
    segs = [
        AlignedSegment(0.0, 2.0, "Hello there.", -0.2, 0.1, "SPEAKER_00"),
        AlignedSegment(20.0, 22.0, "Hi, good to see you.", -0.2, 0.1, "SPEAKER_01"),
    ]
    body = render._render_transcript_body(segs)
    assert "**SPEAKER_00**" in body
    assert "**SPEAKER_01**" in body
    # Large inter-speaker gap is response latency, not an intra-speaker pause.
    assert "*(pause" not in body


def test_intra_speaker_long_pause_inserts_marker_and_new_paragraph():
    segs = [
        AlignedSegment(0.0, 2.0, "First part.", -0.2, 0.1, "SPEAKER_00"),
        AlignedSegment(2.0 + LONG_PAUSE_SEC, 2.0 + LONG_PAUSE_SEC + 2.0, "Second part.", -0.2, 0.1, "SPEAKER_00"),
    ]
    body = render._render_transcript_body(segs)
    assert body.count("**SPEAKER_00**") == 1  # still the same speaker header
    assert f"*(pause {LONG_PAUSE_SEC:.1f}s)*" in body
    # The pause marker starts a new paragraph: a fresh timestamp for "Second part."
    assert body.count("[00:") == 2


def test_intra_speaker_short_gap_does_not_insert_pause_marker():
    segs = [
        AlignedSegment(0.0, 2.0, "First part.", -0.2, 0.1, "SPEAKER_00"),
        AlignedSegment(2.0 + (LONG_PAUSE_SEC - 0.5), 5.0, "Second part.", -0.2, 0.1, "SPEAKER_00"),
    ]
    body = render._render_transcript_body(segs)
    assert "*(pause" not in body
    assert body.count("[00:") == 1
    assert "First part. Second part." in body


def test_paragraph_breaks_at_word_count_on_sentence_boundary():
    # 8 sentences x 11 words = 88 words >= PARAGRAPH_MAX_WORDS (80).
    segs = []
    t = 0.0
    for i in range(9):
        text = f"This is sentence number {i} with some words in it."
        segs.append(AlignedSegment(t, t + 1.0, text, -0.2, 0.1, "SPEAKER_00"))
        t += 1.2

    body = render._render_transcript_body(segs)
    assert body.count("**SPEAKER_00**") == 1  # same speaker throughout
    # The paragraph should break into at least two timestamped chunks.
    assert body.count("[00:") >= 2


def test_paragraph_does_not_break_mid_sentence():
    """Even past the word threshold, a paragraph only breaks at a sentence
    boundary (i.e. after text ending in . ! ? or …)."""
    segs = [
        AlignedSegment(0.0, 1.0, "word " * 79 + "continuing", -0.2, 0.1, "SPEAKER_00"),
        AlignedSegment(1.2, 2.0, "the sentence now.", -0.2, 0.1, "SPEAKER_00"),
    ]
    body = render._render_transcript_body(segs)
    # 79+ words without a sentence end — must not break before "the sentence now."
    assert "continuing the sentence now." in body
    assert body.count("[00:") == 1


def test_low_confidence_annotation_preserved_within_paragraph():
    segs = [
        AlignedSegment(0.0, 2.0, "Clear text.", -0.2, 0.1, "SPEAKER_00"),
        AlignedSegment(2.2, 4.0, "Mumbled text.", -1.5, 0.1, "SPEAKER_00"),
    ]
    body = render._render_transcript_body(segs)
    assert body.count("**SPEAKER_00**") == 1
    assert "Clear text. Mumbled text." in body
    assert "low-confidence" in body


def test_near_silence_annotation_preserved_within_paragraph():
    segs = [
        AlignedSegment(0.0, 2.0, "Clear text.", -0.2, 0.1, "SPEAKER_00"),
        AlignedSegment(2.2, 4.0, "...", -0.2, 0.7, "SPEAKER_00"),
    ]
    body = render._render_transcript_body(segs)
    assert "near-silence" in body


# ── --no-plain-text (6.1) ──────────────────────────────────────────────────────

def test_build_document_includes_plain_text_by_default():
    doc = render._build_document(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        segments=_make_aligned(),
        mood=_make_mood(),
        speaker_stats=_make_stats(),
        date_str="2026-05-31",
        slug="test",
        duration_sec=12.0,
    )
    assert "## Plain Text" in doc


def test_build_document_omits_plain_text_when_disabled():
    doc = render._build_document(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        segments=_make_aligned(),
        mood=_make_mood(),
        speaker_stats=_make_stats(),
        date_str="2026-05-31",
        slug="test",
        duration_sec=12.0,
        plain_text=False,
    )
    assert "## Plain Text" not in doc
    # Everything else should still be present.
    assert "## Full Transcript" in doc


def test_save_transcript_respects_no_plain_text(tmp_path):
    out = tmp_path / "test-transcript.md"
    render.save_transcript(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        segments=_make_aligned(),
        mood=_make_mood(),
        speaker_stats=_make_stats(),
        date_str="2026-05-31",
        slug="test",
        duration_sec=12.0,
        out_path=out,
        plain_text=False,
    )
    content = out.read_text()
    assert "## Plain Text" not in content


# ── Quality notes (6.1) ─────────────────────────────────────────────────────────

def test_header_includes_quality_notes_when_hallucinations_filtered():
    transcription = TranscriptionResult(
        language="en",
        word_count=10,
        text="Hello world.",
        segments=(Segment(0.0, 3.0, "Hello world.", -0.2, 0.1),),
        hallucinations_filtered=2,
    )
    header = render._render_header("test", "2026-05-31", transcription, 12.0, Path("raw/audio/test.m4a"))
    assert "Quality notes" in header
    assert "2 hallucination-suspect segments filtered" in header


def test_header_omits_quality_notes_when_no_hallucinations_filtered():
    header = render._render_header("test", "2026-05-31", _make_transcription(), 12.0, Path("raw/audio/test.m4a"))
    assert "Quality notes" not in header


def test_header_quality_notes_singular_for_one_segment():
    transcription = TranscriptionResult(
        language="en",
        word_count=10,
        text="Hello world.",
        segments=(Segment(0.0, 3.0, "Hello world.", -0.2, 0.1),),
        hallucinations_filtered=1,
    )
    header = render._render_header("test", "2026-05-31", transcription, 12.0, Path("raw/audio/test.m4a"))
    assert "1 hallucination-suspect segment filtered" in header
    assert "segments filtered" not in header


def test_save_transcript_with_interaction_and_per_speaker_mood(tmp_path):
    out = tmp_path / "test-transcript.md"
    result = render.save_transcript(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        segments=_make_aligned(),
        mood=_make_mood_with_per_speaker(),
        speaker_stats=_make_stats(),
        diarize_backend="pyannote",
        interaction=_make_interaction(),
        date_str="2026-05-31",
        slug="test",
        duration_sec=12.0,
        out_path=out,
    )
    content = result.read_text()
    assert "## Interaction Dynamics" in content
    assert "Per-Speaker Signals" in content
    parsed = yaml.safe_load(content.split("---\n")[1])
    assert "interaction" in parsed
    assert len(parsed["mood_signals"]["per_speaker"]) == 2


# ── Phase 7.2: prosody render integration ──────────────────────────────────────

def _make_prosody_two_speakers():
    return ProsodyResult(
        segments=(),
        speakers=(
            SpeakerProsody(
                speaker="SPEAKER_00",
                f0_baseline_hz=142.3,
                f0_variability_st=2.1,
                energy_mean_db=-23.4,
                energy_dynamics_db=8.7,
                jitter_local=0.012,
                shimmer_local=0.061,
                f0_trend_by_third_st=(0.0, -0.4, -1.1),
            ),
            SpeakerProsody(
                speaker="SPEAKER_01",
                f0_baseline_hz=210.5,
                f0_variability_st=1.4,
                energy_mean_db=-19.8,
                energy_dynamics_db=6.2,
                jitter_local=None,
                shimmer_local=None,
                f0_trend_by_third_st=(0.2, 0.1, -0.1),
            ),
        ),
    )


def _make_prosody_single_speaker():
    return ProsodyResult(
        segments=(),
        speakers=(
            SpeakerProsody(
                speaker="speaker",
                f0_baseline_hz=118.0,
                f0_variability_st=1.0,
                energy_mean_db=-20.0,
                energy_dynamics_db=5.0,
                jitter_local=0.01,
                shimmer_local=0.05,
                f0_trend_by_third_st=(0.0, 0.0, 0.0),
            ),
        ),
    )


def test_frontmatter_includes_prosody_block_when_provided():
    fm = render._build_frontmatter(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        mood=_make_mood_with_per_speaker(),
        speaker_stats=_make_stats(),
        date_str="2026-05-31",
        duration_sec=12.0,
        slug="test",
        diarize_backend="pyannote",
        interaction=_make_interaction(),
        prosody=_make_prosody_two_speakers(),
    )
    parsed = yaml.safe_load(fm)
    prosody_block = parsed["prosody"]
    assert len(prosody_block) == 2
    assert prosody_block[0]["speaker"] == "SPEAKER_00"
    assert prosody_block[0]["f0_baseline_hz"] == 142.3
    assert prosody_block[0]["f0_variability_st"] == 2.1
    assert prosody_block[0]["energy_mean_db"] == -23.4
    assert prosody_block[0]["energy_dynamics_db"] == 8.7
    assert prosody_block[0]["jitter_local"] == 0.012
    assert prosody_block[0]["shimmer_local"] == 0.061
    assert prosody_block[0]["f0_trend_by_third_st"] == [0.0, -0.4, -1.1]
    assert prosody_block[1]["jitter_local"] is None
    assert prosody_block[1]["shimmer_local"] is None


def test_frontmatter_omits_prosody_block_when_none():
    fm = render._build_frontmatter(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        mood=_make_mood(),
        speaker_stats=None,
        date_str="2026-05-31",
        duration_sec=12.0,
        slug="test",
        prosody=None,
    )
    parsed = yaml.safe_load(fm)
    assert "prosody" not in parsed


def test_frontmatter_omits_prosody_block_when_no_speakers():
    fm = render._build_frontmatter(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        mood=_make_mood(),
        speaker_stats=None,
        date_str="2026-05-31",
        duration_sec=12.0,
        slug="test",
        prosody=ProsodyResult(segments=(), speakers=()),
    )
    parsed = yaml.safe_load(fm)
    assert "prosody" not in parsed


def test_prosody_section_renders_two_speaker_table():
    section = render._render_prosody_section(_make_prosody_two_speakers())
    assert "## Voice / Prosody" in section
    assert "`SPEAKER_00`" in section
    assert "`SPEAKER_01`" in section
    assert "F0 baseline (Hz)" in section
    assert "F0 variability (semitones)" in section
    assert "Energy mean (dB)" in section
    assert "Energy dynamics (dB)" in section
    assert "Jitter (local)" in section
    assert "Shimmer (local)" in section
    assert "F0 trend by third (semitones)" in section
    assert "(uncertain)" in section
    assert "not emotional diagnoses" in section
    # missing jitter/shimmer for SPEAKER_01 rendered as a dash
    assert "—" in section


def test_prosody_section_single_speaker_label():
    section = render._render_prosody_section(_make_prosody_single_speaker())
    assert "`speaker`" in section


def test_save_transcript_omits_prosody_section_when_none(tmp_path):
    out = tmp_path / "test-transcript.md"
    result = render.save_transcript(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        segments=_make_aligned(),
        mood=_make_mood(),
        speaker_stats=None,
        date_str="2026-05-31",
        slug="test",
        duration_sec=12.0,
        out_path=out,
        prosody=None,
    )
    content = result.read_text()
    assert "## Voice / Prosody" not in content
    parsed = yaml.safe_load(content.split("---\n")[1])
    assert "prosody" not in parsed


def test_save_transcript_includes_prosody_section_when_provided(tmp_path):
    out = tmp_path / "test-transcript.md"
    result = render.save_transcript(
        audio=Path("Inbox/test.m4a"),
        transcription=_make_transcription(),
        segments=_make_aligned(),
        mood=_make_mood_with_per_speaker(),
        speaker_stats=_make_stats(),
        diarize_backend="pyannote",
        interaction=_make_interaction(),
        prosody=_make_prosody_two_speakers(),
        date_str="2026-05-31",
        slug="test",
        duration_sec=12.0,
        out_path=out,
    )
    content = result.read_text()
    assert "## Voice / Prosody" in content
    parsed = yaml.safe_load(content.split("---\n")[1])
    assert "prosody" in parsed
    assert len(parsed["prosody"]) == 2


# ── JSON transcript ────────────────────────────────────────────────────────────

def test_save_transcript_json_carries_segments_and_summary(tmp_path, simple_segments, sample_aligned, sample_mood):
    from shumary.models import TranscriptionResult

    transcription = TranscriptionResult(
        language="en", word_count=12, text="hello world", segments=simple_segments,
    )
    out = tmp_path / "transcript.json"

    render.save_transcript_json(
        audio=tmp_path / "meeting.wav",
        transcription=transcription,
        segments=sample_aligned,
        mood=sample_mood,
        speaker_stats=None,
        date_str="2026-07-29",
        slug="meeting",
        duration_sec=13.0,
        out_path=out,
    )

    payload = json.loads(out.read_text())
    assert payload["language"] == "en"
    assert payload["text"] == "hello world"
    assert len(payload["segments"]) == len(sample_aligned)
    assert payload["segments"][0]["speaker"] == "SPEAKER_00"
    assert "mood_signals" in payload


def test_json_and_markdown_agree_on_the_summary(tmp_path, simple_segments, sample_aligned, sample_mood):
    from shumary.models import TranscriptionResult

    transcription = TranscriptionResult(
        language="en", word_count=12, text="hello world", segments=simple_segments,
    )
    common = dict(
        audio=tmp_path / "meeting.wav",
        transcription=transcription,
        segments=sample_aligned,
        mood=sample_mood,
        speaker_stats=None,
        date_str="2026-07-29",
        slug="meeting",
        duration_sec=13.0,
    )
    md = render.save_transcript(out_path=tmp_path / "t.md", **common)
    js = render.save_transcript_json(out_path=tmp_path / "t.json", **common)

    payload = json.loads(js.read_text())
    assert payload["title"] in md.read_text()
    assert payload["date"] == "2026-07-29"

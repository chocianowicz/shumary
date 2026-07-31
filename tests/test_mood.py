"""Tests for stages/mood.py — signal extraction."""

from shumary.models import AlignedSegment, MoodSignals, PauseDetail, SpeakerMood
from shumary.stages import mood


def _seg(start, end, text, logprob=-0.2, no_speech=0.0, speaker=None):
    return AlignedSegment(start=start, end=end, text=text, avg_logprob=logprob, no_speech_prob=no_speech, speaker=speaker)


def test_long_pause_detected():
    segs = [_seg(0, 2, "hello"), _seg(6, 8, "pause after this")]
    result = mood.run(segs, "en")
    assert result.long_pauses == 1
    assert result.long_pause_details[0].at == "00:02"


def test_no_long_pause():
    segs = [_seg(0, 2, "hello"), _seg(2.5, 4, "world")]
    result = mood.run(segs, "en")
    assert result.long_pauses == 0


def test_filler_count_english():
    segs = [_seg(0, 5, "um I think uh this is like a test um")]
    result = mood.run(segs, "en")
    assert result.filler_word_count >= 3


def test_filler_count_russian():
    segs = [_seg(0, 5, "ну вот это значит так ну как бы")]
    result = mood.run(segs, "ru")
    assert result.filler_word_count >= 3


def test_low_confidence_count():
    segs = [
        _seg(0, 2, "clear", logprob=-0.1),
        _seg(2, 4, "mumble", logprob=-1.5),
        _seg(4, 6, "mumble again", logprob=-2.0),
    ]
    result = mood.run(segs, "en")
    assert result.low_confidence_segments == 2


def test_near_silence_count():
    segs = [
        _seg(0, 2, "speech", no_speech=0.1),
        _seg(2, 4, "silence", no_speech=0.8),
    ]
    result = mood.run(segs, "en")
    assert result.near_silence_segments == 1


def test_inferred_state_has_uncertain_prefix():
    segs = [_seg(0, 2, "normal")]
    result = mood.run(segs, "en")
    assert result.inferred_state.startswith("(uncertain)")


def test_slow_rate_detected():
    # 1 word / 2 seconds = 0.5 wps — well below SLOW_RATE threshold
    segs = [_seg(i * 2, i * 2 + 2, "slow") for i in range(10)]
    result = mood.run(segs, "en")
    assert "slow delivery" in result.inferred_state


def test_serialisation_roundtrip(sample_mood):
    d = sample_mood.to_dict()
    restored = sample_mood.__class__.from_dict(d)
    assert restored == sample_mood


# ── filler regex ordering (1.7 / A5) ─────────────────────────────────────────────

def test_multiword_filler_no_to_beats_no():
    """'no to' must match as a whole, not be shadowed by the shorter 'no'.

    Asserted against the pattern the pipeline actually uses — both `no` and
    `no to` live in the discourse-marker lexicon, so this is where the
    longest-alternative-first ordering has to hold.
    """
    pattern = mood._discourse_marker_pattern()
    matches = pattern.findall("no to zaczynamy")
    assert "no to" in matches
    assert "no" not in matches


def test_filler_count_polish_no_to():
    segs = [_seg(0, 5, "no to zaczynamy pracę")]
    result = mood.run(segs, "pl")
    assert any("no to" in f for f in result.top_fillers)


# ── B7: filler matcher always merges en+ru+pl regardless of language ─────────────

def test_filler_matcher_merges_all_languages_regardless_of_detected_language():
    """A Russian discourse marker must be detected even when language='en',
    and an English hesitation must be detected even when language='ru'."""
    segs_ru_in_en = [_seg(0, 5, "ну вот this is fine", speaker=None)]
    result = mood.run(segs_ru_in_en, "en")
    assert result.filler_word_count >= 2

    segs_en_in_ru = [_seg(0, 5, "um и потом я сказал", speaker=None)]
    result2 = mood.run(segs_en_in_ru, "ru")
    assert result2.filler_word_count >= 1


# ── B6: hesitations vs discourse markers split ────────────────────────────────────

def test_hesitation_pattern_matches_pure_fillers():
    pattern = mood._hesitation_pattern()
    for word in ("uh", "um", "er", "hmm", "эм", "эхм", "eee", "yyy"):
        assert pattern.search(word), f"{word!r} should match hesitation pattern"


def test_discourse_marker_pattern_matches_lexical_fillers():
    pattern = mood._discourse_marker_pattern()
    for word in ("like", "you know", "ну", "вот", "короче", "типа", "как бы", "no", "właśnie", "jakby", "znaczy"):
        assert pattern.search(word), f"{word!r} should match discourse marker pattern"


def test_high_frequency_function_words_are_not_discourse_markers():
    """`это` is an ordinary demonstrative pronoun. On a real Russian call it was
    138 of 271 marker hits, over half the signal from one very common word."""
    pattern = mood._discourse_marker_pattern()
    assert not pattern.search("это")


def test_count_fillers_split_separates_hesitations_and_markers():
    text = "uh so like i was um you know thinking ну вот"
    hes, markers, top = mood._count_fillers_split(text)
    assert hes >= 2   # uh, um
    assert markers >= 3  # like, you know, ну, вот


def test_global_filler_word_count_is_hesitations_plus_markers():
    segs = [_seg(0, 5, "uh um like you know")]
    result = mood.run(segs, "en")
    hes, markers, _ = mood._count_fillers_split("uh um like you know")
    assert result.filler_word_count == hes + markers


# ── 5.1: per-speaker mood signals ─────────────────────────────────────────────────

def test_per_speaker_empty_when_single_speaker():
    segs = [_seg(0, 2, "hello", speaker="SPEAKER_00"), _seg(2, 4, "world", speaker="SPEAKER_00")]
    result = mood.run(segs, "en")
    assert result.per_speaker == ()


def test_per_speaker_empty_when_no_speaker_labels():
    segs = [_seg(0, 2, "hello"), _seg(2, 4, "world")]
    result = mood.run(segs, "en")
    assert result.per_speaker == ()


def test_per_speaker_separates_two_speakers():
    """Two speakers with very different speaking rates must yield distinct
    per-speaker speaking_rate_wps values."""
    # SPEAKER_00: 1 word per 2 seconds (slow)
    slow_segs = [_seg(i * 4, i * 4 + 2, "slow", speaker="SPEAKER_00") for i in range(5)]
    # SPEAKER_01: 4 words per 1 second (fast), interleaved
    fast_segs = [_seg(i * 4 + 2, i * 4 + 3, "this is very fast", speaker="SPEAKER_01") for i in range(5)]

    segs = []
    for a, b in zip(slow_segs, fast_segs):
        segs.append(a)
        segs.append(b)

    result = mood.run(segs, "en")
    assert len(result.per_speaker) == 2

    by_speaker = {sm.speaker: sm for sm in result.per_speaker}
    assert set(by_speaker) == {"SPEAKER_00", "SPEAKER_01"}

    slow = by_speaker["SPEAKER_00"]
    fast = by_speaker["SPEAKER_01"]
    assert slow.speaking_rate_wps < fast.speaking_rate_wps
    assert slow.speaking_rate_wps == 0.5  # 1 word / 2 sec
    assert fast.speaking_rate_wps == 4.0  # 4 words / 1 sec


def test_per_speaker_duration_weighted_not_mean_of_segment_rates():
    """speaking_rate_wps must be total words / total speech seconds, not the
    mean of per-segment rates — these differ when segment durations vary."""
    segs = [
        _seg(0, 1, "one word here now", speaker="SPEAKER_00"),   # 4 words / 1s = 4 wps
        _seg(1, 11, "slow", speaker="SPEAKER_00"),                # 1 word / 10s = 0.1 wps
        _seg(11, 12, "other", speaker="SPEAKER_01"),
    ]
    result = mood.run(segs, "en")
    by_speaker = {sm.speaker: sm for sm in result.per_speaker}
    speaker0 = by_speaker["SPEAKER_00"]

    # Duration-weighted: (4 + 1) words / (1 + 10) sec = 5/11
    expected = round(5 / 11, 2)
    # Mean of per-segment rates would be (4 + 0.1) / 2 = 2.05 — must NOT match
    mean_of_rates = round((4.0 + 0.1) / 2, 2)

    assert speaker0.speaking_rate_wps == expected
    assert speaker0.speaking_rate_wps != mean_of_rates


def test_intra_pause_only_counted_for_same_speaker_consecutive_segments():
    """A gap >= LONG_PAUSE_SEC between SPEAKER_00 segments with no
    intervening SPEAKER_01 segment is an intra-pause for SPEAKER_00. A gap
    where the *other* speaker's segment falls in between is NOT an
    intra-pause for either speaker (it's inter-speaker latency, Spec 5)."""
    segs = [
        _seg(0, 1, "first", speaker="SPEAKER_00"),
        _seg(6, 7, "second after long gap", speaker="SPEAKER_00"),  # 5s gap, same speaker
        _seg(8, 9, "third", speaker="SPEAKER_01"),
        _seg(13, 14, "fourth after long gap", speaker="SPEAKER_00"),  # 4s gap, but SPEAKER_01 in between
    ]
    result = mood.run(segs, "en")
    by_speaker = {sm.speaker: sm for sm in result.per_speaker}

    # SPEAKER_00 has exactly one true intra-pause (segments 0->1, gap=5s)
    assert by_speaker["SPEAKER_00"].intra_pauses == 1
    assert by_speaker["SPEAKER_00"].intra_pause_details[0].duration_sec == 5.0

    # SPEAKER_01 has no intra-pauses (only one segment)
    assert by_speaker["SPEAKER_01"].intra_pauses == 0


def test_per_speaker_hesitations_vs_discourse_markers():
    segs = [
        _seg(0, 5, "uh um I think this is fine", speaker="SPEAKER_00"),
        _seg(5, 10, "like you know it's complicated", speaker="SPEAKER_01"),
    ]
    result = mood.run(segs, "en")
    by_speaker = {sm.speaker: sm for sm in result.per_speaker}

    assert by_speaker["SPEAKER_00"].hesitations >= 2  # uh, um
    assert by_speaker["SPEAKER_01"].discourse_markers >= 2  # like, you know


def test_per_speaker_low_confidence_segments():
    segs = [
        _seg(0, 2, "clear", logprob=-0.1, speaker="SPEAKER_00"),
        _seg(2, 4, "mumble", logprob=-1.5, speaker="SPEAKER_00"),
        _seg(4, 6, "fine", logprob=-0.2, speaker="SPEAKER_01"),
    ]
    result = mood.run(segs, "en")
    by_speaker = {sm.speaker: sm for sm in result.per_speaker}

    assert by_speaker["SPEAKER_00"].low_confidence_segments == 1
    assert by_speaker["SPEAKER_01"].low_confidence_segments == 0


def test_per_speaker_inferred_state_has_uncertain_prefix():
    segs = [
        _seg(0, 2, "hello", speaker="SPEAKER_00"),
        _seg(2, 4, "world", speaker="SPEAKER_01"),
    ]
    result = mood.run(segs, "en")
    for sm in result.per_speaker:
        assert sm.inferred_state.startswith("(uncertain)")


# ── 5.1: backward-compatible deserialization ──────────────────────────────────────

def test_mood_signals_from_dict_without_per_speaker_key_defaults_empty():
    """Old cached mood entries (pre-Phase-5) lack `per_speaker` entirely —
    deserialization must not raise and must default to an empty tuple."""
    old_payload = {
        "avg_speaking_rate_wps": 2.0,
        "rate_variability": None,
        "long_pauses": 0,
        "long_pause_details": [],
        "filler_word_count": 0,
        "top_fillers": [],
        "repeated_phrases": [],
        "low_confidence_segments": 0,
        "near_silence_segments": 0,
        "inferred_state": "(uncertain) baseline",
        # no "per_speaker" key
    }
    restored = MoodSignals.from_dict(old_payload)
    assert restored.per_speaker == ()


def test_speaker_mood_serialisation_roundtrip():
    sm = SpeakerMood(
        speaker="SPEAKER_00",
        speaking_rate_wps=2.5,
        rate_variability=0.5,
        intra_pauses=1,
        intra_pause_details=(PauseDetail(at="00:05", duration_sec=4.0),),
        hesitations=2,
        discourse_markers=1,
        top_fillers=("uh ×2",),
        repeated_phrases=("i don't know",),
        low_confidence_segments=1,
        inferred_state="(uncertain) baseline",
    )
    d = sm.to_dict()
    restored = SpeakerMood.from_dict(d)
    assert restored == sm


def test_mood_signals_with_per_speaker_roundtrip():
    segs = [
        _seg(0, 2, "hello there", speaker="SPEAKER_00"),
        _seg(2, 4, "world again", speaker="SPEAKER_01"),
    ]
    result = mood.run(segs, "en")
    d = result.to_dict()
    restored = MoodSignals.from_dict(d)
    assert restored == result
    assert len(restored.per_speaker) == 2


# ── the filler verdict rests on hesitations, not on ordinary words ─────────────

def _seg_text(text, speaker="SPEAKER_00", start=0.0, end=10.0):
    from shumary.models import AlignedSegment
    return AlignedSegment(start=start, end=end, text=text, avg_logprob=-0.2,
                          no_speech_prob=0.05, speaker=speaker)


def test_many_discourse_markers_do_not_imply_discomfort():
    """The failure this guards: a Russian call with zero hesitations was reported
    as showing "cognitive load or discomfort" purely from ordinary speech."""
    text = " ".join(["ну вот значит короче типа"] * 20)   # 100 markers, 0 hesitations
    m = mood.run([_seg_text(text)], "ru")

    assert m.hesitations == 0
    assert m.discourse_markers >= 40
    assert "discomfort" not in m.inferred_state
    assert "conversational style" in m.inferred_state


def test_many_hesitations_do_imply_load():
    text = " ".join(["uh um er"] * 10)   # 30 hesitations
    m = mood.run([_seg_text(text)], "en")

    assert m.hesitations >= 15
    assert "frequent hesitations — cognitive load or discomfort" in m.inferred_state


def test_filler_word_count_still_reports_the_combined_total():
    m = mood.run([_seg_text("uh ну вот")], "ru")
    assert m.filler_word_count == m.hesitations + m.discourse_markers


def test_per_speaker_breakdown_restricted_to_accepted_speakers():
    """A label disqualified from the speaker table must not reappear here."""
    segs = [
        _seg_text("hello there friends", "SPEAKER_00", 0.0, 5.0),
        _seg_text("yes indeed quite so", "SPEAKER_01", 5.0, 10.0),
        _seg_text("ага", "SPEAKER_02", 10.0, 11.4),
    ]
    m = mood.run(segs, "en", {"SPEAKER_00", "SPEAKER_01"})

    assert {s.speaker for s in m.per_speaker} == {"SPEAKER_00", "SPEAKER_01"}
    # Global text stats still cover every transcribed word.
    assert m.avg_speaking_rate_wps is not None

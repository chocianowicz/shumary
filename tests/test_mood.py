"""Tests for stages/mood.py — signal extraction."""

from shumary.models import AlignedSegment
from shumary.stages import mood


def _seg(start, end, text, logprob=-0.2, no_speech=0.0):
    return AlignedSegment(start=start, end=end, text=text, avg_logprob=logprob, no_speech_prob=no_speech, speaker=None)


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

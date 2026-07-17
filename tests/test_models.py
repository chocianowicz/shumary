"""Tests for models.py — Word/Segment serialization (Spec 3 data model)."""

from shumary.models import Segment, TranscriptionResult, Word


def test_word_to_dict_from_dict_roundtrip():
    w = Word(start=0.0, end=0.5, text="hello", prob=0.9)
    d = w.to_dict()
    assert d == {"start": 0.0, "end": 0.5, "text": "hello", "prob": 0.9}
    assert Word.from_dict(d) == w


def test_segment_default_words_is_empty_tuple():
    seg = Segment(start=0.0, end=1.0, text="hi", avg_logprob=-0.1, no_speech_prob=0.05)
    assert seg.words == ()


def test_segment_to_dict_from_dict_roundtrip_with_words():
    seg = Segment(
        start=0.0, end=1.0, text="hi there",
        avg_logprob=-0.1, no_speech_prob=0.05,
        words=(
            Word(start=0.0, end=0.4, text="hi", prob=0.95),
            Word(start=0.4, end=1.0, text="there", prob=0.88),
        ),
    )
    d = seg.to_dict()
    assert d["words"] == [
        {"start": 0.0, "end": 0.4, "text": "hi", "prob": 0.95},
        {"start": 0.4, "end": 1.0, "text": "there", "prob": 0.88},
    ]
    restored = Segment.from_dict(d)
    assert restored == seg


def test_segment_from_dict_defaults_missing_words_to_empty():
    d = {"start": 0.0, "end": 1.0, "text": "hi", "avg_logprob": -0.1, "no_speech_prob": 0.05}
    seg = Segment.from_dict(d)
    assert seg.words == ()


def test_transcription_result_roundtrip_with_words():
    tr = TranscriptionResult(
        language="en",
        word_count=2,
        text="hi there",
        segments=(
            Segment(
                start=0.0, end=1.0, text="hi there",
                avg_logprob=-0.1, no_speech_prob=0.05,
                words=(Word(start=0.0, end=0.4, text="hi", prob=0.95),),
            ),
        ),
    )
    d = tr.to_dict()
    restored = TranscriptionResult.from_dict(d)
    assert restored == tr

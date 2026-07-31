"""Tests for backends.py — TranscribeBackend protocol, normalization, pick_backend (Spec 3).

No real models are loaded or downloaded: each backend's underlying library call is
faked via monkeypatching, using result shapes observed during the manual smoke test
(see the 3.2 commit message for the raw shapes).
"""

from pathlib import Path

import pytest

from shumary import backends
from shumary.errors import ShumaryError
from shumary.models import Segment, TranscriptionResult


# ── _normalize_words ─────────────────────────────────────────────────────────────

def test_normalize_words_from_dicts_strips_leading_space():
    raw = [
        {"word": " Hello", "start": 0.0, "end": 0.4, "probability": 0.9},
        {"word": " world", "start": 0.4, "end": 0.8, "probability": 0.8},
    ]
    words = backends._normalize_words(raw)
    assert words[0].text == "Hello"
    assert words[1].text == "world"
    assert words[0].prob == 0.9
    assert words[0].start == 0.0
    assert words[0].end == 0.4


def test_normalize_words_missing_probability_defaults_to_one():
    raw = [{"word": " Hi", "start": 0.0, "end": 0.2}]
    words = backends._normalize_words(raw)
    assert words[0].prob == 1.0


class _FakeFasterWord:
    """Mimics faster_whisper.transcribe.Word (start, end, word, probability)."""

    def __init__(self, start, end, word, probability):
        self.start = start
        self.end = end
        self.word = word
        self.probability = probability


def test_normalize_words_from_faster_whisper_objects():
    raw = [_FakeFasterWord(0.0, 0.24, " This", 0.94)]
    words = backends._normalize_words(raw)
    assert words[0].text == "This"
    assert words[0].prob == 0.94
    assert words[0].start == 0.0
    assert words[0].end == 0.24


# ── pick_backend ──────────────────────────────────────────────────────────────────

class _FakeBackend:
    def __init__(self, name, is_available):
        self.name = name
        self._is_available = is_available

    def available(self):
        return self._is_available


@pytest.fixture
def fake_backends(monkeypatch):
    fakes = {
        "mlx": _FakeBackend("mlx", False),
        "faster": _FakeBackend("faster", False),
        "openai": _FakeBackend("openai", True),
    }
    monkeypatch.setattr(backends, "ALL_BACKENDS", fakes)
    monkeypatch.delenv("SHUMARY_BACKEND", raising=False)
    return fakes


def test_pick_backend_unknown_override_raises(fake_backends):
    with pytest.raises(ShumaryError, match="Unknown backend"):
        backends.pick_backend("nope")


def test_pick_backend_unavailable_override_raises(fake_backends):
    with pytest.raises(ShumaryError, match="not installed"):
        backends.pick_backend("mlx")


def test_pick_backend_available_override_wins(fake_backends):
    fake_backends["mlx"]._is_available = True
    assert backends.pick_backend("mlx").name == "mlx"


def test_pick_backend_env_var_override(fake_backends, monkeypatch):
    fake_backends["faster"]._is_available = True
    monkeypatch.setenv("SHUMARY_BACKEND", "faster")
    assert backends.pick_backend().name == "faster"


def test_pick_backend_prefers_mlx_on_apple_silicon(fake_backends, monkeypatch):
    fake_backends["mlx"]._is_available = True
    fake_backends["faster"]._is_available = True
    monkeypatch.setattr(backends, "_is_apple_silicon", lambda: True)
    assert backends.pick_backend().name == "mlx"


def test_pick_backend_skips_mlx_when_unavailable_falls_back_to_faster(fake_backends, monkeypatch):
    fake_backends["mlx"]._is_available = False
    fake_backends["faster"]._is_available = True
    monkeypatch.setattr(backends, "_is_apple_silicon", lambda: True)
    assert backends.pick_backend().name == "faster"


def test_pick_backend_not_apple_silicon_uses_faster_if_available(fake_backends, monkeypatch):
    fake_backends["mlx"]._is_available = True  # would be picked on darwin/arm64
    fake_backends["faster"]._is_available = True
    monkeypatch.setattr(backends, "_is_apple_silicon", lambda: False)
    assert backends.pick_backend().name == "faster"


def test_pick_backend_falls_back_to_openai(fake_backends, monkeypatch):
    monkeypatch.setattr(backends, "_is_apple_silicon", lambda: True)
    assert backends.pick_backend().name == "openai"


# ── OpenaiBackend.transcribe ────────────────────────────────────────────────────

class _FakeOpenaiModel:
    def __init__(self, captured):
        self._captured = captured

    def transcribe(self, path, **opts):
        self._captured["path"] = path
        self._captured["opts"] = opts
        return {
            "language": "en",
            "text": " This is a short smoke test for the shumary pipeline backends.",
            "segments": [
                {
                    "start": 0.0, "end": 3.16,
                    "text": " This is a short smoke test for the shumary pipeline backends.",
                    "avg_logprob": -0.498, "no_speech_prob": 0.001,
                    "words": [
                        {"word": " This", "start": 0.0, "end": 0.24, "probability": 0.93},
                        {"word": " is", "start": 0.24, "end": 0.5, "probability": 0.99},
                    ],
                }
            ],
        }


def test_openai_backend_transcribe_normalizes_result(monkeypatch):
    captured = {}
    fake_model = _FakeOpenaiModel(captured)
    monkeypatch.setattr("shumary.modelhub.whisper_model", lambda name: fake_model)

    backend = backends.OpenaiBackend()
    result = backend.transcribe(Path("/tmp/audio.wav"), "tiny", fast=True)

    assert isinstance(result, TranscriptionResult)
    assert result.language == "en"
    assert result.word_count == len(result.text.split())
    seg = result.segments[0]
    assert seg.start == 0.0 and seg.end == 3.16
    assert seg.words[0].text == "This"  # leading space stripped
    assert seg.words[0].prob == 0.93

    # Mandatory call options.
    opts = captured["opts"]
    assert opts["word_timestamps"] is True
    assert opts["condition_on_previous_text"] is False
    assert opts["temperature"] == 0.0  # --fast


def test_openai_backend_no_fast_omits_temperature(monkeypatch):
    captured = {}
    fake_model = _FakeOpenaiModel(captured)
    monkeypatch.setattr("shumary.modelhub.whisper_model", lambda name: fake_model)

    backend = backends.OpenaiBackend()
    backend.transcribe(Path("/tmp/audio.wav"), "tiny", fast=False)

    assert "temperature" not in captured["opts"]


# ── MlxBackend.transcribe ────────────────────────────────────────────────────────

def test_mlx_backend_transcribe_normalizes_result(monkeypatch):
    import sys
    import types

    captured = {}

    def _fake_transcribe(path, **opts):
        captured["path"] = path
        captured["opts"] = opts
        return {
            "language": "en",
            "text": " This is a short smoke test.",
            "segments": [
                {
                    "start": 0.0, "end": 3.16,
                    "text": " This is a short smoke test.",
                    "avg_logprob": -0.498, "no_speech_prob": 0.001,
                    "words": [
                        {"word": " This", "start": 0.0, "end": 0.24, "probability": 0.93},
                    ],
                }
            ],
        }

    fake_module = types.ModuleType("mlx_whisper")
    fake_module.transcribe = _fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_module)

    backend = backends.MlxBackend()
    result = backend.transcribe(Path("/tmp/audio.wav"), "tiny", fast=True)

    assert result.language == "en"
    assert result.segments[0].words[0].text == "This"

    opts = captured["opts"]
    assert opts["path_or_hf_repo"] == backends.KNOWN_REPOS["tiny"]
    assert opts["word_timestamps"] is True
    assert opts["condition_on_previous_text"] is False
    assert opts["temperature"] == 0.0


# ── FasterBackend.transcribe ────────────────────────────────────────────────────

class _FakeFasterSegment:
    def __init__(self, start, end, text, avg_logprob, no_speech_prob, words):
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.words = words


class _FakeFasterInfo:
    def __init__(self, language):
        self.language = language


class _FakeFasterModel:
    def __init__(self, captured):
        self._captured = captured

    def transcribe(self, path, **opts):
        self._captured["path"] = path
        self._captured["opts"] = opts

        def _gen():
            yield _FakeFasterSegment(
                0.0, 3.16, " This is a short smoke test.",
                -0.45, 0.001,
                words=[_FakeFasterWord(0.0, 0.24, " This", 0.94)],
            )

        return _gen(), _FakeFasterInfo(language="en")


def test_faster_backend_transcribe_consumes_generator_and_normalizes(monkeypatch):
    captured = {}
    fake_model = _FakeFasterModel(captured)
    monkeypatch.setattr(backends, "_faster_model", lambda name: fake_model)

    backend = backends.FasterBackend()
    result = backend.transcribe(Path("/tmp/audio.wav"), "tiny", fast=True)

    assert result.language == "en"
    seg = result.segments[0]
    assert seg.start == 0.0 and seg.end == 3.16
    assert seg.words[0].text == "This"
    assert seg.words[0].prob == 0.94

    opts = captured["opts"]
    assert opts["word_timestamps"] is True
    assert opts["vad_filter"] is True
    assert opts["condition_on_previous_text"] is False
    assert opts["beam_size"] == 1
    assert opts["temperature"] == 0.0


def test_faster_backend_no_fast_omits_beam_and_temperature(monkeypatch):
    captured = {}
    fake_model = _FakeFasterModel(captured)
    monkeypatch.setattr(backends, "_faster_model", lambda name: fake_model)

    backend = backends.FasterBackend()
    backend.transcribe(Path("/tmp/audio.wav"), "tiny", fast=False)

    assert "beam_size" not in captured["opts"]
    assert "temperature" not in captured["opts"]


# ── Hallucination post-filter (Spec 3-VAD, 3.3) ──────────────────────────────────

def _seg(start, end, text, no_speech_prob=0.05):
    return Segment(start=start, end=end, text=text, avg_logprob=-0.2, no_speech_prob=no_speech_prob)


def test_filter_hallucinations_drops_repeated_high_silence_long_segment():
    segments = (
        _seg(0.0, 5.0, "Thank you.", no_speech_prob=0.05),
        _seg(5.0, 8.0, "Thank you.", no_speech_prob=0.9),  # repeats prev, >=2s, high no_speech
    )
    kept, dropped = backends.filter_hallucinations(segments)
    assert dropped == 1
    assert kept == (segments[0],)


def test_filter_hallucinations_keeps_short_repeats():
    segments = (
        _seg(0.0, 5.0, "Thank you.", no_speech_prob=0.05),
        _seg(5.0, 6.5, "Thank you.", no_speech_prob=0.9),  # duration 1.5s < 2.0s
    )
    kept, dropped = backends.filter_hallucinations(segments)
    assert dropped == 0
    assert kept == segments


def test_filter_hallucinations_keeps_low_no_speech_repeats():
    segments = (
        _seg(0.0, 5.0, "Thank you.", no_speech_prob=0.05),
        _seg(5.0, 8.0, "Thank you.", no_speech_prob=0.1),  # below threshold
    )
    kept, dropped = backends.filter_hallucinations(segments)
    assert dropped == 0
    assert kept == segments


def test_filter_hallucinations_keeps_different_text():
    segments = (
        _seg(0.0, 5.0, "Thank you.", no_speech_prob=0.05),
        _seg(5.0, 8.0, "Goodbye.", no_speech_prob=0.9),
    )
    kept, dropped = backends.filter_hallucinations(segments)
    assert dropped == 0
    assert kept == segments


def test_filter_hallucinations_normalization_ignores_case_and_punctuation():
    segments = (
        _seg(0.0, 5.0, "Thank you.", no_speech_prob=0.05),
        _seg(5.0, 8.0, "thank you", no_speech_prob=0.9),  # same modulo case/punctuation
    )
    kept, dropped = backends.filter_hallucinations(segments)
    assert dropped == 1
    assert kept == (segments[0],)


def test_filter_hallucinations_collapses_run_of_repeats():
    segments = (
        _seg(0.0, 5.0, "Thank you.", no_speech_prob=0.05),
        _seg(5.0, 8.0, "Thank you.", no_speech_prob=0.9),
        _seg(8.0, 11.0, "Thank you.", no_speech_prob=0.9),
        _seg(11.0, 14.0, "Thank you.", no_speech_prob=0.9),
    )
    kept, dropped = backends.filter_hallucinations(segments)
    assert dropped == 3
    assert kept == (segments[0],)


def test_filter_hallucinations_first_segment_never_dropped():
    segments = (_seg(0.0, 5.0, "Thank you.", no_speech_prob=0.95),)
    kept, dropped = backends.filter_hallucinations(segments)
    assert dropped == 0
    assert kept == segments


def test_apply_hallucination_filter_recomputes_text_and_word_count_and_sets_field(capsys):
    result = TranscriptionResult(
        language="en",
        word_count=4,
        text="Thank you. Thank you.",
        segments=(
            _seg(0.0, 5.0, "Thank you.", no_speech_prob=0.05),
            _seg(5.0, 8.0, "Thank you.", no_speech_prob=0.9),
        ),
    )
    filtered = backends._apply_hallucination_filter(result)

    assert filtered.hallucinations_filtered == 1
    assert len(filtered.segments) == 1
    assert filtered.text == "Thank you."
    assert filtered.word_count == 2

    out = capsys.readouterr().out
    assert "filtered 1 hallucination-suspect segment(s)" in out


def test_apply_hallucination_filter_noop_when_nothing_dropped():
    result = TranscriptionResult(
        language="en",
        word_count=2,
        text="hello world",
        segments=(_seg(0.0, 1.0, "hello world", no_speech_prob=0.05),),
    )
    filtered = backends._apply_hallucination_filter(result)
    assert filtered is result


# ── Backends apply the filter (openai + mlx) ────────────────────────────────────

class _FakeOpenaiModelWithHallucination:
    def transcribe(self, path, **opts):
        return {
            "language": "en",
            "text": " Thank you. Thank you.",
            "segments": [
                {
                    "start": 0.0, "end": 5.0, "text": " Thank you.",
                    "avg_logprob": -0.2, "no_speech_prob": 0.05,
                    "words": [{"word": " Thank", "start": 0.0, "end": 0.5, "probability": 0.9},
                              {"word": " you.", "start": 0.5, "end": 1.0, "probability": 0.9}],
                },
                {
                    "start": 5.0, "end": 8.0, "text": " Thank you.",
                    "avg_logprob": -0.2, "no_speech_prob": 0.9,
                    "words": [{"word": " Thank", "start": 5.0, "end": 5.5, "probability": 0.5},
                              {"word": " you.", "start": 5.5, "end": 6.0, "probability": 0.5}],
                },
            ],
        }


def test_openai_backend_applies_hallucination_filter(monkeypatch, capsys):
    monkeypatch.setattr("shumary.modelhub.whisper_model", lambda name: _FakeOpenaiModelWithHallucination())

    backend = backends.OpenaiBackend()
    result = backend.transcribe(Path("/tmp/audio.wav"), "tiny", fast=False)

    assert len(result.segments) == 1
    assert result.hallucinations_filtered == 1
    out = capsys.readouterr().out
    assert "filtered 1 hallucination-suspect segment(s)" in out


def test_mlx_backend_applies_hallucination_filter(monkeypatch):
    import sys
    import types

    def _fake_transcribe(path, **opts):
        return {
            "language": "en",
            "text": " Thank you. Thank you.",
            "segments": [
                {
                    "start": 0.0, "end": 5.0, "text": " Thank you.",
                    "avg_logprob": -0.2, "no_speech_prob": 0.05,
                    "words": [],
                },
                {
                    "start": 5.0, "end": 8.0, "text": " Thank you.",
                    "avg_logprob": -0.2, "no_speech_prob": 0.9,
                    "words": [],
                },
            ],
        }

    fake_module = types.ModuleType("mlx_whisper")
    fake_module.transcribe = _fake_transcribe
    monkeypatch.setitem(sys.modules, "mlx_whisper", fake_module)

    backend = backends.MlxBackend()
    result = backend.transcribe(Path("/tmp/audio.wav"), "tiny", fast=False)

    assert len(result.segments) == 1
    assert result.hallucinations_filtered == 1


def test_faster_backend_does_not_apply_post_filter(monkeypatch):
    """faster-whisper relies on its built-in vad_filter, not the post-filter."""
    captured = {}

    def _gen():
        yield _FakeFasterSegment(
            0.0, 5.0, " Thank you.", -0.2, 0.05,
            words=[],
        )
        yield _FakeFasterSegment(
            5.0, 8.0, " Thank you.", -0.2, 0.9,
            words=[],
        )

    fake_model = _FakeFasterModel(captured)
    fake_model.transcribe = lambda path, **opts: (_gen(), _FakeFasterInfo(language="en"))
    monkeypatch.setattr(backends, "_faster_model", lambda name: fake_model)

    backend = backends.FasterBackend()
    result = backend.transcribe(Path("/tmp/audio.wav"), "tiny", fast=False)

    assert len(result.segments) == 2  # not filtered
    assert result.hallucinations_filtered == 0


# ── tqdm progress redirect ────────────────────────────────────────────────────
#
# mlx-whisper and openai-whisper expose no progress callback; both drive a tqdm
# counter internally, and redirecting it is the only real progress signal they
# offer. These tests stand in for that library code with a plain tqdm user.

def test_tqdm_reporter_forwards_updates_as_fractions():
    seen: list[float] = []
    with backends._tqdm_reporter(seen.append):
        import tqdm
        with tqdm.tqdm(total=10) as bar:
            for _ in range(10):
                bar.update(1)

    assert seen[0] == pytest.approx(0.1)
    assert seen[-1] == pytest.approx(1.0)
    assert seen == sorted(seen), "fractions must never go backwards"


def test_tqdm_reporter_draws_nothing():
    """The bar must not be rendered: stderr carries the JSON progress stream,
    and a half-drawn bar interleaved with it would corrupt the events the app
    parses."""
    import io as _io
    import contextlib

    err = _io.StringIO()
    with contextlib.redirect_stderr(err):
        with backends._tqdm_reporter(lambda f: None):
            import tqdm
            with tqdm.tqdm(total=4) as bar:
                bar.update(4)

    assert err.getvalue() == ""


def test_tqdm_reporter_restores_the_real_tqdm():
    import tqdm
    original = tqdm.tqdm

    with backends._tqdm_reporter(lambda f: None):
        assert tqdm.tqdm is not original

    assert tqdm.tqdm is original


def test_tqdm_reporter_restores_even_when_the_body_raises():
    import tqdm
    original = tqdm.tqdm

    with pytest.raises(ValueError):
        with backends._tqdm_reporter(lambda f: None):
            raise ValueError("transcription blew up")

    assert tqdm.tqdm is original


def test_tqdm_reporter_without_a_sink_leaves_tqdm_alone():
    import tqdm
    original = tqdm.tqdm

    with backends._tqdm_reporter(None):
        assert tqdm.tqdm is original


def test_tqdm_reporter_survives_a_failing_sink():
    """Progress reporting must never take down a transcription."""
    def _boom(_fraction):
        raise RuntimeError("sink exploded")

    with backends._tqdm_reporter(_boom):
        import tqdm
        with tqdm.tqdm(total=2) as bar:
            bar.update(1)
            bar.update(1)

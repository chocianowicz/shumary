"""Tests for stages/diarize.py — DiarizationResult serialisation and backend attribution."""

from pathlib import Path

from shumary.models import DiarizationResult, Turn
from shumary.stages import diarize


def test_diarization_result_roundtrip():
    result = DiarizationResult(
        turns=(
            Turn(start=0.0, end=4.0, speaker="SPEAKER_00"),
            Turn(start=4.0, end=9.0, speaker="SPEAKER_01"),
        ),
        backend="resemblyzer",
    )
    d = result.to_dict()
    restored = DiarizationResult.from_dict(d)
    assert restored == result
    assert d["backend"] == "resemblyzer"


def test_diarization_result_old_cache_without_backend_is_unknown():
    """Old cache entries written before backend attribution lack the field."""
    old_cache_entry = {
        "turns": [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}],
    }
    restored = DiarizationResult.from_dict(old_cache_entry)
    assert restored.backend == "unknown"
    assert restored.turns[0].speaker == "SPEAKER_00"


def test_speaker_stats_from_turns():
    turns = [
        Turn(start=0.0, end=3.0, speaker="SPEAKER_00"),
        Turn(start=3.0, end=10.0, speaker="SPEAKER_01"),
    ]
    stats = diarize.speaker_stats(turns)
    assert stats is not None
    assert stats.count == 2


def test_speaker_stats_none_for_no_turns():
    assert diarize.speaker_stats(None) is None
    assert diarize.speaker_stats([]) is None


def test_run_falls_back_to_resemblyzer_when_pyannote_fails_at_runtime(monkeypatch):
    """Pyannote loads successfully but raises during diarization → resemblyzer runs instead."""
    sentinel_pipeline = object()

    monkeypatch.setattr(diarize, "diarization_pipeline", lambda: sentinel_pipeline)
    monkeypatch.setattr(diarize, "_run_pyannote", lambda audio, pipeline: None)

    fallback_turns = [Turn(start=0.0, end=5.0, speaker="SPEAKER_00")]
    monkeypatch.setattr(diarize, "_run_resemblyzer", lambda audio: fallback_turns)

    result = diarize.run(Path("/tmp/fake.wav"))

    assert result is not None
    assert result.backend == "resemblyzer"
    assert result.turns == tuple(fallback_turns)


def test_run_returns_none_when_pyannote_and_resemblyzer_both_fail(monkeypatch):
    sentinel_pipeline = object()

    monkeypatch.setattr(diarize, "diarization_pipeline", lambda: sentinel_pipeline)
    monkeypatch.setattr(diarize, "_run_pyannote", lambda audio, pipeline: None)
    monkeypatch.setattr(diarize, "_run_resemblyzer", lambda audio: None)

    assert diarize.run(Path("/tmp/fake.wav")) is None


def test_silhouette_for_k_low_for_single_cluster_data():
    """Points drawn from one tight cluster: a forced 2-way split should score
    low (well below SOLO_SILHOUETTE) — this is the "solo speaker" signal."""
    import numpy as np
    from scipy.cluster.hierarchy import linkage

    rng = np.random.default_rng(42)
    # 40 points tightly clustered around one center — simulates one speaker
    E = rng.normal(loc=0.0, scale=0.01, size=(40, 8))
    Z = linkage(E, method="ward")

    sil2 = diarize._silhouette_for_k(E, Z, 2)

    assert sil2 is not None
    assert sil2 < diarize.SOLO_SILHOUETTE


def test_silhouette_for_k_high_for_two_separated_clusters():
    """Points drawn from two well-separated clusters: a forced 2-way split
    should score well above SOLO_SILHOUETTE — the "duo speaker" signal."""
    import numpy as np
    from scipy.cluster.hierarchy import linkage

    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=0.0, scale=0.01, size=(20, 8))
    cluster_b = rng.normal(loc=5.0, scale=0.01, size=(20, 8))
    E = np.vstack([cluster_a, cluster_b])
    Z = linkage(E, method="ward")

    sil2 = diarize._silhouette_for_k(E, Z, 2)

    assert sil2 is not None
    assert sil2 >= 2 * diarize.SOLO_SILHOUETTE


def test_silhouette_for_k_none_for_too_few_points():
    import numpy as np
    from scipy.cluster.hierarchy import linkage

    E = np.random.default_rng(0).normal(size=(4, 8))  # < _MIN_SPEAKERS * 4
    Z = linkage(E, method="ward")

    assert diarize._silhouette_for_k(E, Z, 2) is None


def test_run_resemblyzer_returns_single_turn_for_solo_recording(monkeypatch):
    """End-to-end _run_resemblyzer with mocked encoder/audio I/O: a single
    tight cluster of embeddings → one SPEAKER_00 turn spanning the file."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 40
    embeddings = rng.normal(loc=0.0, scale=0.01, size=(n, 8))

    class _FakeSlice:
        def __init__(self, start, stop):
            self.start = start
            self.stop = stop

    sr = 16_000
    wav_splits = [_FakeSlice(i * sr, (i + 1) * sr) for i in range(n)]

    class _FakeEncoder:
        def embed_utterance(self, wav, return_partials, rate, min_coverage):
            return None, embeddings, wav_splits

    monkeypatch.setattr(diarize, "resemblyzer_encoder", lambda: _FakeEncoder())
    monkeypatch.setattr(
        "resemblyzer.preprocess_wav", lambda path: np.zeros(n * sr), raising=False
    )

    turns = diarize._run_resemblyzer(Path("/tmp/fake.wav"))

    assert turns is not None
    assert len(turns) == 1
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[0].start == 0.0
    assert turns[0].end > 0.0


class _FakeAnnotation:
    """Mimics pyannote Annotation.itertracks(yield_label=True)."""

    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        return iter(self._tracks)


class _FakeSegment:
    def __init__(self, start, end):
        self.start = start
        self.end = end


def test_run_pyannote_unwraps_4x_diarize_output():
    """pyannote.audio 4.x returns a DiarizeOutput wrapper whose Annotation
    lives at .speaker_diarization — _run_pyannote must unwrap it."""

    class _FakeDiarizeOutput:
        speaker_diarization = _FakeAnnotation(
            [(_FakeSegment(0.0, 4.5), None, "SPEAKER_00")]
        )

    turns = diarize._run_pyannote(
        Path("/tmp/fake.wav"), lambda path: _FakeDiarizeOutput()
    )

    assert turns == [Turn(start=0.0, end=4.5, speaker="SPEAKER_00")]


def test_run_pyannote_handles_3x_bare_annotation():
    """pyannote.audio 3.x returns the Annotation directly — still supported."""
    annotation = _FakeAnnotation([(_FakeSegment(1.0, 2.0), None, "SPEAKER_01")])

    turns = diarize._run_pyannote(Path("/tmp/fake.wav"), lambda path: annotation)

    assert turns == [Turn(start=1.0, end=2.0, speaker="SPEAKER_01")]


def test_run_uses_pyannote_when_it_succeeds(monkeypatch):
    sentinel_pipeline = object()
    pyannote_turns = [Turn(start=0.0, end=5.0, speaker="SPEAKER_00")]

    monkeypatch.setattr(diarize, "diarization_pipeline", lambda: sentinel_pipeline)
    monkeypatch.setattr(diarize, "_run_pyannote", lambda audio, pipeline: pyannote_turns)

    def _fail_resemblyzer(audio):
        raise AssertionError("resemblyzer should not run when pyannote succeeds")

    monkeypatch.setattr(diarize, "_run_resemblyzer", _fail_resemblyzer)

    result = diarize.run(Path("/tmp/fake.wav"))

    assert result is not None
    assert result.backend == "pyannote"
    assert result.turns == tuple(pyannote_turns)

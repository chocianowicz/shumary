"""Tests for stages/diarize.py — DiarizationResult serialisation and backend attribution."""

from pathlib import Path

from shumary.models import AlignedSegment, DiarizationResult, Turn
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
    monkeypatch.setattr(diarize, "_run_pyannote", lambda audio, pipeline, progress=None: None)

    fallback_turns = [Turn(start=0.0, end=5.0, speaker="SPEAKER_00")]
    monkeypatch.setattr(diarize, "_run_resemblyzer", lambda audio: fallback_turns)

    result = diarize.run(Path("/tmp/fake.wav"))

    assert result is not None
    assert result.backend == "resemblyzer"
    assert result.turns == tuple(fallback_turns)


def test_run_returns_none_when_pyannote_and_resemblyzer_both_fail(monkeypatch):
    sentinel_pipeline = object()

    monkeypatch.setattr(diarize, "diarization_pipeline", lambda: sentinel_pipeline)
    monkeypatch.setattr(diarize, "_run_pyannote", lambda audio, pipeline, progress=None: None)
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
    monkeypatch.setattr(diarize, "_run_pyannote", lambda audio, pipeline, progress=None: pyannote_turns)

    def _fail_resemblyzer(audio):
        raise AssertionError("resemblyzer should not run when pyannote succeeds")

    monkeypatch.setattr(diarize, "_run_resemblyzer", _fail_resemblyzer)

    result = diarize.run(Path("/tmp/fake.wav"))

    assert result is not None
    assert result.backend == "pyannote"
    assert result.turns == tuple(pyannote_turns)


# ── speaker stats derived from transcribed speech ──────────────────────────────

def _seg(start, end, speaker, text="words here"):
    return AlignedSegment(
        start=start, end=end, text=text,
        avg_logprob=-0.2, no_speech_prob=0.05, speaker=speaker,
    )


def test_speaker_stats_from_speech_measures_attributed_speech_not_turns():
    """The mic label's turns come from an energy gate, so they can be many times
    longer than what the speaker actually said. The reported share must follow
    the transcript."""
    aligned = [
        _seg(0.0, 2.0, "Me"),
        _seg(2.0, 8.0, "SPEAKER_00"),
    ]
    stats = diarize.speaker_stats_from_speech(aligned)
    assert stats is not None
    by_label = {s.label: s for s in stats.speakers}
    assert by_label["Me"].duration_sec == 2.0
    assert by_label["Me"].pct == 25
    assert by_label["SPEAKER_00"].pct == 75


def test_speaker_stats_from_speech_drops_labels_with_no_speech():
    """A phantom cluster that carried no words is not a participant."""
    aligned = [
        _seg(0.0, 5.0, "SPEAKER_00"),
        _seg(5.0, 5.2, "SPEAKER_01", text="   "),   # whitespace only
    ]
    stats = diarize.speaker_stats_from_speech(aligned)
    assert stats.count == 1
    assert [s.label for s in stats.speakers] == ["SPEAKER_00"]


def test_speaker_stats_from_speech_ignores_unattributed_labels():
    aligned = [
        _seg(0.0, 4.0, "SPEAKER_00"),
        _seg(4.0, 6.0, "SPEAKER_??"),
        _seg(6.0, 7.0, None),
    ]
    stats = diarize.speaker_stats_from_speech(aligned)
    assert stats.count == 1
    assert stats.speakers[0].pct == 100


def test_speaker_stats_from_speech_does_not_double_count_overlapping_segments():
    """Segments from different Whisper segments can abut or overlap slightly;
    nobody should be credited twice for the same moment."""
    aligned = [
        _seg(0.0, 3.0, "SPEAKER_00"),
        _seg(2.0, 4.0, "SPEAKER_00"),   # 1s of overlap with the previous
    ]
    stats = diarize.speaker_stats_from_speech(aligned)
    assert stats.speakers[0].duration_sec == 4.0


def test_speaker_stats_from_speech_none_without_segments():
    assert diarize.speaker_stats_from_speech(None) is None
    assert diarize.speaker_stats_from_speech([]) is None


# ── pruning turns with no transcribed speech ──────────────────────────────────

def test_prune_unvoiced_turns_drops_turns_alignment_gave_no_words():
    turns = [
        Turn(start=0.0, end=2.0, speaker="Me"),      # real speech
        Turn(start=4.0, end=9.0, speaker="Me"),      # keyboard: no words landed
        Turn(start=2.0, end=4.0, speaker="SPEAKER_00"),
    ]
    aligned = [_seg(0.0, 2.0, "Me"), _seg(2.0, 4.0, "SPEAKER_00")]

    kept = diarize.prune_unvoiced_turns(turns, aligned)
    assert [(t.start, t.speaker) for t in kept] == [(0.0, "Me"), (2.0, "SPEAKER_00")]


def test_prune_unvoiced_turns_matches_speaker_as_well_as_time():
    """A turn is not voiced just because *somebody* spoke during it — this is
    exactly the mic-turn-over-remote-speech case."""
    turns = [
        Turn(start=0.0, end=5.0, speaker="Me"),          # mic energy over their speech
        Turn(start=0.0, end=5.0, speaker="SPEAKER_00"),  # who actually spoke
    ]
    aligned = [_seg(0.0, 5.0, "SPEAKER_00")]

    kept = diarize.prune_unvoiced_turns(turns, aligned)
    assert [t.speaker for t in kept] == ["SPEAKER_00"]


def test_prune_unvoiced_turns_keeps_everything_rather_than_return_nothing():
    """If no turn has evidence, the evidence is missing — not every turn spurious."""
    turns = [Turn(start=0.0, end=5.0, speaker="Me")]
    aligned = [_seg(9.0, 10.0, "SPEAKER_00")]

    assert diarize.prune_unvoiced_turns(turns, aligned) == turns


def test_prune_unvoiced_turns_keeps_short_backchannel():
    """250 ms of genuine speech must survive — backchannel is what makes the
    turn-taking metrics meaningful."""
    turns = [Turn(start=1.0, end=1.25, speaker="Me")]
    aligned = [_seg(1.0, 1.25, "Me", text="mhm")]

    assert diarize.prune_unvoiced_turns(turns, aligned) == turns


def test_prune_unvoiced_turns_passes_through_without_evidence():
    turns = [Turn(start=0.0, end=2.0, speaker="Me")]
    assert diarize.prune_unvoiced_turns(turns, None) == turns
    assert diarize.prune_unvoiced_turns(turns, []) == turns
    # No aligned speech at all is missing evidence, not proof of silence.
    assert diarize.prune_unvoiced_turns(turns, [_seg(0.0, 1.0, None)]) == turns


def test_speaker_stats_from_speech_drops_single_word_phantom_cluster():
    """Diarization sometimes emits a cluster holding one low-confidence word.
    Counting it as a participant corrupts the headline speaker count."""
    aligned = [
        _seg(0.0, 60.0, "SPEAKER_00"),
        _seg(60.0, 120.0, "SPEAKER_01"),
        _seg(120.0, 121.4, "SPEAKER_02", text="ага"),   # 1.4s — below the floor
    ]
    stats = diarize.speaker_stats_from_speech(aligned)
    assert stats.count == 2
    assert "SPEAKER_02" not in {s.label for s in stats.speakers}


def test_prune_unvoiced_turns_drops_turns_of_disqualified_labels():
    """A label dropped from the speaker table must stop contributing turn
    metrics too, or it still shows up as interruptions and latencies."""
    turns = [
        Turn(start=0.0, end=10.0, speaker="SPEAKER_00"),
        Turn(start=10.0, end=11.4, speaker="SPEAKER_02"),
    ]
    aligned = [_seg(0.0, 10.0, "SPEAKER_00"), _seg(10.0, 11.4, "SPEAKER_02", text="ага")]

    kept = diarize.prune_unvoiced_turns(turns, aligned, keep_labels={"SPEAKER_00"})
    assert [t.speaker for t in kept] == ["SPEAKER_00"]

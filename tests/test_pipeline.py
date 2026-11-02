"""Tests for pipeline.py orchestration with faked stages."""

from pathlib import Path

import pytest

from shumary import cache as cache_mod
from shumary import pipeline
from shumary.models import (
    AlignedSegment,
    DiarizationResult,
    MoodSignals,
    ProsodyResult,
    Segment,
    SpeakerProsody,
    TranscriptionResult,
    Turn,
)


def _fake_transcription():
    return TranscriptionResult(
        language="en",
        word_count=4,
        text="hello world test again",
        segments=(Segment(0.0, 2.0, "hello world", -0.1, 0.05),
                  Segment(2.0, 4.0, "test again", -0.1, 0.05)),
    )


def _fake_mood():
    return MoodSignals(
        avg_speaking_rate_wps=2.0,
        rate_variability=None,
        long_pauses=0,
        long_pause_details=(),
        filler_word_count=0,
        top_fillers=(),
        repeated_phrases=(),
        low_confidence_segments=0,
        near_silence_segments=0,
        inferred_state="(uncertain) baseline",
    )


@pytest.fixture(autouse=True)
def _patch_pipeline(tmp_path, monkeypatch):
    """Redirect cache + paths into tmp_path and stub out heavy stages."""
    cache_dir = tmp_path / ".cache"
    monkeypatch.setattr("shumary.paths.CACHE", cache_dir)
    monkeypatch.setattr("shumary.cache.cache_dir_for", lambda slug: cache_dir / slug)

    monkeypatch.setattr(pipeline.audio_stage, "validate", lambda audio: None)
    monkeypatch.setattr(pipeline.audio_stage, "duration_seconds", lambda audio: 4.0)
    monkeypatch.setattr(pipeline.audio_stage, "decode_to_wav", lambda audio, slug: audio)
    monkeypatch.setattr(pipeline.audio_stage, "wav_duration_seconds", lambda wav: 4.0)

    monkeypatch.setattr(
        pipeline.transcribe_stage, "run",
        lambda audio, model_name, fast=False, language=None, backend=None, progress=None: _fake_transcription(),
    )
    monkeypatch.setattr(pipeline, "pick_backend", lambda override=None: None)
    def _fake_align(segments, turns):
        """Stand-in for align: assign each segment to its max-overlap turn.

        Cruder than the real stage (no word-level splitting), but it must at
        least route different segments to different speakers — downstream stages
        now derive the speaker table and turn metrics from what alignment
        actually attributed, so a stub that credits every segment to `turns[0]`
        would make every multi-speaker fixture look like a monologue.
        """
        def _speaker(seg):
            if not turns:
                return None
            best = max(
                turns,
                key=lambda t: max(0.0, min(seg.end, t.end) - max(seg.start, t.start)),
            )
            overlap = max(0.0, min(seg.end, best.end) - max(seg.start, best.start))
            return best.speaker if overlap > 0 else turns[0].speaker

        return [
            AlignedSegment(s.start, s.end, s.text, s.avg_logprob, s.no_speech_prob,
                           speaker=_speaker(s))
            for s in segments
        ]

    monkeypatch.setattr(pipeline.align_stage, "run", _fake_align)
    monkeypatch.setattr(pipeline.mood_stage, "run", lambda aligned, language, speakers=None: _fake_mood())
    monkeypatch.setattr(pipeline.prosody_stage, "available", lambda: False)
    monkeypatch.setattr(pipeline.render_stage, "save_transcript", lambda **kwargs: kwargs["out_path"])

    return tmp_path


def test_no_diarize_does_not_write_cache(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")

    pipeline.process_file(audio, no_diarize=True)

    slug = "memo"
    assert not cache_mod.is_cached(slug, "diarize")


def test_solo_flag_skips_diarization_with_single_speaker(tmp_path, monkeypatch):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")

    def _fail_diarize(*args, **kwargs):
        raise AssertionError("diarize_stage.run should not be called with --solo")

    monkeypatch.setattr(pipeline.diarize_stage, "run", _fail_diarize)

    captured = {}

    def _fake_render(**kwargs):
        captured["speaker_stats"] = kwargs["speaker_stats"]
        captured["diarize_backend"] = kwargs["diarize_backend"]
        return kwargs["out_path"]

    monkeypatch.setattr(pipeline.render_stage, "save_transcript", _fake_render)

    pipeline.process_file(audio, solo=True)

    slug = "memo"
    assert not cache_mod.is_cached(slug, "diarize")

    speaker_stats = captured["speaker_stats"]
    assert speaker_stats is not None
    assert speaker_stats.count == 1
    assert speaker_stats.speakers[0].label == "SPEAKER_00"
    assert captured["diarize_backend"] == "user-flag"


def test_process_directory_continues_past_failures(tmp_path, monkeypatch, capsys):
    src = tmp_path / "recordings"
    src.mkdir()

    good = src / "good.m4a"
    good.write_text("fake-audio")
    bad = src / "bad.m4a"
    bad.write_text("fake-audio")

    def _validate(audio):
        if audio.name == "bad.m4a":
            from shumary.errors import ShumaryError
            raise ShumaryError("boom")

    monkeypatch.setattr(pipeline.audio_stage, "validate", _validate)

    succeeded, failed = pipeline.process_directory(src, no_diarize=True)

    assert len(succeeded) == 1
    assert succeeded[0].name == "good.md"
    assert [p.name for p in failed] == ["bad.m4a"]

    out = capsys.readouterr().out
    assert "1 succeeded / 1 failed" in out


def test_no_diarize_does_not_block_later_normal_run(tmp_path, monkeypatch):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")
    slug = "memo"

    diarize_calls = []

    def _fake_diarize_run(audio, progress=None):
        diarize_calls.append(audio)
        return DiarizationResult(turns=(Turn(0.0, 4.0, "SPEAKER_00"),), backend="resemblyzer")

    monkeypatch.setattr(pipeline.diarize_stage, "run", _fake_diarize_run)

    # First run with --no-diarize: nothing written to diarize cache.
    pipeline.process_file(audio, no_diarize=True)
    assert not cache_mod.is_cached(slug, "diarize")

    # Second, normal run: diarization actually executes and is cached.
    pipeline.process_file(audio, no_diarize=False)
    assert cache_mod.is_cached(slug, "diarize")
    assert len(diarize_calls) == 1

    cached = cache_mod.read(slug, "diarize")
    assert cached["backend"] == "resemblyzer"


def test_output_path_defaults_next_to_input(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")

    out_path = pipeline.process_file(audio, no_diarize=True)

    assert out_path.name == "memo.md"


def test_output_path_honors_explicit_output(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")
    dest = tmp_path / "custom.md"

    out_path = pipeline.process_file(audio, no_diarize=True, output=dest)

    assert out_path == dest


def test_process_file_writes_manifest_on_first_run(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")
    slug = "memo"

    pipeline.process_file(audio, no_diarize=True, model_name="large", fast=False)

    manifest = cache_mod.read_manifest(slug)
    assert manifest is not None
    assert manifest["model_name"] == "large"
    assert manifest["fast"] is False
    assert manifest["pipeline_version"] == cache_mod.PIPELINE_VERSION


def test_process_file_model_change_invalidates_whisper_and_descendants(tmp_path, capsys):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")
    slug = "memo"

    pipeline.process_file(audio, no_diarize=True, model_name="large", fast=False)
    assert cache_mod.is_cached(slug, "whisper")
    assert cache_mod.is_cached(slug, "align")
    assert cache_mod.is_cached(slug, "mood")

    capsys.readouterr()  # discard first-run output

    pipeline.process_file(audio, no_diarize=True, model_name="turbo", fast=False)

    out = capsys.readouterr().out
    assert "cache invalidated" in out
    assert "large" in out and "turbo" in out

    manifest = cache_mod.read_manifest(slug)
    assert manifest["model_name"] == "turbo"


def test_process_file_different_audio_same_slug_raises(tmp_path):
    from shumary.errors import ShumaryError

    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio-one")

    pipeline.process_file(audio, no_diarize=True)

    # Re-create the audio with different content but the same slug/name.
    audio.write_text("a totally different recording with different bytes")

    with pytest.raises(ShumaryError):
        pipeline.process_file(audio, no_diarize=True)


def test_process_file_legacy_cache_without_manifest_is_invalidated(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")
    slug = "memo"

    # Simulate a v1 cache: stage files present, no manifest.
    cache_mod.write(slug, "whisper", {"language": "en", "word_count": 1, "text": "hi", "segments": []})
    cache_mod.write(slug, "align", {"segments": []})
    cache_mod.write(slug, "mood", {})
    cache_mod.mark_done(slug)
    assert cache_mod.read_manifest(slug) is None

    pipeline.process_file(audio, no_diarize=True)

    manifest = cache_mod.read_manifest(slug)
    assert manifest is not None
    assert manifest["pipeline_version"] == cache_mod.PIPELINE_VERSION


# ── Interaction stage (5.2) ───────────────────────────────────────────────────────

def test_single_speaker_does_not_run_interaction_stage(tmp_path, monkeypatch):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")
    slug = "memo"

    monkeypatch.setattr(
        pipeline.diarize_stage, "run",
        lambda audio, progress=None: DiarizationResult(turns=(Turn(0.0, 4.0, "SPEAKER_00"),), backend="resemblyzer"),
    )

    captured = {}

    def _fake_render(**kwargs):
        captured["interaction"] = kwargs["interaction"]
        return kwargs["out_path"]

    monkeypatch.setattr(pipeline.render_stage, "save_transcript", _fake_render)

    pipeline.process_file(audio, no_diarize=False)

    assert not cache_mod.is_cached(slug, "interaction")
    assert captured["interaction"] is None


def test_multi_speaker_runs_and_caches_interaction_stage(tmp_path, monkeypatch):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")
    slug = "memo"

    turns = (
        Turn(0.0, 2.0, "SPEAKER_00"),
        Turn(2.0, 4.0, "SPEAKER_01"),
    )
    monkeypatch.setattr(
        pipeline.diarize_stage, "run",
        lambda audio, progress=None: DiarizationResult(turns=turns, backend="pyannote"),
    )

    captured = {}

    def _fake_render(**kwargs):
        captured["interaction"] = kwargs["interaction"]
        return kwargs["out_path"]

    monkeypatch.setattr(pipeline.render_stage, "save_transcript", _fake_render)

    pipeline.process_file(audio, no_diarize=False)

    assert cache_mod.is_cached(slug, "interaction")
    interaction = captured["interaction"]
    assert interaction is not None
    assert interaction.backend == "pyannote"
    assert {s.speaker for s in interaction.speakers} == {"SPEAKER_00", "SPEAKER_01"}


def test_interaction_cache_invalidated_when_diarize_recomputed(tmp_path, monkeypatch):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")
    slug = "memo"

    turns = (
        Turn(0.0, 2.0, "SPEAKER_00"),
        Turn(2.0, 4.0, "SPEAKER_01"),
    )
    monkeypatch.setattr(
        pipeline.diarize_stage, "run",
        lambda audio, progress=None: DiarizationResult(turns=turns, backend="pyannote"),
    )
    monkeypatch.setattr(pipeline.render_stage, "save_transcript", lambda **kwargs: kwargs["out_path"])

    pipeline.process_file(audio, no_diarize=False)
    assert cache_mod.is_cached(slug, "interaction")

    # Recomputing diarize must drop the now-stale interaction cache.
    cache_mod.write(slug, "diarize", DiarizationResult(turns=turns, backend="pyannote").to_dict())
    assert not cache_mod.is_cached(slug, "interaction")


# ── Prosody stage (Phase 7.1) ───────────────────────────────────────────────────

def _fake_prosody_result():
    return ProsodyResult(
        segments=(),
        speakers=(
            SpeakerProsody(
                speaker="speaker",
                f0_baseline_hz=142.3,
                f0_variability_st=2.1,
                energy_mean_db=-23.4,
                energy_dynamics_db=8.7,
                jitter_local=0.012,
                shimmer_local=0.061,
                f0_trend_by_third_st=(0.0, -0.4, -1.1),
            ),
        ),
    )


def test_prosody_runs_and_caches_when_available(tmp_path, monkeypatch):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")
    slug = "memo"

    monkeypatch.setattr(pipeline.prosody_stage, "available", lambda: True)
    monkeypatch.setattr(pipeline.prosody_stage, "run", lambda wav, segments, speakers=None: _fake_prosody_result())

    captured = {}

    def _fake_render(**kwargs):
        captured["prosody"] = kwargs["prosody"]
        return kwargs["out_path"]

    monkeypatch.setattr(pipeline.render_stage, "save_transcript", _fake_render)

    pipeline.process_file(audio, no_diarize=True)

    assert cache_mod.is_cached(slug, "prosody")
    assert captured["prosody"] is not None
    assert captured["prosody"].speakers[0].speaker == "speaker"


def test_prosody_skipped_and_uncached_when_unavailable(tmp_path, monkeypatch):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")
    slug = "memo"

    monkeypatch.setattr(pipeline.prosody_stage, "available", lambda: False)

    def _fail_run(*args, **kwargs):
        raise AssertionError("prosody_stage.run should not be called when unavailable")

    monkeypatch.setattr(pipeline.prosody_stage, "run", _fail_run)

    captured = {}

    def _fake_render(**kwargs):
        captured["prosody"] = kwargs["prosody"]
        return kwargs["out_path"]

    monkeypatch.setattr(pipeline.render_stage, "save_transcript", _fake_render)

    pipeline.process_file(audio, no_diarize=True)

    assert not cache_mod.is_cached(slug, "prosody")
    assert captured["prosody"] is None


def test_prosody_cache_invalidated_when_align_recomputed(tmp_path, monkeypatch):
    audio = tmp_path / "memo.m4a"
    audio.write_text("fake-audio")
    slug = "memo"

    monkeypatch.setattr(pipeline.prosody_stage, "available", lambda: True)
    monkeypatch.setattr(pipeline.prosody_stage, "run", lambda wav, segments, speakers=None: _fake_prosody_result())
    monkeypatch.setattr(pipeline.render_stage, "save_transcript", lambda **kwargs: kwargs["out_path"])

    pipeline.process_file(audio, no_diarize=True)
    assert cache_mod.is_cached(slug, "prosody")

    # Recomputing align must drop the now-stale prosody cache.
    cache_mod.write(slug, "align", {"segments": []})
    assert not cache_mod.is_cached(slug, "prosody")


# ── dual-source detection ──────────────────────────────────────────────────────

def test_resolve_mic_label_auto_detects_a_recording(tmp_path):
    from shumary import recorder

    wav = tmp_path / "meeting.wav"
    recorder.write_sidecar(wav, mic_label="Denis")

    assert pipeline._resolve_mic_label(wav, None) == "Denis"


def test_resolve_mic_label_is_none_for_an_ordinary_file(tmp_path):
    assert pipeline._resolve_mic_label(tmp_path / "memo.m4a", None) is None


def test_resolve_mic_label_can_be_forced_without_a_sidecar(tmp_path):
    from shumary import recorder

    assert pipeline._resolve_mic_label(tmp_path / "memo.wav", True) == recorder.MIC_LABEL


def test_resolve_mic_label_can_be_disabled_for_a_recording(tmp_path):
    from shumary import recorder

    wav = tmp_path / "meeting.wav"
    recorder.write_sidecar(wav)

    assert pipeline._resolve_mic_label(wav, False) is None


def test_describe_invalidation_reports_a_dual_source_switch():
    prev = {"model_name": "large", "fast": False, "dual_source": False,
            "pipeline_version": cache_mod.PIPELINE_VERSION}

    described = pipeline._describe_invalidation(prev, "large", False, True, ["diarize"])

    assert "dual-source" in described
    assert "diarize" in described


# ── progress events ────────────────────────────────────────────────────────────

def test_progress_is_monotonic_and_ends_at_one():
    events = []
    prog = pipeline._Progress(events.append)

    for stage in pipeline.STAGE_WEIGHTS:
        prog.start(stage)
        prog.done(stage)
    prog.finished(Path("/tmp/out.md"))

    pcts = [e["pct"] for e in events]
    assert pcts == sorted(pcts), "progress must never go backwards"
    assert pcts[-1] == 1.0
    assert events[-1]["event"] == "done"


def test_progress_weights_sum_to_one():
    assert sum(pipeline.STAGE_WEIGHTS.values()) == pytest.approx(1.0)


def test_progress_is_a_no_op_without_a_sink():
    prog = pipeline._Progress(None)
    prog.start("decode")
    prog.done("decode")
    prog.finished(Path("/tmp/out.md"))  # must not raise


def test_progress_reports_stage_and_status():
    events = []
    prog = pipeline._Progress(events.append)
    prog.start("transcribe")
    prog.done("transcribe")

    assert events[0] == {"event": "stage", "stage": "transcribe", "status": "start", "pct": 0.0}
    assert events[1]["status"] == "done"
    assert events[1]["pct"] == pytest.approx(pipeline.STAGE_WEIGHTS["transcribe"])


def test_progress_failure_event_carries_message():
    events = []
    pipeline._Progress(events.append).failed("boom")
    assert events == [{"event": "error", "message": "boom", "pct": 0.0}]


# ── sub-stage progress ────────────────────────────────────────────────────────

def test_substage_emits_inside_the_stages_own_span():
    events = []
    p = pipeline._Progress(lambda e: events.append(e))
    p.start("decode"); p.done("decode")      # 2%
    p.start("transcribe")
    for f in (0.25, 0.5, 0.75):
        p.substage("transcribe", f)
    pcts = [round(e["pct"] * 100) for e in events if e.get("status") == "progress"]
    # transcribe spans 2%..72%, so quarter-way is ~19.5%, half ~37%, three-quarters ~54.5%
    assert pcts == [20, 37, 55]


def test_substage_never_reaches_the_next_stage():
    events = []
    p = pipeline._Progress(lambda e: events.append(e))
    p.start("decode"); p.done("decode")
    p.substage("transcribe", 1.0)
    assert round(events[-1]["pct"] * 100) == 72     # exactly the boundary, never past
    p.substage("transcribe", 5.0)                   # clamped
    assert round(events[-1]["pct"] * 100) == 72


def test_substage_is_monotonic_and_throttled():
    events = []
    p = pipeline._Progress(lambda e: events.append(e))
    p.start("transcribe")
    p.substage("transcribe", 0.5)
    before = len(events)
    p.substage("transcribe", 0.4)   # backwards — must not emit
    p.substage("transcribe", 0.5)   # same percent — must not emit
    assert len(events) == before


def test_substage_is_silent_without_a_sink():
    p = pipeline._Progress(None)
    p.substage("transcribe", 0.5)   # must not raise


def test_substage_ignores_unknown_stages():
    events = []
    p = pipeline._Progress(lambda e: events.append(e))
    p.substage("not-a-stage", 0.5)
    assert events == []

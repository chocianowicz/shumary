"""Tests for cache.py — disk-backed stage cache."""

import json

import pytest
from shumary import cache as cache_mod
from shumary.errors import ShumaryError


@pytest.fixture(autouse=True)
def patch_cache(tmp_path, monkeypatch):
    """Redirect cache reads/writes to tmp_path."""
    monkeypatch.setattr("shumary.paths.CACHE", tmp_path)
    monkeypatch.setattr("shumary.cache.cache_dir_for", lambda slug: tmp_path / slug)


def test_write_and_read():
    cache_mod.write("slug1", "whisper", {"language": "en", "words": 42})
    result = cache_mod.read("slug1", "whisper")
    assert result == {"language": "en", "words": 42}


def test_is_cached_false_before_write():
    assert not cache_mod.is_cached("nope", "whisper")


def test_is_cached_true_after_write():
    cache_mod.write("slug2", "diarize", {"turns": []})
    assert cache_mod.is_cached("slug2", "diarize")


def test_clear_single_stage_with_no_descendants_leaves_others_intact():
    cache_mod.write("slug3", "whisper", {})
    cache_mod.write("slug3", "mood", {})
    cache_mod.clear("slug3", "mood")
    assert cache_mod.is_cached("slug3", "whisper")
    assert not cache_mod.is_cached("slug3", "mood")


def test_clear_all_stages():
    cache_mod.write("slug4", "whisper", {})
    cache_mod.write("slug4", "align", {})
    cache_mod.clear("slug4")
    assert not cache_mod.is_cached("slug4", "whisper")
    assert not cache_mod.is_cached("slug4", "align")


def test_mark_done_and_is_done():
    assert not cache_mod.is_done("slug5")
    cache_mod.mark_done("slug5")
    assert cache_mod.is_done("slug5")


def test_clear_stage_also_removes_done_sentinel():
    cache_mod.mark_done("slug6")
    cache_mod.clear("slug6", "whisper")  # clear a stage clears DONE too
    assert not cache_mod.is_done("slug6")


def test_list_slugs(tmp_path, monkeypatch):
    monkeypatch.setattr("shumary.cache.list_slugs", lambda: ["a", "b", "c"])
    assert cache_mod.list_slugs() == ["a", "b", "c"]


def test_unicode_payload():
    payload = {"top_fillers": ["ну ×5", 'i don\'t know ×2', '"quoted" ×1']}
    cache_mod.write("slug7", "mood", payload)
    result = cache_mod.read("slug7", "mood")
    assert result == payload


# ── Fingerprint manifest (2.1) ────────────────────────────────────────────────────

def test_compute_audio_hash_deterministic_and_size_sensitive(tmp_path):
    a = tmp_path / "a.m4a"
    b = tmp_path / "b.m4a"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello world")
    assert cache_mod.compute_audio_hash(a) == cache_mod.compute_audio_hash(b)

    c = tmp_path / "c.m4a"
    c.write_bytes(b"hello world!")
    assert cache_mod.compute_audio_hash(a) != cache_mod.compute_audio_hash(c)


def test_ensure_manifest_first_contact_writes_manifest(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"audio-bytes")

    invalidated = cache_mod.ensure_manifest("slug-fresh", audio, "large", False)

    assert invalidated == []
    manifest = cache_mod.read_manifest("slug-fresh")
    assert manifest["audio_hash"] == cache_mod.compute_audio_hash(audio)
    assert manifest["model_name"] == "large"
    assert manifest["fast"] is False
    assert manifest["pipeline_version"] == cache_mod.PIPELINE_VERSION


def test_ensure_manifest_same_params_is_noop(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"audio-bytes")

    cache_mod.ensure_manifest("slug-same", audio, "large", False)
    cache_mod.write("slug-same", "whisper", {"language": "en"})

    invalidated = cache_mod.ensure_manifest("slug-same", audio, "large", False)

    assert invalidated == []
    assert cache_mod.is_cached("slug-same", "whisper")


def test_ensure_manifest_model_change_invalidates_whisper_and_descendants(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"audio-bytes")

    cache_mod.ensure_manifest("slug-model", audio, "large", False)
    cache_mod.write("slug-model", "whisper", {})
    cache_mod.write("slug-model", "diarize", {})
    cache_mod.write("slug-model", "align", {})
    cache_mod.write("slug-model", "mood", {})
    cache_mod.mark_done("slug-model")

    invalidated = cache_mod.ensure_manifest("slug-model", audio, "turbo", False)

    assert set(invalidated) == {"whisper", "align", "mood"}
    assert not cache_mod.is_cached("slug-model", "whisper")
    assert not cache_mod.is_cached("slug-model", "align")
    assert not cache_mod.is_cached("slug-model", "mood")
    # diarize is unaffected by a whisper-stage invalidation
    assert cache_mod.is_cached("slug-model", "diarize")
    assert not cache_mod.is_done("slug-model")

    manifest = cache_mod.read_manifest("slug-model")
    assert manifest["model_name"] == "turbo"


def test_ensure_manifest_fast_flag_change_invalidates_whisper_and_descendants(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"audio-bytes")

    cache_mod.ensure_manifest("slug-fast", audio, "large", False)
    cache_mod.write("slug-fast", "whisper", {})
    cache_mod.write("slug-fast", "align", {})
    cache_mod.write("slug-fast", "mood", {})

    invalidated = cache_mod.ensure_manifest("slug-fast", audio, "large", True)

    assert set(invalidated) == {"whisper", "align", "mood"}
    manifest = cache_mod.read_manifest("slug-fast")
    assert manifest["fast"] is True


def test_ensure_manifest_audio_hash_mismatch_raises(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"original-audio")
    cache_mod.ensure_manifest("slug-mismatch", audio, "large", False)

    audio.write_bytes(b"a completely different file with different content")

    with pytest.raises(ShumaryError):
        cache_mod.ensure_manifest("slug-mismatch", audio, "large", False)


def test_ensure_manifest_legacy_cache_without_manifest_is_invalidated(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"audio-bytes")

    # Simulate a v1 (pre-manifest) cache: stage files exist, no manifest.
    cache_mod.write("slug-legacy", "whisper", {"language": "en"})
    cache_mod.write("slug-legacy", "diarize", {"turns": []})
    cache_mod.write("slug-legacy", "align", {"segments": []})
    cache_mod.write("slug-legacy", "mood", {})
    cache_mod.mark_done("slug-legacy")
    assert cache_mod.read_manifest("slug-legacy") is None

    invalidated = cache_mod.ensure_manifest("slug-legacy", audio, "large", False)

    assert set(invalidated) == {"whisper", "diarize", "align", "mood"}
    assert not cache_mod.is_cached("slug-legacy", "whisper")
    assert not cache_mod.is_cached("slug-legacy", "diarize")
    assert not cache_mod.is_cached("slug-legacy", "align")
    assert not cache_mod.is_cached("slug-legacy", "mood")
    assert not cache_mod.is_done("slug-legacy")

    manifest = cache_mod.read_manifest("slug-legacy")
    assert manifest is not None
    assert manifest["model_name"] == "large"
    assert manifest["pipeline_version"] == cache_mod.PIPELINE_VERSION


def test_ensure_manifest_first_contact_no_existing_files_no_invalidation(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"audio-bytes")

    invalidated = cache_mod.ensure_manifest("slug-brand-new", audio, "large", False)

    assert invalidated == []


def test_descendants_of_whisper_reach_every_derived_stage():
    """New segments change alignment, which changes every stage downstream of it
    — including interaction, whose turns are pruned against aligned speech."""
    assert set(cache_mod.descendants_of("whisper")) == {
        "align", "mood", "prosody", "interaction",
    }


def test_descendants_of_diarize_includes_align_mood_interaction_and_prosody():
    assert set(cache_mod.descendants_of("diarize")) == {"align", "mood", "interaction", "prosody"}


def test_descendants_of_align_includes_mood_prosody_and_interaction():
    assert set(cache_mod.descendants_of("align")) == {"mood", "prosody", "interaction"}


def test_descendants_of_mood_is_empty():
    assert cache_mod.descendants_of("mood") == []


def test_descendants_of_interaction_is_empty():
    assert cache_mod.descendants_of("interaction") == []


def test_descendants_of_prosody_is_empty():
    assert cache_mod.descendants_of("prosody") == []


# ── Dependency-aware invalidation (2.2) ───────────────────────────────────────────

def test_clear_stage_whisper_also_clears_align_and_mood():
    cache_mod.write("slug-clear-whisper", "whisper", {})
    cache_mod.write("slug-clear-whisper", "diarize", {})
    cache_mod.write("slug-clear-whisper", "align", {})
    cache_mod.write("slug-clear-whisper", "mood", {})
    cache_mod.mark_done("slug-clear-whisper")

    cache_mod.clear("slug-clear-whisper", "whisper")

    assert not cache_mod.is_cached("slug-clear-whisper", "whisper")
    assert not cache_mod.is_cached("slug-clear-whisper", "align")
    assert not cache_mod.is_cached("slug-clear-whisper", "mood")
    assert not cache_mod.is_done("slug-clear-whisper")
    # diarize is not a descendant of whisper
    assert cache_mod.is_cached("slug-clear-whisper", "diarize")


def test_clear_stage_diarize_also_clears_align_and_mood():
    cache_mod.write("slug-clear-diarize", "whisper", {})
    cache_mod.write("slug-clear-diarize", "diarize", {})
    cache_mod.write("slug-clear-diarize", "align", {})
    cache_mod.write("slug-clear-diarize", "mood", {})

    cache_mod.clear("slug-clear-diarize", "diarize")

    assert not cache_mod.is_cached("slug-clear-diarize", "diarize")
    assert not cache_mod.is_cached("slug-clear-diarize", "align")
    assert not cache_mod.is_cached("slug-clear-diarize", "mood")
    assert cache_mod.is_cached("slug-clear-diarize", "whisper")


def test_write_drops_descendant_caches_and_done_sentinel():
    cache_mod.write("slug-write-drop", "whisper", {})
    cache_mod.write("slug-write-drop", "align", {})
    cache_mod.write("slug-write-drop", "mood", {})
    cache_mod.mark_done("slug-write-drop")

    # Recomputing whisper (e.g. after a manifest invalidation + --force)
    # must drop the now-stale align/mood and the DONE sentinel.
    cache_mod.write("slug-write-drop", "whisper", {"language": "en"})

    assert cache_mod.is_cached("slug-write-drop", "whisper")
    assert not cache_mod.is_cached("slug-write-drop", "align")
    assert not cache_mod.is_cached("slug-write-drop", "mood")
    assert not cache_mod.is_done("slug-write-drop")


# ── cache list fingerprint info (2.3) ─────────────────────────────────────────────

def test_fingerprint_info_legacy_when_no_manifest():
    cache_mod.write("slug-fp-legacy", "whisper", {})
    assert cache_mod.fingerprint_info("slug-fp-legacy") == "v1/legacy"


def test_fingerprint_info_no_cache_dir_at_all():
    assert cache_mod.fingerprint_info("slug-fp-nonexistent") == "v1/legacy"


def test_fingerprint_info_shows_model_and_short_hash(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"audio-bytes")
    cache_mod.ensure_manifest("slug-fp-fresh", audio, "large", False)

    info = cache_mod.fingerprint_info("slug-fp-fresh")

    full_hash = cache_mod.compute_audio_hash(audio)
    assert info == f"large · {full_hash[:8]}"


def test_fingerprint_info_shows_fast_flag(tmp_path):
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"audio-bytes")
    cache_mod.ensure_manifest("slug-fp-fast", audio, "turbo", True)

    info = cache_mod.fingerprint_info("slug-fp-fast")

    assert info.startswith("turbo --fast · ")


def test_write_leaf_stage_does_not_touch_unrelated_stages():
    cache_mod.write("slug-write-leaf", "whisper", {})
    cache_mod.write("slug-write-leaf", "align", {})
    cache_mod.write("slug-write-leaf", "mood", {"inferred_state": "x"})

    cache_mod.write("slug-write-leaf", "mood", {"inferred_state": "y"})

    assert cache_mod.is_cached("slug-write-leaf", "whisper")
    assert cache_mod.is_cached("slug-write-leaf", "align")
    assert cache_mod.read("slug-write-leaf", "mood") == {"inferred_state": "y"}


# ── dual-source manifest reconciliation ────────────────────────────────────────

def test_manifest_records_dual_source_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "cache_dir_for", lambda slug: tmp_path / slug)

    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio-bytes")
    cache_mod.ensure_manifest("meeting", audio, "large", False, True)

    assert cache_mod.read_manifest("meeting")["dual_source"] is True


def test_switching_dual_source_invalidates_diarize_not_whisper(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "cache_dir_for", lambda slug: tmp_path / slug)

    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio-bytes")
    cache_mod.ensure_manifest("meeting", audio, "large", False, False)

    cache_mod.write("meeting", "whisper", {"ok": True})
    cache_mod.write("meeting", "diarize", {"ok": True})

    invalidated = cache_mod.ensure_manifest("meeting", audio, "large", False, True)

    assert "diarize" in invalidated
    assert cache_mod.is_cached("meeting", "whisper")
    assert not cache_mod.is_cached("meeting", "diarize")


def test_unchanged_dual_source_flag_is_a_no_op(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "cache_dir_for", lambda slug: tmp_path / slug)

    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"audio-bytes")
    cache_mod.ensure_manifest("meeting", audio, "large", False, True)
    cache_mod.write("meeting", "diarize", {"ok": True})

    assert cache_mod.ensure_manifest("meeting", audio, "large", False, True) == []
    assert cache_mod.is_cached("meeting", "diarize")


def test_mark_done_removes_split_channel_wavs(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "cache_dir_for", lambda slug: tmp_path / slug)

    d = tmp_path / "meeting"
    d.mkdir()
    for name in ("audio.wav", "left.wav", "right.wav"):
        (d / name).write_bytes(b"pcm")

    cache_mod.mark_done("meeting")

    assert not any((d / n).exists() for n in ("audio.wav", "left.wav", "right.wav"))
    assert cache_mod.is_done("meeting")


def test_ensure_manifest_metrics_version_bump_invalidates_only_metric_stages(tmp_path):
    """Derived metrics changed without the transcription or turns changing:
    recompute the cheap analysis stages, keep the expensive transcription."""
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"audio-bytes")

    cache_mod.ensure_manifest("slug-metrics", audio, "large", False)
    for stage in ("whisper", "diarize", "align", "mood", "interaction", "prosody"):
        cache_mod.write("slug-metrics", stage, {"stage": stage})

    # Simulate a manifest written by an older version.
    manifest = cache_mod.read_manifest("slug-metrics")
    manifest["metrics_version"] = cache_mod.METRICS_VERSION - 1
    cache_mod._manifest_path("slug-metrics").write_text(json.dumps(manifest), encoding="utf-8")

    invalidated = cache_mod.ensure_manifest("slug-metrics", audio, "large", False)

    assert set(invalidated) == {"mood", "interaction", "prosody"}
    for stage in ("mood", "interaction", "prosody"):
        assert not cache_mod.is_cached("slug-metrics", stage), stage
    # The expensive stages survive — nothing about the audio or the turns changed.
    for stage in ("whisper", "diarize", "align"):
        assert cache_mod.is_cached("slug-metrics", stage), stage
    assert cache_mod.read_manifest("slug-metrics")["metrics_version"] == cache_mod.METRICS_VERSION


def test_ensure_manifest_treats_missing_metrics_version_as_stale(tmp_path):
    """Manifests written before metrics were versioned report `null` overlaps for
    dual-channel recordings, so they must be recomputed."""
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"audio-bytes")

    cache_mod.ensure_manifest("slug-legacy", audio, "large", False)
    cache_mod.write("slug-legacy", "interaction", {"backend": "dual-channel+pyannote"})

    manifest = cache_mod.read_manifest("slug-legacy")
    del manifest["metrics_version"]
    cache_mod._manifest_path("slug-legacy").write_text(json.dumps(manifest), encoding="utf-8")

    assert cache_mod.ensure_manifest("slug-legacy", audio, "large", False) == ["interaction"]


def test_align_invalidation_also_drops_interaction(tmp_path):
    """Interaction measures turns pruned against aligned speech, so a new
    alignment means the cached turn metrics describe a turn set that is gone."""
    cache_mod.write("slug-dag", "align", {"segments": []})
    cache_mod.write("slug-dag", "interaction", {"backend": "dual-channel+pyannote"})

    removed = cache_mod.invalidate_stage("slug-dag", "align")

    assert "interaction" in removed
    assert not cache_mod.is_cached("slug-dag", "interaction")


def test_model_change_invalidates_interaction_through_align(tmp_path):
    """The transitive path whisper -> align -> interaction must hold: a different
    Whisper model changes the segments, which changes the pruning."""
    audio = tmp_path / "memo.m4a"
    audio.write_bytes(b"audio-bytes")

    cache_mod.ensure_manifest("slug-model-int", audio, "large", False)
    for stage in ("whisper", "align", "interaction"):
        cache_mod.write("slug-model-int", stage, {"stage": stage})

    invalidated = cache_mod.ensure_manifest("slug-model-int", audio, "turbo", False)

    assert "interaction" in invalidated
    assert not cache_mod.is_cached("slug-model-int", "interaction")

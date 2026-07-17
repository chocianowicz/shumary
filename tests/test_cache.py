"""Tests for cache.py — disk-backed stage cache."""

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


def test_descendants_of_whisper_includes_align_mood_and_prosody():
    assert set(cache_mod.descendants_of("whisper")) == {"align", "mood", "prosody"}


def test_descendants_of_diarize_includes_align_mood_interaction_and_prosody():
    assert set(cache_mod.descendants_of("diarize")) == {"align", "mood", "interaction", "prosody"}


def test_descendants_of_align_includes_mood_and_prosody():
    assert set(cache_mod.descendants_of("align")) == {"mood", "prosody"}


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

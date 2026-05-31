"""Tests for cache.py — disk-backed stage cache."""

import pytest
from shumary import cache as cache_mod


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


def test_clear_single_stage():
    cache_mod.write("slug3", "whisper", {})
    cache_mod.write("slug3", "mood", {})
    cache_mod.clear("slug3", "whisper")
    assert not cache_mod.is_cached("slug3", "whisper")
    assert cache_mod.is_cached("slug3", "mood")


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

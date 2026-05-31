"""Disk-backed per-slug, per-stage JSON cache for the transcription pipeline."""

import json
import os
from pathlib import Path
from typing import Any

from shumary.paths import cache_dir_for

DONE_SENTINEL = "DONE"


def _stage_path(slug: str, stage: str) -> Path:
    return cache_dir_for(slug) / f"{stage}.json"


def is_cached(slug: str, stage: str) -> bool:
    return _stage_path(slug, stage).exists()


def read(slug: str, stage: str) -> dict[str, Any] | None:
    p = _stage_path(slug, stage)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write(slug: str, stage: str, payload: dict[str, Any]) -> None:
    p = _stage_path(slug, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def clear(slug: str, stage: str | None = None) -> None:
    d = cache_dir_for(slug)
    if not d.exists():
        return
    if stage is None:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    else:
        p = _stage_path(slug, stage)
        if p.exists():
            p.unlink()
        # Also clear DONE sentinel so pipeline re-runs the tail
        done = d / DONE_SENTINEL
        if done.exists():
            done.unlink()


def mark_done(slug: str) -> None:
    d = cache_dir_for(slug)
    d.mkdir(parents=True, exist_ok=True)
    (d / DONE_SENTINEL).touch()


def is_done(slug: str) -> bool:
    return (cache_dir_for(slug) / DONE_SENTINEL).exists()


def list_slugs() -> list[str]:
    from shumary.paths import CACHE
    if not CACHE.exists():
        return []
    return sorted(d.name for d in CACHE.iterdir() if d.is_dir())


def stage_status(slug: str) -> dict[str, bool]:
    stages = ["whisper", "diarize", "align", "mood"]
    return {s: is_cached(slug, s) for s in stages} | {"done": is_done(slug)}

"""Disk-backed per-slug, per-stage JSON cache for the transcription pipeline."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from shumary.errors import ShumaryError
from shumary.paths import cache_dir_for

DONE_SENTINEL = "DONE"
MANIFEST_FILE = "manifest.json"

# Bumped when the cache schema or stage outputs change in a way that makes
# previously-cached data unusable (e.g. word timestamps, new diarize format).
# A cache dir with stage files but no manifest predates this constant
# (version 1) and is treated as stale.
#
# v3 (Phase 3): whisper cache gained per-segment word timestamps
# (`Segment.words`) via the new backend abstraction — old whisper caches
# (v1/v2) lack this field and must be recomputed.
#
# v4 (Phase 5.1): MoodSignals schema gained `per_speaker` (tuple of
# SpeakerMood) and the filler lexicon was split into HESITATIONS vs
# DISCOURSE_MARKERS — old mood caches lack per-speaker signals and must be
# recomputed. `MoodSignals.from_dict` still deserializes old cache entries
# (missing `per_speaker` defaults to `()`), but the version bump ensures
# `ensure_manifest` invalidates `whisper` (and therefore `mood`, via the
# align/mood DAG) so freshly-recomputed transcripts get per-speaker signals.
#
# v5 (Phase 7.1): new `prosody` cached stage (align -> prosody) extracting
# per-segment F0/energy/voice-quality features. No existing cache entries
# need invalidation for this addition (prosody is purely additive), but the
# version bump documents the schema change and keeps `ensure_manifest`'s
# `pipeline_version` check meaningful.
PIPELINE_VERSION = 5

# Bumped when *derived metrics* change without the transcription or the turns
# changing. Invalidates the cheap analysis stages only — unlike PIPELINE_VERSION,
# which invalidates whisper and therefore the entire expensive tail.
#
# v2: overlap/interruption counts are no longer suppressed for dual-channel
# backends, and turns with no transcribed speech no longer contribute to
# turn-taking metrics. Cached v1 entries report `null` overlaps for exactly the
# backend where overlap is measurable.
# v3: mood and prosody take the accepted-speaker set, so their per-speaker tables
# no longer list disqualified labels; the filler verdict is split into
# hesitations vs discourse markers, and `это` left the Russian marker lexicon.
METRICS_VERSION = 3

# Stages recomputed on a METRICS_VERSION bump.
METRICS_STAGES: tuple[str, ...] = ("mood", "interaction", "prosody")

# Stage dependency graph: stage -> stages that must be invalidated if it is.
STAGE_DEPENDENTS: dict[str, list[str]] = {
    "whisper": ["align"],
    "diarize": ["align", "interaction"],
    # `interaction` depends on align as well as on diarize: the turns it measures
    # are pruned against what alignment attributed, so different segments mean
    # different turns. Without this edge a model change would invalidate align
    # but leave interaction reporting metrics for a turn set that no longer exists.
    "align": ["mood", "prosody", "interaction"],
    "mood": [],
    "interaction": [],
    "prosody": [],
}


# ── Audio fingerprint ────────────────────────────────────────────────────────────

def compute_audio_hash(audio: Path) -> str:
    """Cheap fingerprint: SHA-256 of the first 1 MB plus the file size."""
    h = hashlib.sha256()
    with open(audio, "rb") as f:
        h.update(f.read(1024 * 1024))
    h.update(str(audio.stat().st_size).encode("ascii"))
    return h.hexdigest()


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
    """Write `stage`'s payload and drop any descendants' cached files.

    A freshly-recomputed stage can never feed a stale downstream cache: the
    next run will recompute `stage`'s descendants (and re-mark DONE) from the
    new data.
    """
    p = _stage_path(slug, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)

    for dep in descendants_of(stage):
        _remove_stage_file(slug, dep)
    done = cache_dir_for(slug) / DONE_SENTINEL
    if done.exists():
        done.unlink()


def clear(slug: str, stage: str | None = None) -> None:
    d = cache_dir_for(slug)
    if not d.exists():
        return
    if stage is None:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    else:
        invalidate_stage(slug, stage)


def mark_done(slug: str) -> None:
    d = cache_dir_for(slug)
    d.mkdir(parents=True, exist_ok=True)
    for name in ("audio.wav", "left.wav", "right.wav"):
        (d / name).unlink(missing_ok=True)
    (d / DONE_SENTINEL).touch()


def is_done(slug: str) -> bool:
    return (cache_dir_for(slug) / DONE_SENTINEL).exists()


def list_slugs() -> list[str]:
    from shumary.paths import CACHE
    if not CACHE.exists():
        return []
    return sorted(d.name for d in CACHE.iterdir() if d.is_dir())


def stage_status(slug: str) -> dict[str, bool]:
    stages = ["whisper", "diarize", "align", "mood", "interaction", "prosody"]
    return {s: is_cached(slug, s) for s in stages} | {"done": is_done(slug)}


# ── Dependency-aware invalidation ────────────────────────────────────────────────

def descendants_of(stage: str) -> list[str]:
    """All stages transitively invalidated by recomputing `stage` (not including itself)."""
    seen: list[str] = []
    queue = list(STAGE_DEPENDENTS.get(stage, []))
    while queue:
        nxt = queue.pop(0)
        if nxt not in seen:
            seen.append(nxt)
            queue.extend(STAGE_DEPENDENTS.get(nxt, []))
    return seen


def _remove_stage_file(slug: str, stage: str) -> bool:
    """Remove a single stage's cached JSON file, if present. Returns whether it existed."""
    p = _stage_path(slug, stage)
    if p.exists():
        p.unlink()
        return True
    return False


def invalidate_stage(slug: str, stage: str) -> list[str]:
    """Drop `stage` and all its transitive descendants' cached files + DONE sentinel.

    Returns the list of stages actually removed (including `stage` itself, if it
    was cached).
    """
    removed: list[str] = []
    if _remove_stage_file(slug, stage):
        removed.append(stage)
    for dep in descendants_of(stage):
        if _remove_stage_file(slug, dep):
            removed.append(dep)
    done = cache_dir_for(slug) / DONE_SENTINEL
    if done.exists():
        done.unlink()
    return removed


# ── Fingerprint manifest ──────────────────────────────────────────────────────────

def _manifest_path(slug: str) -> Path:
    return cache_dir_for(slug) / MANIFEST_FILE


def read_manifest(slug: str) -> dict[str, Any] | None:
    p = _manifest_path(slug)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_manifest(
    slug: str, *, audio_hash: str, model_name: str, fast: bool, dual_source: bool = False
) -> None:
    p = _manifest_path(slug)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audio_hash": audio_hash,
        "model_name": model_name,
        "fast": fast,
        "dual_source": dual_source,
        "pipeline_version": PIPELINE_VERSION,
        "metrics_version": METRICS_VERSION,
    }
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def fingerprint_info(slug: str) -> str:
    """One-line summary of a slug's manifest for `cache list` — or "v1/legacy"."""
    manifest = read_manifest(slug)
    if manifest is None:
        return "v1/legacy"
    model_name = manifest.get("model_name", "?")
    fast = manifest.get("fast")
    fast_suffix = " --fast" if fast else ""
    audio_hash = manifest.get("audio_hash", "")
    short_hash = audio_hash[:8] if audio_hash else "????????"
    return f"{model_name}{fast_suffix} · {short_hash}"


def ensure_manifest(
    slug: str, audio: Path, model_name: str, fast: bool, dual_source: bool = False
) -> list[str]:
    """Reconcile the slug's cache manifest against the current run's parameters.

    - No manifest but stage files present: a v1/legacy cache (predates the
      manifest) — invalidate everything for the slug and write a fresh manifest.
    - No manifest and no stage files: first contact — just write the manifest.
    - Manifest present, audio hash differs: same slug, different file —
      raise `ShumaryError` (rename the file or clear the cache).
    - Manifest present, `model_name`/`fast`/`pipeline_version` differ: invalidate
      the `whisper` stage and its descendants, then update the manifest.
    - Manifest present, `dual_source` differs: invalidate the `diarize` stage and
      its descendants — the same audio diarized per-channel and mixed-down
      produces different turns.
    - Otherwise: no-op.

    Returns the list of stages invalidated (possibly empty).
    """
    audio_hash = compute_audio_hash(audio)
    manifest = read_manifest(slug)

    if manifest is None:
        d = cache_dir_for(slug)
        had_stage_files = d.exists() and any(
            f.name not in (DONE_SENTINEL, MANIFEST_FILE) and not f.name.endswith(".tmp")
            for f in d.iterdir()
        )
        invalidated: list[str] = []
        if had_stage_files:
            for stage in STAGE_DEPENDENTS:
                if _remove_stage_file(slug, stage):
                    invalidated.append(stage)
            done = d / DONE_SENTINEL
            if done.exists():
                done.unlink()
        write_manifest(
            slug, audio_hash=audio_hash, model_name=model_name,
            fast=fast, dual_source=dual_source,
        )
        return invalidated

    if manifest.get("audio_hash") != audio_hash:
        raise ShumaryError(
            f"Cache for `{slug}` was fingerprinted from a different file with the "
            "same name — rename the new file or run `shumary cache clear "
            f"{slug}` to recompute."
        )

    params_changed = (
        manifest.get("model_name") != model_name
        or manifest.get("fast") != fast
        or manifest.get("pipeline_version") != PIPELINE_VERSION
    )
    dual_changed = bool(manifest.get("dual_source", False)) != bool(dual_source)
    # Absent on manifests written before metrics were versioned — treat as v1.
    metrics_changed = manifest.get("metrics_version", 1) != METRICS_VERSION

    if params_changed or dual_changed or metrics_changed:
        invalidated: list[str] = []
        if params_changed:
            invalidated += invalidate_stage(slug, "whisper")
        if dual_changed:
            invalidated += [s for s in invalidate_stage(slug, "diarize") if s not in invalidated]
        if metrics_changed:
            for stage in METRICS_STAGES:
                invalidated += [s for s in invalidate_stage(slug, stage) if s not in invalidated]
        write_manifest(
            slug, audio_hash=audio_hash, model_name=model_name,
            fast=fast, dual_source=dual_source,
        )
        return invalidated

    return []

"""Speaker diarization stage.

Strategy:
  1. Try pyannote (best accuracy) if HF_TOKEN is available.
  2. If pyannote loads but fails at runtime (e.g. a bad/corrupt audio file),
     fall back to resemblyzer instead of giving up.
  3. Fall back to resemblyzer + agglomerative clustering (tokenless, ~5s).
  4. Return None if neither is available.

Speaker count estimation uses silhouette score (not gap heuristic) because
gap heuristic is unreliable on long recordings where within-speaker variance
grows and creates false gaps in the dendrogram.
"""

from pathlib import Path

from shumary.config import SOLO_SILHOUETTE
from shumary.models import Turn, DiarizationResult, SpeakerStat, SpeakerStats
from shumary.modelhub import diarization_pipeline, resemblyzer_encoder


def run(audio: Path) -> DiarizationResult | None:
    """Return speaker turns plus the backend that produced them, or None on failure."""
    pipeline = diarization_pipeline()
    if pipeline is not None:
        turns = _run_pyannote(audio, pipeline)
        if turns is not None:
            return DiarizationResult(turns=tuple(turns), backend="pyannote")
        # pyannote loaded but failed at runtime — fall through to resemblyzer
        # instead of giving up entirely.
        print("  ↪  falling back to resemblyzer")
    turns = _run_resemblyzer(audio)
    if turns is not None:
        return DiarizationResult(turns=tuple(turns), backend="resemblyzer")
    return None


# ── pyannote backend ───────────────────────────────────────────────────────────

def _run_pyannote(audio: Path, pipeline) -> list[Turn] | None:
    try:
        diarization = pipeline(str(audio))
    except Exception as exc:
        print(f"  ⚠  pyannote diarization failed: {exc}")
        return None
    # pyannote.audio 4.x returns a DiarizeOutput wrapper; the Annotation lives
    # at .speaker_diarization. 3.x returns the Annotation directly.
    annotation = getattr(diarization, "speaker_diarization", diarization)
    turns = [
        Turn(start=turn.start, end=turn.end, speaker=speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    return turns if turns else None


# ── resemblyzer backend ────────────────────────────────────────────────────────

_WINDOW_SEC   = 1.5
_HOP_SEC      = 0.75
_MIN_SPEAKERS = 2
_MAX_SPEAKERS = 8
_SUBSAMPLE_N  = 400   # max windows used for silhouette estimation


def _run_resemblyzer(audio: Path) -> list[Turn] | None:
    import numpy as np

    encoder = resemblyzer_encoder()
    if encoder is None:
        return None

    try:
        from resemblyzer import preprocess_wav  # type: ignore[import-untyped]
        from scipy.cluster.hierarchy import linkage, fcluster  # type: ignore[import-untyped]
    except ImportError as exc:
        print(f"  ⚠  resemblyzer backend missing dependency: {exc}")
        return None

    try:
        wav = preprocess_wav(str(audio))
    except Exception as exc:
        print(f"  ⚠  resemblyzer audio load failed: {exc}")
        return None

    sr = 16_000  # preprocess_wav resamples to 16kHz

    try:
        _, partials, wav_splits = encoder.embed_utterance(
            wav,
            return_partials=True,
            rate=1.0 / _HOP_SEC,
            min_coverage=0.5,
        )
    except Exception as exc:
        print(f"  ⚠  resemblyzer embedding failed: {exc}")
        return None

    if partials is None or len(partials) == 0:
        return None

    E = np.array(partials)  # (N, 256)
    # Ensure Python floats so numpy types don't leak into downstream YAML/JSON
    times = [float((sl.start + sl.stop) / 2 / sr) for sl in wav_splits]

    Z = linkage(E, method="ward")

    # Spec 4.3 — single-speaker decision rule. A weak forced 2-way split means
    # there's really only one voice; avoid forcing a spurious k>=2 split.
    sil2 = _silhouette_for_k(E, Z, 2)
    if sil2 is not None and sil2 < SOLO_SILHOUETTE:
        print(
            f"  resemblyzer: weak 2-way split (silhouette {sil2:.2f}) "
            "→ treating as single speaker"
        )
        return [Turn(start=max(0.0, times[0] - _WINDOW_SEC / 2),
                      end=times[-1] + _WINDOW_SEC / 2,
                      speaker="SPEAKER_00")]

    n_speakers = _estimate_speakers(E, Z)
    print(f"  resemblyzer: estimated {n_speakers} speaker(s)")

    labels = fcluster(Z, n_speakers, criterion="maxclust")
    turns = _labels_to_turns(labels, times)
    return turns if turns else None


def _subsample(E, Z, k: int):
    """Evenly-spaced subsample of E (preserves temporal distribution) plus its
    fcluster labels for k clusters. Returns (E_sub, labels_sub) or (None, None)
    if subsampling left a cluster empty."""
    import numpy as np
    from scipy.cluster.hierarchy import fcluster  # type: ignore[import-untyped]

    n = len(E)
    if n > _SUBSAMPLE_N:
        idx = np.linspace(0, n - 1, _SUBSAMPLE_N, dtype=int)
        E_sub = E[idx]
    else:
        idx = np.arange(n)
        E_sub = E

    labels_full = fcluster(Z, k, criterion="maxclust")
    labels_sub = labels_full[idx]

    if len(np.unique(labels_sub)) < k:
        return None, None
    return E_sub, labels_sub


def _silhouette_for_k(E, Z, k: int) -> float | None:
    """Mean silhouette score for a forced k-way split, or None if too few
    points to evaluate (matches the `n < _MIN_SPEAKERS * 4` guard below)."""
    if len(E) < _MIN_SPEAKERS * 4:
        return None
    E_sub, labels_sub = _subsample(E, Z, k)
    if E_sub is None:
        return None
    return _mean_silhouette(E_sub, labels_sub)


def _estimate_speakers(E, Z, max_k: int = _MAX_SPEAKERS) -> int:
    """Silhouette-based speaker count: try k=2..max_k, return highest-scoring k.

    Subsamples E to _SUBSAMPLE_N points for speed on long recordings.
    Gap-heuristic methods fail here because within-speaker variance grows over
    long files, producing false gaps at k>2 in the Ward dendrogram.
    """
    import numpy as np
    from scipy.cluster.hierarchy import fcluster  # type: ignore[import-untyped]

    n = len(E)
    if n < _MIN_SPEAKERS * 4:
        return _MIN_SPEAKERS

    # Evenly-spaced subsample — preserves temporal distribution
    if n > _SUBSAMPLE_N:
        idx = np.linspace(0, n - 1, _SUBSAMPLE_N, dtype=int)
        E_sub = E[idx]
    else:
        idx = np.arange(n)
        E_sub = E

    best_k, best_score = _MIN_SPEAKERS, -2.0
    k_limit = min(max_k, n // 4)

    for k in range(_MIN_SPEAKERS, k_limit + 1):
        labels_full = fcluster(Z, k, criterion="maxclust")
        labels_sub  = labels_full[idx]

        # Skip if subsampling happened to leave a cluster empty
        if len(np.unique(labels_sub)) < k:
            continue

        score = _mean_silhouette(E_sub, labels_sub)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _mean_silhouette(E, labels) -> float:
    """Mean silhouette score using cosine distance. O(N²) on the subsample."""
    import numpy as np
    from scipy.spatial.distance import cdist  # type: ignore[import-untyped]

    unique = np.unique(labels)
    if len(unique) < 2:
        return -1.0

    D = cdist(E, E, metric="cosine")
    scores = []
    for i in range(len(E)):
        mask_same = labels == labels[i]
        same_dists = D[i, mask_same]
        a = same_dists[same_dists > 0].mean() if mask_same.sum() > 1 else 0.0

        b = min(
            D[i, labels == c].mean()
            for c in unique if c != labels[i]
        )

        denom = max(a, b)
        scores.append(float((b - a) / denom) if denom > 0 else 0.0)

    return float(np.mean(scores))


def _labels_to_turns(labels, times: list[float]) -> list[Turn]:
    """Merge consecutive windows with the same label into Turn objects."""
    turns: list[Turn] = []
    if not len(labels):
        return turns

    current_label = labels[0]
    seg_start = times[0] - _WINDOW_SEC / 2

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            seg_end = times[i - 1] + _WINDOW_SEC / 2
            turns.append(Turn(
                start=max(0.0, seg_start),
                end=seg_end,
                speaker=f"SPEAKER_{current_label - 1:02d}",
            ))
            current_label = labels[i]
            seg_start = times[i] - _WINDOW_SEC / 2

    seg_end = times[-1] + _WINDOW_SEC / 2
    turns.append(Turn(
        start=max(0.0, seg_start),
        end=seg_end,
        speaker=f"SPEAKER_{current_label - 1:02d}",
    ))
    return turns


# ── speaker stats ──────────────────────────────────────────────────────────────

def speaker_stats(turns: list[Turn] | None) -> SpeakerStats | None:
    if not turns:
        return None

    total_sec = sum(t.end - t.start for t in turns)
    if total_sec <= 0:
        return None

    seen: dict[str, float] = {}
    for t in turns:
        seen[t.speaker] = seen.get(t.speaker, 0.0) + (t.end - t.start)

    stats = tuple(
        SpeakerStat(
            label=label,
            duration_sec=dur,
            pct=round(dur / total_sec * 100),
        )
        for label, dur in seen.items()
    )
    return SpeakerStats(count=len(stats), speakers=stats)

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

import wave
from pathlib import Path

from shumary.config import (
    MIN_SPEAKER_SPEECH_SEC,
    DUAL_ECHO_MARGIN_DB,
    DUAL_FRAME_SEC,
    DUAL_HOP_SEC,
    DUAL_MIN_GAP_SEC,
    DUAL_MIN_ONSET_SEC,
    DUAL_MIN_SPEECH_SEC,
    DUAL_NOISE_FLOOR_PCT,
    DUAL_ONSET_MARGIN_DB,
    DUAL_SILENT_CHANNEL_PEAK,
    DUAL_SPEECH_MARGIN_DB,
    SOLO_SILHOUETTE,
)
from shumary.models import (
    AlignedSegment,
    DiarizationResult,
    SpeakerStat,
    SpeakerStats,
    Turn,
)
from shumary.modelhub import diarization_pipeline, resemblyzer_encoder


def run(audio: Path, progress=None) -> DiarizationResult | None:
    """Return speaker turns plus the backend that produced them, or None on failure."""
    pipeline = diarization_pipeline()
    if pipeline is not None:
        turns = _run_pyannote(audio, pipeline, progress=progress)
        if turns is not None:
            return DiarizationResult(turns=tuple(turns), backend="pyannote")
        # pyannote loaded but failed at runtime — fall through to resemblyzer
        # instead of giving up entirely.
        print("  ↪  falling back to resemblyzer")
    turns = _run_resemblyzer(audio)
    if turns is not None:
        return DiarizationResult(turns=tuple(turns), backend="resemblyzer")
    return None


# ── dual-source (mic + system) ─────────────────────────────────────────────────

def run_dual(
    left: Path, right: Path, *, mic_label: str, progress=None
) -> DiarizationResult | None:
    """Diarize a recording whose channels are separate sources.

    The mic channel needs no clustering — anything on it is the person at the
    keyboard — so it is gated by energy and labelled directly. Only the system
    channel goes through a diarization backend, to split the remote
    participants from each other.

    Turns from the two channels are kept as-is rather than made mutually
    exclusive: genuine simultaneous speech is exactly what makes interruption
    metrics meaningful here.
    """
    mic_turns = _mic_turns(left, right, label=mic_label)

    remote = run(right, progress=progress) if _has_signal(right) else None
    if remote is None and not mic_turns:
        return None

    remote_turns = list(remote.turns) if remote else []
    backend = f"dual-channel+{remote.backend}" if remote else "dual-channel"

    turns = sorted(mic_turns + remote_turns, key=lambda t: (t.start, t.end))
    return DiarizationResult(turns=tuple(turns), backend=backend)


def _read_mono_wav(path: Path):
    """16-bit mono WAV as a float array in [-1, 1], plus its sample rate."""
    import numpy as np

    with wave.open(str(path), "rb") as f:
        rate = f.getframerate()
        frames = f.readframes(f.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, rate


def _rms_envelope(samples, rate: int):
    """Frame-wise RMS of `samples`, using the dual-source framing constants."""
    import numpy as np

    frame = max(1, int(DUAL_FRAME_SEC * rate))
    hop = max(1, int(DUAL_HOP_SEC * rate))
    if len(samples) < frame:
        return np.zeros(0, dtype=np.float32)

    count = 1 + (len(samples) - frame) // hop
    # Strided view over the samples — avoids materializing count*frame floats
    # for a long recording.
    windows = np.lib.stride_tricks.as_strided(
        samples,
        shape=(count, frame),
        strides=(samples.strides[0] * hop, samples.strides[0]),
    )
    return np.sqrt((windows.astype(np.float32) ** 2).mean(axis=1))


def _has_signal(path: Path) -> bool:
    import numpy as np

    try:
        samples, _ = _read_mono_wav(path)
    except (OSError, wave.Error):
        return False
    return bool(len(samples)) and float(np.max(np.abs(samples))) > DUAL_SILENT_CHANNEL_PEAK


def _mic_turns(left: Path, right: Path, *, label: str) -> list[Turn]:
    """Speech turns for the mic channel, rejecting bleed from the speakers."""
    import numpy as np

    try:
        mic, rate = _read_mono_wav(left)
        system, _ = _read_mono_wav(right)
    except (OSError, wave.Error) as exc:
        print(f"  ⚠  could not read recording channels: {exc}")
        return []

    mic_rms = _rms_envelope(mic, rate)
    system_rms = _rms_envelope(system, rate)
    if not len(mic_rms):
        return []

    # ffmpeg can emit a frame or two of difference between the two channels.
    n = min(len(mic_rms), len(system_rms)) if len(system_rms) else len(mic_rms)
    mic_rms = mic_rms[:n]
    system_rms = system_rms[:n] if len(system_rms) else np.zeros(n, dtype=np.float32)

    floor = float(np.percentile(mic_rms, DUAL_NOISE_FLOOR_PCT))
    speech_threshold = max(floor * 10 ** (DUAL_SPEECH_MARGIN_DB / 20), DUAL_SILENT_CHANNEL_PEAK)
    onset_threshold = max(floor * 10 ** (DUAL_ONSET_MARGIN_DB / 20), DUAL_SILENT_CHANNEL_PEAK)
    echo_ratio = 10 ** (DUAL_ECHO_MARGIN_DB / 20)

    beats_system = mic_rms > system_rms * echo_ratio
    active = (mic_rms > speech_threshold) & beats_system
    onset = (mic_rms > onset_threshold) & beats_system

    spans = _spans_from_mask(active, hop_sec=DUAL_HOP_SEC, frame_sec=DUAL_FRAME_SEC)
    spans = [s for s in spans if _has_sustained_onset(onset, s)]
    return [Turn(start=start, end=end, speaker=label) for start, end in spans]


def _has_sustained_onset(onset, span: tuple[float, float]) -> bool:
    """Whether `span` holds the onset threshold long enough to be speech.

    Applied after smoothing, so a span already bridges the ordinary pauses
    between words: a single sustained syllable anywhere inside it is enough to
    admit the whole turn.
    """
    import numpy as np

    start, end = span
    i0 = max(0, int(start / DUAL_HOP_SEC))
    i1 = min(len(onset), int(round(end / DUAL_HOP_SEC)) + 1)
    if i1 <= i0:
        return False

    window = onset[i0:i1]
    if not window.any():
        return False

    min_frames = max(1, int(round(DUAL_MIN_ONSET_SEC / DUAL_HOP_SEC)))
    if min_frames == 1:
        return True

    # Longest contiguous run of True in `window`.
    padded = np.concatenate(([False], window, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return bool(len(starts)) and int((ends - starts).max()) >= min_frames


def _spans_from_mask(mask, *, hop_sec: float, frame_sec: float) -> list[tuple[float, float]]:
    """Convert a boolean frame mask into smoothed (start, end) second spans."""
    import numpy as np

    if not mask.any():
        return []

    # Frame index -> span, then smooth in the time domain where the thresholds
    # are expressed.
    edges = np.diff(mask.astype(np.int8))
    starts = list((np.flatnonzero(edges == 1) + 1))
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(len(mask))

    spans = [
        (float(s * hop_sec), float(e * hop_sec + frame_sec))
        for s, e in zip(starts, ends)
    ]

    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= DUAL_MIN_GAP_SEC:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return [(s, e) for s, e in merged if e - s >= DUAL_MIN_SPEECH_SEC]


# ── pyannote backend ───────────────────────────────────────────────────────────

class _PyannoteProgress:
    """Adapts pyannote's `hook(step_name, artifact, total=, completed=)` protocol
    to a single 0..1 fraction.

    pyannote reports per *step* (segmentation, embeddings, clustering, …) and
    each step restarts its own counter, so the raw numbers march 0→1 several
    times over. Steps are given equal slices of the stage in the order they first
    appear, which is an approximation — but the caller clamps monotonically, so
    the worst case is a bar that pauses, never one that jumps backwards.
    """

    # speaker-diarization-3.1 runs segmentation, embeddings and clustering.
    EXPECTED_STEPS = 3

    def __init__(self, report) -> None:
        self._report = report
        self._steps: list[str] = []

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        try:
            if step_name not in self._steps:
                self._steps.append(step_name)
            index = self._steps.index(step_name)
            within = (completed / total) if (total and completed is not None) else 0.0
            slices = max(len(self._steps), self.EXPECTED_STEPS)
            self._report(min(1.0, (index + min(1.0, within)) / slices))
        except Exception:
            pass  # progress reporting must never break diarization


def _run_pyannote(audio: Path, pipeline, progress=None) -> list[Turn] | None:
    try:
        if progress is not None:
            try:
                diarization = pipeline(str(audio), hook=_PyannoteProgress(progress))
            except TypeError:
                # Older/newer pipelines that don't accept `hook`.
                diarization = pipeline(str(audio))
        else:
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


# Labels that alignment invents for text no turn accounts for — not participants.
NON_SPEAKER_LABELS = frozenset({None, "SPEAKER_??"})


def speaker_stats_from_speech(segments: list[AlignedSegment] | None) -> SpeakerStats | None:
    """Talk time per speaker measured from *transcribed speech*, not from turns.

    A turn is a claim that someone was audible; an aligned segment is evidence
    that they said words. The two diverge sharply for the dual-channel mic
    label, whose turns come from RMS energy gating and so also fire on
    keyboards, breathing and chair noise — inflating the mic speaker's share
    several-fold while the transcript itself stays correct.

    Reporting the aligned figure also drops any label holding less than
    MIN_SPEAKER_SPEECH_SEC of speech, which is how single-word phantom clusters
    disappear from the speaker list instead of being presented as a participant.
    """
    if not segments:
        return None

    spans: dict[str, list[tuple[float, float]]] = {}
    for s in segments:
        if s.speaker in NON_SPEAKER_LABELS or not s.text.strip():
            continue
        if s.end > s.start:
            spans.setdefault(s.speaker, []).append((s.start, s.end))

    # Words are assigned to exactly one speaker, but segment boundaries from
    # different Whisper segments can still abut or overlap slightly — merge per
    # speaker so nobody is credited twice for the same moment.
    seen = {label: _merged_duration(v) for label, v in spans.items()}
    seen = {label: dur for label, dur in seen.items() if dur >= MIN_SPEAKER_SPEECH_SEC}
    total_sec = sum(seen.values())
    if total_sec <= 0:
        return None

    stats = tuple(
        SpeakerStat(
            label=label,
            duration_sec=dur,
            pct=round(dur / total_sec * 100),
        )
        for label, dur in sorted(seen.items(), key=lambda kv: -kv[1])
    )
    return SpeakerStats(count=len(stats), speakers=stats)


def _merged_duration(spans: list[tuple[float, float]]) -> float:
    total = 0.0
    cur_start, cur_end = None, None
    for start, end in sorted(spans):
        if cur_end is None:
            cur_start, cur_end = start, end
        elif start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def prune_unvoiced_turns(
    turns: list[Turn] | None,
    segments: list[AlignedSegment] | None,
    *,
    keep_labels: set[str] | None = None,
) -> list[Turn] | None:
    """Drop turns that alignment attributed no speech to.

    `keep_labels`, when given, additionally restricts turns to labels that
    qualified as speakers, so a phantom cluster cannot go on contributing
    interruptions and response latencies after being dropped from the table.

    Turn-derived metrics (turn counts, response latency, overlap, interruptions)
    are only meaningful over turns that contain words. Energy-gated mic turns
    that caught a keyboard press are indistinguishable from speech to the
    detector but receive no aligned text, so this is the point where they can be
    told apart — using evidence the pipeline has already computed.

    Alignment is deliberately *not* re-run afterwards: a pruned turn never won a
    word, so removing it cannot change any attribution, and the transcript stays
    byte-identical.
    """
    if not turns or not segments:
        return turns

    voiced: list[tuple[float, float, str]] = sorted(
        (s.start, s.end, s.speaker)
        for s in segments
        if s.speaker not in NON_SPEAKER_LABELS and s.text.strip() and s.end > s.start
    )
    if not voiced:
        return turns

    kept = [
        t for t in turns
        if (keep_labels is None or t.speaker in keep_labels)
        and _has_voiced_overlap(t, voiced)
    ]
    # Never hand back an empty timeline: if nothing survives, the evidence is
    # more likely missing than every turn being spurious.
    return kept if kept else turns


def _has_voiced_overlap(turn: Turn, voiced: list[tuple[float, float, str]]) -> bool:
    for start, end, speaker in voiced:
        if start >= turn.end:
            break  # sorted by start — no later segment can overlap this turn
        if speaker == turn.speaker and min(turn.end, end) - max(turn.start, start) > 0:
            return True
    return False

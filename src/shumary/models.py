"""Dataclasses for pipeline stage I/O — all frozen, JSON-serialisable."""

import dataclasses
from dataclasses import dataclass, asdict
from typing import Any, get_args, get_origin


def _is_dataclass_type(tp: Any) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _tuples_to_lists(value: Any) -> Any:
    """Recursively convert tuples to lists (asdict() preserves tuples;
    the previous hand-written to_dict() implementations returned lists)."""
    if isinstance(value, (tuple, list)):
        return [_tuples_to_lists(v) for v in value]
    if isinstance(value, dict):
        return {k: _tuples_to_lists(v) for k, v in value.items()}
    return value


def to_dict_helper(obj) -> dict[str, Any]:
    """Generic `to_dict`: `dataclasses.asdict()` with nested tuples
    normalized to lists, matching the previous hand-written shape."""
    return _tuples_to_lists(asdict(obj))


def from_dict_helper(cls: type, d: dict[str, Any], *, defaults: dict[str, Any] | None = None):
    """Generic `from_dict`: reconstruct a frozen dataclass from a plain dict.

    For each field:
      - `tuple[<Dataclass>, ...]` fields recurse via `<Dataclass>.from_dict`.
      - other `tuple[...]` fields are coerced to `tuple(...)`.
      - scalar fields are passed through as-is.

    `defaults` supplies fallback values for keys that may be missing from
    older cache entries (e.g. `{"backend": "unknown"}`); fields with a
    dataclass-level default are otherwise simply omitted from `kwargs` so the
    dataclass default applies.
    """
    defaults = defaults or {}
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name in d:
            value = d[f.name]
        elif f.name in defaults:
            value = defaults[f.name]
        else:
            continue  # field has its own dataclass default

        origin = get_origin(f.type)
        if origin is tuple:
            args = get_args(f.type)
            elem_type = args[0] if args else None
            if _is_dataclass_type(elem_type):
                kwargs[f.name] = tuple(elem_type.from_dict(item) for item in value)
            else:
                kwargs[f.name] = tuple(value)
        else:
            kwargs[f.name] = value

    return cls(**kwargs)


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str          # stripped — openai/mlx words carry a leading space
    prob: float        # word-level probability; 1.0 if backend doesn't provide it

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Word":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    words: tuple[Word, ...] = ()       # empty = word timestamps unavailable

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Segment":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class TranscriptionResult:
    language: str
    word_count: int
    text: str
    segments: tuple[Segment, ...]
    hallucinations_filtered: int = 0

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TranscriptionResult":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class AlignedSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    speaker: str | None

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AlignedSegment":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class SpeakerInteraction:
    speaker: str
    turn_count: int
    avg_turn_sec: float
    median_response_latency_sec: float | None   # None if never responds
    p90_response_latency_sec: float | None
    interruptions_made: int
    interruptions_received: int
    talk_share_by_third: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpeakerInteraction":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class InteractionMetrics:
    backend: str                    # "pyannote" | "resemblyzer"
    overlap_count: int | None
    overlap_total_sec: float | None
    long_gaps: int                  # inter-speaker gaps > LATENCY_CAP_SEC
    speakers: tuple[SpeakerInteraction, ...]

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InteractionMetrics":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class ProsodySegment:
    """Per-segment acoustic features (Spec 7). All feature fields are
    `None` when the segment is too short, has too few voiced frames, or the
    underlying library call failed."""
    start: float
    end: float
    speaker: str | None
    f0_median_hz: float | None
    f0_range_st: float | None
    energy_mean_db: float | None
    energy_range_db: float | None
    jitter_local: float | None
    shimmer_local: float | None

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProsodySegment":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class SpeakerProsody:
    """Per-speaker prosody aggregation (Spec 7.2)."""
    speaker: str
    f0_baseline_hz: float
    f0_variability_st: float
    energy_mean_db: float
    energy_dynamics_db: float
    jitter_local: float | None
    shimmer_local: float | None
    f0_trend_by_third_st: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpeakerProsody":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class ProsodyResult:
    """Cached output of the prosody stage: per-segment values plus the
    per-speaker aggregation."""
    segments: tuple[ProsodySegment, ...]
    speakers: tuple[SpeakerProsody, ...]

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProsodyResult":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class DiarizationResult:
    turns: tuple[Turn, ...]
    backend: str  # "pyannote" | "resemblyzer"

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiarizationResult":
        return from_dict_helper(cls, d, defaults={"backend": "unknown"})


@dataclass(frozen=True)
class SpeakerStat:
    label: str
    duration_sec: float
    pct: int

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpeakerStat":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class SpeakerStats:
    count: int
    speakers: tuple[SpeakerStat, ...]

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpeakerStats":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class PauseDetail:
    at: str        # "MM:SS"
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PauseDetail":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class SpeakerMood:
    speaker: str
    speaking_rate_wps: float | None       # duration-weighted: total words / total speech seconds
    rate_variability: float | None
    intra_pauses: int
    intra_pause_details: tuple[PauseDetail, ...]
    hesitations: int
    discourse_markers: int
    top_fillers: tuple[str, ...]
    repeated_phrases: tuple[str, ...]
    low_confidence_segments: int
    inferred_state: str

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpeakerMood":
        return from_dict_helper(cls, d)


@dataclass(frozen=True)
class MoodSignals:
    avg_speaking_rate_wps: float | None
    rate_variability: float | None
    long_pauses: int
    long_pause_details: tuple[PauseDetail, ...]
    filler_word_count: int          # hesitations + discourse_markers, combined
    top_fillers: tuple[str, ...]
    repeated_phrases: tuple[str, ...]
    low_confidence_segments: int
    near_silence_segments: int
    inferred_state: str
    per_speaker: tuple[SpeakerMood, ...] = ()
    # Split out because they support different claims: hesitations are evidence
    # of load, discourse markers are evidence of conversational style. Defaulted
    # so cache entries written before the split still deserialize.
    hesitations: int = 0
    discourse_markers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return to_dict_helper(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MoodSignals":
        return from_dict_helper(cls, d, defaults={"per_speaker": []})

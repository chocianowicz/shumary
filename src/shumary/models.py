"""Dataclasses for pipeline stage I/O — all frozen, JSON-serialisable."""

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        return cls(**d)


@dataclass(frozen=True)
class TranscriptionResult:
    language: str
    word_count: int
    text: str
    segments: tuple[Segment, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "word_count": self.word_count,
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TranscriptionResult":
        return cls(
            language=d["language"],
            word_count=d["word_count"],
            text=d["text"],
            segments=tuple(Segment(**s) for s in d["segments"]),
        )


@dataclass(frozen=True)
class AlignedSegment:
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    speaker: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AlignedSegment":
        return cls(**d)


@dataclass(frozen=True)
class SpeakerStat:
    label: str
    duration_sec: float
    pct: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SpeakerStats:
    count: int
    speakers: tuple[SpeakerStat, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "speakers": [s.to_dict() for s in self.speakers],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpeakerStats":
        return cls(
            count=d["count"],
            speakers=tuple(SpeakerStat(**s) for s in d["speakers"]),
        )


@dataclass(frozen=True)
class PauseDetail:
    at: str        # "MM:SS"
    duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MoodSignals:
    avg_speaking_rate_wps: float | None
    rate_variability: float | None
    long_pauses: int
    long_pause_details: tuple[PauseDetail, ...]
    filler_word_count: int
    top_fillers: tuple[str, ...]
    repeated_phrases: tuple[str, ...]
    low_confidence_segments: int
    near_silence_segments: int
    inferred_state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "avg_speaking_rate_wps": self.avg_speaking_rate_wps,
            "rate_variability": self.rate_variability,
            "long_pauses": self.long_pauses,
            "long_pause_details": [p.to_dict() for p in self.long_pause_details],
            "filler_word_count": self.filler_word_count,
            "top_fillers": list(self.top_fillers),
            "repeated_phrases": list(self.repeated_phrases),
            "low_confidence_segments": self.low_confidence_segments,
            "near_silence_segments": self.near_silence_segments,
            "inferred_state": self.inferred_state,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MoodSignals":
        return cls(
            avg_speaking_rate_wps=d["avg_speaking_rate_wps"],
            rate_variability=d["rate_variability"],
            long_pauses=d["long_pauses"],
            long_pause_details=tuple(PauseDetail(**p) for p in d["long_pause_details"]),
            filler_word_count=d["filler_word_count"],
            top_fillers=tuple(d["top_fillers"]),
            repeated_phrases=tuple(d["repeated_phrases"]),
            low_confidence_segments=d["low_confidence_segments"],
            near_silence_segments=d["near_silence_segments"],
            inferred_state=d["inferred_state"],
        )

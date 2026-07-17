"""Transcription backend abstraction (Spec 3).

Three interchangeable Whisper backends, normalized to `TranscriptionResult`:

- `mlx`    — mlx-whisper, fastest on Apple Silicon (M1/M2/M3).
- `faster` — faster-whisper (CTranslate2), fast elsewhere, built-in VAD.
- `openai` — openai-whisper, universal fallback.

`pick_backend()` selects one automatically, or honours an explicit override
(`--backend` flag / `SHUMARY_BACKEND` env var).
"""

import os
import platform
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from shumary.config import HALLUCINATION_MIN_DURATION_SEC, HALLUCINATION_NO_SPEECH
from shumary.errors import ShumaryError
from shumary.models import Segment, TranscriptionResult, Word

# ── Model name mapping ──────────────────────────────────────────────────────────
#
# mlx-whisper takes a HF repo string per call (no local model-size names).
# Verified empirically against the HF Hub (model_info() succeeded for all of
# these) — `tiny` was additionally smoke-tested end-to-end with mlx-whisper
# 0.4.3 on this machine. `large` and `turbo` repo ids are taken from
# mlx-community's published naming convention and HF Hub existence, but were
# NOT run through mlx-whisper.transcribe() here (would require downloading
# multi-GB weights) — unverified-at-runtime.
KNOWN_REPOS: dict[str, str] = {
    "tiny": "mlx-community/whisper-tiny-mlx",       # smoke-tested (tiny model, this run)
    "base": "mlx-community/whisper-base-mlx",       # verified: HF repo exists
    "small": "mlx-community/whisper-small-mlx",     # verified: HF repo exists
    "medium": "mlx-community/whisper-medium-mlx",   # verified: HF repo exists
    "large": "mlx-community/whisper-large-v3-mlx",  # verified: HF repo exists; not transcribed
    "turbo": "mlx-community/whisper-large-v3-turbo",  # verified: HF repo exists; not transcribed
}

# faster-whisper (1.2.1) ships its own `_MODELS` alias table that already maps
# "tiny".."medium" to Systran/faster-whisper-<name>, "large" -> large-v3, and
# "turbo" -> mobiuslabsgmbh/faster-whisper-large-v3-turbo. Confirmed by reading
# faster_whisper.utils._MODELS on this machine — shumary names pass straight
# through to WhisperModel(name, ...) with no extra mapping needed.
FASTER_MODEL_NAMES: dict[str, str] = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large",   # -> Systran/faster-whisper-large-v3
    "turbo": "turbo",   # -> mobiuslabsgmbh/faster-whisper-large-v3-turbo (>= faster-whisper 1.1)
}

# openai-whisper accepts these names directly via whisper.load_model().
OPENAI_MODEL_NAMES: dict[str, str] = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large",
    "turbo": "turbo",
}


class TranscribeBackend(Protocol):
    name: str  # "mlx" | "faster" | "openai"

    def available(self) -> bool: ...

    def transcribe(self, wav: Path, model_name: str, *, fast: bool) -> TranscriptionResult: ...


def _strip_word(text: str) -> str:
    return text.strip()


def _normalize_words(raw_words: list, *, prob_key: str = "probability") -> tuple[Word, ...]:
    """Normalize a list of word dicts/objects (openai/mlx/faster shapes) to `Word` tuples."""
    words = []
    for w in raw_words:
        if isinstance(w, dict):
            start = float(w["start"])
            end = float(w["end"])
            text = _strip_word(w["word"])
            prob = float(w.get(prob_key, 1.0))
        else:
            # faster-whisper Word namedtuple-like object: start, end, word, probability
            start = float(w.start)
            end = float(w.end)
            text = _strip_word(w.word)
            prob = float(getattr(w, prob_key, 1.0) or 1.0)
        words.append(Word(start=start, end=end, text=text, prob=prob))
    return tuple(words)


# ── Hallucination post-filter (Spec 3-VAD) ───────────────────────────────────────
#
# Applies to the openai and mlx paths only — faster-whisper has built-in
# vad_filter=True and does not need this.

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize_text_for_dedup(text: str) -> str:
    """casefold + strip punctuation, for repeated-text comparison."""
    return _PUNCT_RE.sub("", text).strip().casefold()


def filter_hallucinations(segments: tuple[Segment, ...]) -> tuple[tuple[Segment, ...], int]:
    """Drop segment i when all hold:
      - no_speech_prob >= HALLUCINATION_NO_SPEECH
      - normalized text equals normalized text of segment i-1
      - duration >= HALLUCINATION_MIN_DURATION_SEC

    Returns `(kept_segments, dropped_count)`.
    """
    kept: list[Segment] = []
    dropped = 0
    prev_norm: str | None = None
    for seg in segments:
        norm = _normalize_text_for_dedup(seg.text)
        duration = seg.end - seg.start
        if (
            seg.no_speech_prob >= HALLUCINATION_NO_SPEECH
            and prev_norm is not None
            and norm == prev_norm
            and duration >= HALLUCINATION_MIN_DURATION_SEC
        ):
            dropped += 1
        else:
            kept.append(seg)
        # Compare the next segment against this segment's (original-order)
        # text regardless of whether this one was dropped — a run of N
        # identical hallucinated segments collapses correctly either way,
        # since they all share the same normalized text.
        prev_norm = norm
    return tuple(kept), dropped


def _apply_hallucination_filter(result: TranscriptionResult) -> TranscriptionResult:
    kept, dropped = filter_hallucinations(result.segments)
    if dropped == 0:
        return result

    print(f"  filtered {dropped} hallucination-suspect segment(s)")
    text = "".join(s.text for s in kept)
    word_count = len(text.split())
    return TranscriptionResult(
        language=result.language,
        word_count=word_count,
        text=text,
        segments=kept,
        hallucinations_filtered=result.hallucinations_filtered + dropped,
    )


# ── openai-whisper backend ──────────────────────────────────────────────────────

class OpenaiBackend:
    name = "openai"

    def available(self) -> bool:
        try:
            import whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def transcribe(self, wav: Path, model_name: str, *, fast: bool) -> TranscriptionResult:
        from shumary.modelhub import whisper_model

        mapped = OPENAI_MODEL_NAMES.get(model_name, model_name)
        model = whisper_model(mapped)

        opts: dict = dict(
            word_timestamps=True,
            verbose=False,
            condition_on_previous_text=False,
        )
        if fast:
            opts["temperature"] = 0.0

        result = model.transcribe(str(wav), **opts)

        language = result.get("language", "unknown")
        text = result.get("text", "")
        word_count = len(text.split())

        segments = tuple(
            Segment(
                start=float(s["start"]),
                end=float(s["end"]),
                text=s["text"],
                avg_logprob=float(s.get("avg_logprob", 0.0)),
                no_speech_prob=float(s.get("no_speech_prob", 0.0)),
                words=_normalize_words(s.get("words", [])),
            )
            for s in result.get("segments", [])
        )

        result = TranscriptionResult(
            language=language,
            word_count=word_count,
            text=text,
            segments=segments,
        )
        return _apply_hallucination_filter(result)


# ── faster-whisper backend ──────────────────────────────────────────────────────

class FasterBackend:
    name = "faster"

    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def transcribe(self, wav: Path, model_name: str, *, fast: bool) -> TranscriptionResult:
        model = _faster_model(FASTER_MODEL_NAMES.get(model_name, model_name))

        opts: dict = dict(
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        if fast:
            opts["beam_size"] = 1
            opts["temperature"] = 0.0

        seg_iter, info = model.transcribe(str(wav), **opts)

        # The generator streams lazily — must be fully consumed here, inside
        # the stage, before the model/process moves on.
        segments = []
        all_text_parts = []
        for s in seg_iter:
            words = _normalize_words(s.words or [])
            segments.append(
                Segment(
                    start=float(s.start),
                    end=float(s.end),
                    text=s.text,
                    avg_logprob=float(s.avg_logprob),
                    no_speech_prob=float(s.no_speech_prob),
                    words=words,
                )
            )
            all_text_parts.append(s.text)

        text = "".join(all_text_parts)
        word_count = len(text.split())

        return TranscriptionResult(
            language=info.language,
            word_count=word_count,
            text=text,
            segments=tuple(segments),
        )


@lru_cache(maxsize=None)
def _faster_model(name: str):
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]

    print(f"  Loading faster-whisper '{name}' (cpu, int8)…")
    return WhisperModel(name, device="cpu", compute_type="int8")


# ── mlx-whisper backend ─────────────────────────────────────────────────────────

class MlxBackend:
    name = "mlx"

    def available(self) -> bool:
        try:
            import mlx_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def transcribe(self, wav: Path, model_name: str, *, fast: bool) -> TranscriptionResult:
        import mlx_whisper  # type: ignore[import-untyped]

        repo = KNOWN_REPOS.get(model_name, model_name)

        opts: dict = dict(
            path_or_hf_repo=repo,
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        if fast:
            opts["temperature"] = 0.0

        result = mlx_whisper.transcribe(str(wav), **opts)

        language = result.get("language", "unknown")
        text = result.get("text", "")
        word_count = len(text.split())

        segments = tuple(
            Segment(
                start=float(s["start"]),
                end=float(s["end"]),
                text=s["text"],
                avg_logprob=float(s.get("avg_logprob", 0.0)),
                no_speech_prob=float(s.get("no_speech_prob", 0.0)),
                words=_normalize_words(s.get("words", [])),
            )
            for s in result.get("segments", [])
        )

        result = TranscriptionResult(
            language=language,
            word_count=word_count,
            text=text,
            segments=segments,
        )
        return _apply_hallucination_filter(result)


# ── Selection ────────────────────────────────────────────────────────────────────

ALL_BACKENDS: dict[str, TranscribeBackend] = {
    "mlx": MlxBackend(),
    "faster": FasterBackend(),
    "openai": OpenaiBackend(),
}


def _is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def pick_backend(override: str | None = None) -> TranscribeBackend:
    """Pick a transcription backend.

    Selection order:
      1. `override` (from `--backend` or `SHUMARY_BACKEND`) — error if named
         backend is unavailable.
      2. `mlx` if on Apple Silicon and `mlx_whisper` imports.
      3. `faster` if `faster_whisper` imports.
      4. `openai` (fallback — assumed always installed).
    """
    name = override or os.environ.get("SHUMARY_BACKEND")
    if name:
        backend = ALL_BACKENDS.get(name)
        if backend is None:
            raise ShumaryError(
                f"Unknown backend {name!r} — choose from: {', '.join(ALL_BACKENDS)}"
            )
        if not backend.available():
            raise ShumaryError(
                f"Backend {name!r} is not installed/available on this machine."
            )
        return backend

    if _is_apple_silicon() and ALL_BACKENDS["mlx"].available():
        return ALL_BACKENDS["mlx"]

    if ALL_BACKENDS["faster"].available():
        return ALL_BACKENDS["faster"]

    return ALL_BACKENDS["openai"]


__all__ = [
    "TranscribeBackend",
    "MlxBackend",
    "FasterBackend",
    "OpenaiBackend",
    "ALL_BACKENDS",
    "KNOWN_REPOS",
    "FASTER_MODEL_NAMES",
    "OPENAI_MODEL_NAMES",
    "pick_backend",
]

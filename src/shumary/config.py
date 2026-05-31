"""Runtime configuration — tunables and multilingual constants."""

import os

# ── Whisper ────────────────────────────────────────────────────────────────────

WHISPER_MODEL: str = os.environ.get("SHUMARY_WHISPER_MODEL", "large")
WHISPER_DEVICE: str | None = os.environ.get("SHUMARY_WHISPER_DEVICE")

# ── Pyannote ───────────────────────────────────────────────────────────────────

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN_ENVVARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN")

# ── Audio ──────────────────────────────────────────────────────────────────────

SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".webm", ".aac"}
)

# ── Mood signal thresholds ─────────────────────────────────────────────────────

LONG_PAUSE_SEC: float = 3.0
LOW_CONF_THRESHOLD: float = -1.0
HIGH_SILENCE_PROB: float = 0.6
SLOW_RATE: float = 1.8
FAST_RATE: float = 3.5
BASELINE_RATE: float = 2.5

# ── Filler words ───────────────────────────────────────────────────────────────
# Hand-curated trilingual set. English is always merged in regardless of source language.

FILLER_WORDS: dict[str, set[str]] = {
    "en": {
        "uh", "um", "uhh", "umm", "er", "err", "hmm", "hm", "ugh",
        "like", "basically", "literally", "actually", "you know",
    },
    "ru": {
        "ну", "эм", "эхм", "а", "вот", "это", "значит", "короче",
        "как бы", "типа", "собственно", "в общем",
    },
    "pl": {
        "eee", "yyy", "no", "właśnie", "tak", "jakby",
        "znaczy", "bo", "kurde", "no to",
    },
}

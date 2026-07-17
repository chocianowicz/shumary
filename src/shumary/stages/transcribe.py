"""Whisper transcription stage — delegates to a `TranscribeBackend` (Spec 3)."""

from pathlib import Path

from shumary.backends import TranscribeBackend, pick_backend
from shumary.config import WHISPER_MODEL
from shumary.models import TranscriptionResult


def run(
    audio: Path,
    model_name: str = WHISPER_MODEL,
    *,
    fast: bool = False,
    backend: TranscribeBackend | None = None,
) -> TranscriptionResult:
    backend = backend or pick_backend()
    print(f"  backend: {backend.name}")
    return backend.transcribe(audio, model_name, fast=fast)

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
    language: str | None = None,
    backend: TranscribeBackend | None = None,
    progress=None,
) -> TranscriptionResult:
    backend = backend or pick_backend()
    print(f"  backend: {backend.name}")
    if language:
        print(f"  language override: {language}")
    return backend.transcribe(audio, model_name, fast=fast, language=language, progress=progress)

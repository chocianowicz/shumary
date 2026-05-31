"""Audio file validation and metadata extraction."""

import sys
from pathlib import Path

from shumary.config import SUPPORTED_FORMATS
from shumary.utils import get_duration_seconds, slugify, fmt_duration


def validate(audio: Path) -> None:
    if not audio.exists():
        print(f"❌  File not found: {audio}")
        sys.exit(1)
    if audio.suffix.lower() not in SUPPORTED_FORMATS:
        print(f"❌  Unsupported format: {audio.suffix}")
        print(f"    Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")
        sys.exit(1)


def duration_seconds(audio: Path) -> float | None:
    return get_duration_seconds(audio)


__all__ = ["validate", "duration_seconds", "slugify", "fmt_duration"]

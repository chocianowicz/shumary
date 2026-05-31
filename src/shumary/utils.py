"""Shared utilities — formatting, slugification, frontmatter I/O."""

import re
import subprocess
from pathlib import Path

import frontmatter


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def fmt_time(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}m {s:02d}s"


def get_duration_seconds(audio: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def read_frontmatter(path: Path) -> dict:
    post = frontmatter.load(str(path))
    return dict(post.metadata)



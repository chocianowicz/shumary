"""Shared utilities — formatting, slugification, frontmatter I/O."""

import re
import subprocess
import wave
from datetime import date, datetime
from pathlib import Path

_DATE_IN_FILENAME_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")


def fmt_time(seconds: float) -> str:
    """Format a timestamp as `MM:SS`, or `H:MM:SS` once it reaches an hour."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def fmt_duration(seconds: float | None) -> str:
    """Format a duration as `Xm Ys`, or `XhYmZs` once it reaches an hour."""
    if seconds is None:
        return "unknown"
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s:02d}s"
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


def wav_duration_seconds(wav: Path) -> float | None:
    """Read duration from a WAV file's header via the stdlib `wave` module."""
    try:
        with wave.open(str(wav), "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            if rate <= 0:
                return None
            return frames / float(rate)
    except (OSError, wave.Error):
        return None


def resolve_recording_date(audio: Path, override: str | None = None) -> tuple[str, str]:
    """Resolve the recording date for `audio`.

    Resolution order:
      1. `--date` CLI override, if given.
      2. A `YYYY-MM-DD` substring in the filename, if present and valid.
      3. The audio file's mtime date.
      4. Today, as a last resort (e.g. file doesn't exist yet / mtime unreadable).

    Returns `(date_str, source)` where `source` is one of
    `"--date override"`, `"filename"`, `"file mtime"`, `"today"`.
    """
    if override:
        return override, "--date override"

    m = _DATE_IN_FILENAME_RE.search(audio.stem)
    if m:
        try:
            date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return m.group(0), "filename"
        except ValueError:
            pass  # not a real date — fall through

    try:
        mtime = audio.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d"), "file mtime"
    except OSError:
        pass

    return datetime.now().strftime("%Y-%m-%d"), "today"



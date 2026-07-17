"""Cache and output path helpers."""

import os
from pathlib import Path


def _default_cache_root() -> Path:
    override = os.environ.get("SHUMARY_CACHE_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "shumary"


CACHE: Path = _default_cache_root()


def cache_dir_for(slug: str) -> Path:
    return CACHE / slug


def output_path_for(audio: Path, output: Path | None = None) -> Path:
    """Where the rendered transcript lands.

    Default: `<audio-stem>.md` next to the input. `output` may be an existing
    directory (transcript goes inside it, named after the audio) or an explicit
    file path.
    """
    if output is None:
        return audio.with_suffix(".md")
    if output.is_dir():
        return output / f"{audio.stem}.md"
    return output

"""Canonical path constants derived from the repo root."""

from pathlib import Path

# Resolve repo root relative to this file: src/shumary/paths.py → ../../..
REPO_ROOT: Path = Path(__file__).parent.parent.parent

INBOX: Path = REPO_ROOT / "Inbox"
RAW_AUDIO: Path = REPO_ROOT / "raw" / "audio"
RAW_TRANSCRIPTS: Path = REPO_ROOT / "raw" / "transcripts"
RAW_ASSETS: Path = REPO_ROOT / "raw" / "assets"
WIKI: Path = REPO_ROOT / "wiki"
CACHE: Path = REPO_ROOT / ".cache"


def cache_dir_for(slug: str) -> Path:
    return CACHE / slug


def transcript_path_for(date_str: str, slug: str) -> Path:
    return RAW_TRANSCRIPTS / f"{date_str}-{slug}.md"


def archived_audio_path_for(audio: Path) -> Path:
    return RAW_AUDIO / audio.name


def relative_to_repo(p: Path) -> Path:
    if not p.is_absolute():
        return p
    return p.relative_to(REPO_ROOT)

"""Audio file validation, pre-decoding, and metadata extraction."""

import shutil
import subprocess
from pathlib import Path

from shumary.config import SUPPORTED_FORMATS
from shumary.errors import ShumaryError
from shumary.paths import cache_dir_for
from shumary.utils import get_duration_seconds, wav_duration_seconds

WAV_FILENAME = "audio.wav"
LEFT_WAV_FILENAME = "left.wav"
RIGHT_WAV_FILENAME = "right.wav"


def list_audio_files(directory: Path) -> list[Path]:
    """Supported audio files in `directory`, sorted by name."""
    if not directory.exists():
        return []
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    )


def validate(audio: Path) -> None:
    if not audio.exists():
        raise ShumaryError(f"File not found: {audio}")
    if audio.suffix.lower() not in SUPPORTED_FORMATS:
        raise ShumaryError(
            f"Unsupported format: {audio.suffix} "
            f"(supported: {', '.join(sorted(SUPPORTED_FORMATS))})"
        )


def duration_seconds(audio: Path) -> float | None:
    return get_duration_seconds(audio)


def wav_path_for(slug: str) -> Path:
    return cache_dir_for(slug) / WAV_FILENAME


def decode_to_wav(audio: Path, slug: str) -> Path:
    """Pre-decode `audio` to 16 kHz mono PCM WAV in the slug's cache dir.

    Reuses an existing WAV from a resumed run. All downstream stages
    (transcribe, diarize) consume this WAV instead of the original file.
    """
    out = wav_path_for(slug)
    if out.exists():
        return out

    if shutil.which("ffmpeg") is None:
        raise ShumaryError(
            "ffmpeg is not installed — required to pre-decode audio. "
            "Install it with: brew install ffmpeg"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(tmp),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        if tmp.exists():
            tmp.unlink()
        raise ShumaryError(
            f"ffmpeg failed to decode {audio.name}: {exc.stderr.strip()}"
        ) from exc

    tmp.replace(out)
    return out


def channel_wav_paths(slug: str) -> tuple[Path, Path]:
    d = cache_dir_for(slug)
    return d / LEFT_WAV_FILENAME, d / RIGHT_WAV_FILENAME


def channel_count(audio: Path) -> int | None:
    """Number of audio channels in `audio`, or None if ffprobe can't tell."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=channels",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def split_channels(audio: Path, slug: str) -> tuple[Path, Path]:
    """Decode a stereo recording into separate 16 kHz mono left/right WAVs.

    Used for dual-source recordings, where the two channels are independent
    sources (mic and system audio) rather than a stereo image of one source.
    """
    left, right = channel_wav_paths(slug)
    if left.exists() and right.exists():
        return left, right

    if shutil.which("ffmpeg") is None:
        raise ShumaryError(
            "ffmpeg is not installed — required to split recording channels. "
            "Install it with: brew install ffmpeg"
        )

    left.parent.mkdir(parents=True, exist_ok=True)
    tmp_left = left.with_suffix(".tmp.wav")
    tmp_right = right.with_suffix(".tmp.wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(audio),
                "-filter_complex", "[0:a]channelsplit=channel_layout=stereo[l][r]",
                "-map", "[l]", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp_left),
                "-map", "[r]", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(tmp_right),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        tmp_left.unlink(missing_ok=True)
        tmp_right.unlink(missing_ok=True)
        raise ShumaryError(
            f"ffmpeg failed to split channels of {audio.name}: {exc.stderr.strip()}"
        ) from exc

    tmp_left.replace(left)
    tmp_right.replace(right)
    return left, right


__all__ = [
    "list_audio_files",
    "validate",
    "duration_seconds",
    "decode_to_wav",
    "wav_path_for",
    "wav_duration_seconds",
    "channel_count",
    "channel_wav_paths",
    "split_channels",
]

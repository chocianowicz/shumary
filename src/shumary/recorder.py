"""Microphone + system-audio recording via the `shumary-record` native helper.

The helper is a small Swift program (`native/Recorder.swift` + `native/main.swift`)
that uses Core Audio process taps to capture the mic and the system output through
one aggregate device, writing a stereo WAV where the left channel is the mic and
the right channel is everything the machine played. `Recorder.swift` is shared
verbatim with the menu bar app in `app/`.

It is compiled on demand into the cache dir and reused until its source
changes, so nothing needs building at install time and no compiler is required
unless the user actually records something.

A recording is paired with a small JSON sidecar (`<name>.shumary.json`) that
marks the file as dual-source. The pipeline reads it to decide whether the two
channels can be trusted as "you" vs "them" instead of guessing with pyannote.
"""

import json
import os
import shutil
import signal
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path

from shumary.errors import ShumaryError
from shumary.paths import CACHE

HELPER_NAME = "shumary-record"
SIDECAR_SUFFIX = ".shumary.json"
SIDECAR_VERSION = 1

# Label used for the microphone channel. Unlike SPEAKER_XX this is not a guess:
# whatever arrives on the mic is by construction the person at the keyboard.
# Deliberately "Me" rather than a real name — it needs no configuration and
# keeps transcripts shareable without identifying anyone.
MIC_LABEL = "Me"

_FRAMEWORKS = ("CoreAudio", "AudioToolbox", "AVFoundation", "Foundation")


def helper_sources() -> list[Path]:
    """The Swift sources the helper is built from, in a stable order.

    `Recorder.swift` is shared with the menu bar app; `main.swift` is the CLI
    front end. Both must be passed to swiftc together.
    """
    return sorted((Path(__file__).parent / "native").glob("*.swift"))


def _helper_binary_path() -> Path:
    """Cache path for the compiled helper, keyed by a hash of its sources.

    Editing any Swift source yields a new path, so a stale binary can never be
    silently reused.
    """
    h = hashlib.sha256()
    for source in helper_sources():
        h.update(source.read_bytes())
    return CACHE / "bin" / f"{HELPER_NAME}-{h.hexdigest()[:12]}"


def build_helper(*, force: bool = False) -> Path:
    """Compile the native helper if needed and return the binary's path."""
    binary = _helper_binary_path()
    if binary.exists() and not force:
        return binary

    if shutil.which("swiftc") is None:
        raise ShumaryError(
            "swiftc is not installed — needed once to build the audio recorder. "
            "Install the Xcode command line tools with: xcode-select --install"
        )

    binary.parent.mkdir(parents=True, exist_ok=True)
    tmp = binary.with_name(binary.name + ".tmp")
    cmd = ["swiftc", "-O", *(str(s) for s in helper_sources()), "-o", str(tmp)]
    for framework in _FRAMEWORKS:
        cmd += ["-framework", framework]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise ShumaryError(f"could not build the audio recorder:\n{result.stderr.strip()}")

    os.replace(tmp, binary)
    return binary


def helper_status() -> tuple[bool, str]:
    """`(ok, note)` for `shumary doctor`."""
    if _helper_binary_path().exists():
        return True, "built"
    if shutil.which("swiftc") is None:
        return False, "xcode-select --install  (needed to build the recorder)"
    return True, "builds on first use"


# ── Sidecar ──────────────────────────────────────────────────────────────────

def sidecar_path(audio: Path) -> Path:
    return audio.with_suffix(audio.suffix + SIDECAR_SUFFIX)


def write_sidecar(audio: Path, *, mic_label: str = MIC_LABEL) -> Path:
    path = sidecar_path(audio)
    payload = {
        "tool": HELPER_NAME,
        "version": SIDECAR_VERSION,
        "dual_source": True,
        "channels": {"left": "mic", "right": "system"},
        "mic_label": mic_label,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_sidecar(audio: Path) -> dict | None:
    path = sidecar_path(audio)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def dual_source_info(audio: Path) -> dict | None:
    """Sidecar payload if `audio` is a mic+system recording, else None."""
    data = read_sidecar(audio)
    if data and data.get("dual_source"):
        return data
    return None


# ── Recording ────────────────────────────────────────────────────────────────

def default_output_path(directory: Path | None = None) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    return (directory or Path.cwd()) / f"recording-{stamp}.wav"


def record(output: Path) -> Path:
    """Record mic + system audio to `output` until interrupted.

    Blocks until the user presses Ctrl-C (or the helper exits on its own), then
    writes the sidecar and returns the WAV path.
    """
    binary = build_helper()
    output = output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Own process group, so Ctrl-C reaches this process first and the helper is
    # signalled deliberately — otherwise both get SIGINT and the helper can be
    # torn down mid-write while finalizing the WAV header.
    process = subprocess.Popen([str(binary), str(output)], start_new_session=True)

    try:
        process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

    if not output.exists() or output.stat().st_size <= 44:  # header only
        raise ShumaryError(
            "recording produced no audio — check that the terminal has microphone "
            "and system-audio permission in System Settings → Privacy & Security."
        )

    write_sidecar(output)
    return output


__all__ = [
    "MIC_LABEL",
    "build_helper",
    "default_output_path",
    "dual_source_info",
    "helper_sources",
    "helper_status",
    "read_sidecar",
    "record",
    "sidecar_path",
    "write_sidecar",
]

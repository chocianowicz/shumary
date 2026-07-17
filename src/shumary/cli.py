"""Shumary CLI — Typer-based entry point."""

import os
import subprocess
import warnings
from functools import lru_cache

# Suppress noisy torchcodec/pyannote import-time warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
from pathlib import Path
from typing import Annotated, Callable, Optional

import typer

from shumary import cache as cache_mod
from shumary.config import WHISPER_MODEL
from shumary.paths import CACHE

app = typer.Typer(
    name="shumary",
    help="Audio → detailed markdown transcripts with diarization, mood, and prosody signals.",
    add_completion=False,
)
cache_app = typer.Typer(name="cache", help="Inspect or clear pipeline cache.")
app.add_typer(cache_app, name="cache")


@lru_cache(maxsize=None)
def _console():
    from rich.console import Console
    return Console()


def _table(*columns: str):
    from rich.table import Table
    return Table(*columns)


# ── shumary transcribe ───────────────────────────────────────────────────────────

@app.command()
def transcribe(
    path: Annotated[Path, typer.Argument(help="Audio file, or a directory to process every audio file in it.")],
    output: Annotated[Optional[Path], typer.Option(
        "--output", "-o",
        help="Output .md path, or a directory (batch mode). Defaults to next to the input.",
    )] = None,
    force: bool = typer.Option(False, "--force", "-f", help="Re-run all stages, ignoring cache."),
    no_diarize: bool = typer.Option(False, "--no-diarize", help="Skip speaker diarization."),
    solo: bool = typer.Option(
        False, "--solo",
        help="Single-speaker recording — skip diarization entirely and label everything SPEAKER_00.",
    ),
    fast: bool = typer.Option(False, "--fast", help="Greedy Whisper decode (beam_size=1, single temperature). ~25-35% faster, minimal quality trade-off on clean speech."),
    model: str = typer.Option(
        WHISPER_MODEL,
        "--model", "-m",
        help=(
            "Whisper model size. Options: tiny, base, small, medium, large (default), turbo. "
            "'turbo' is ~5-8x faster than 'large' with minor quality drift on hard audio. "
            "'medium' is ~3x faster with more drift. "
            "Combine with --fast for maximum speed: --model turbo --fast."
        ),
    ),
    date: Optional[str] = typer.Option(
        None, "--date",
        help="Recording date as YYYY-MM-DD, overriding filename/mtime detection.",
    ),
    backend: Optional[str] = typer.Option(
        None, "--backend",
        help=(
            "Transcription backend: mlx, faster, or openai. "
            "Defaults to auto-detection (mlx on Apple Silicon, then faster-whisper, "
            "then openai). Can also be set via SHUMARY_BACKEND."
        ),
    ),
    no_plain_text: bool = typer.Option(
        False, "--no-plain-text",
        help="Omit the '## Plain Text' section (the full transcript text is "
             "already in '## Full Transcript'; this halves file size).",
    ),
) -> None:
    """Transcribe one audio file, or every audio file in a directory."""
    from shumary import pipeline
    from shumary.errors import ShumaryError

    if date is not None:
        try:
            from datetime import date as _date
            y, m, d = (int(p) for p in date.split("-"))
            _date(y, m, d)
        except (ValueError, TypeError):
            _console().print(f"[red]❌  Invalid --date {date!r} — expected YYYY-MM-DD.[/red]")
            raise typer.Exit(code=1)

    plain_text = not no_plain_text

    if not path.exists():
        _console().print(f"[red]❌  File not found: {path}[/red]")
        raise typer.Exit(code=1)

    if path.is_dir():
        try:
            _succeeded, failed = pipeline.process_directory(
                path, force=force, no_diarize=no_diarize, solo=solo, fast=fast, model_name=model,
                date_override=date, backend_override=backend, plain_text=plain_text,
                output_dir=output,
            )
        except ShumaryError as exc:
            _console().print(f"[red]❌  {exc}[/red]")
            raise typer.Exit(code=1)
        if failed:
            raise typer.Exit(code=1)
    else:
        try:
            pipeline.process_file(
                path, force=force, no_diarize=no_diarize, solo=solo, fast=fast, model_name=model,
                date_override=date, backend_override=backend, plain_text=plain_text,
                output=output,
            )
        except ShumaryError as exc:
            _console().print(f"[red]❌  {exc}[/red]")
            raise typer.Exit(code=1)


# ── shumary cache list / clear ─────────────────────────────────────────────────

@cache_app.command("list")
def cache_list(
    slug: Optional[str] = typer.Argument(None, help="Show status for this slug only."),
) -> None:
    """List pipeline cache state."""
    slugs = [slug] if slug else cache_mod.list_slugs()
    if not slugs:
        _console().print("[dim]Cache is empty.[/dim]")
        return

    t = _table("Slug", "whisper", "diarize", "align", "mood", "done", "fingerprint")
    for s in slugs:
        status_map = cache_mod.stage_status(s)
        t.add_row(
            s,
            "✓" if status_map["whisper"] else "·",
            "✓" if status_map["diarize"] else "·",
            "✓" if status_map["align"] else "·",
            "✓" if status_map["mood"] else "·",
            "✓" if status_map["done"] else "·",
            cache_mod.fingerprint_info(s),
        )
    _console().print(t)


@cache_app.command("clear")
def cache_clear(
    slug: Optional[str] = typer.Argument(None, help="Clear cache for this slug. Omit to clear all."),
    stage: Optional[str] = typer.Option(None, "--stage", "-s", help="Clear only this stage."),
) -> None:
    """Clear pipeline cache."""
    if slug is None:
        if not CACHE.exists():
            _console().print("[dim]Cache already empty.[/dim]")
            return
        for d in CACHE.iterdir():
            cache_mod.clear(d.name)
        _console().print("Cache cleared.")
    else:
        cache_mod.clear(slug, stage)
        msg = f"Cleared cache for `{slug}`"
        if stage:
            msg += f" (stage: {stage})"
        _console().print(msg)


# ── shumary doctor ─────────────────────────────────────────────────────────────

def _pyannote_weights_cached() -> bool:
    """True if any pyannote model weights are present in the local HF hub cache."""
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub_dir.exists():
        return False
    return any(hub_dir.glob("models--pyannote--*"))


# Table-driven import checks for `doctor`: (row label, module to import,
# success-note callback taking the imported module, install hint).
_IMPORT_CHECKS: list[tuple[str, str, Callable[[object], str], str]] = [
    (
        "torch", "torch",
        lambda torch: (
            f"{torch.__version__} — "
            + ("MPS ✅" if torch.backends.mps.is_available()
               else "CUDA ✅" if torch.cuda.is_available()
               else "CPU only")
        ),
        "pip install torch",
    ),
    ("pyannote", "pyannote.audio", lambda m: m.__version__, "pip install pyannote.audio"),
    ("resemblyzer", "resemblyzer", lambda m: "tokenless fallback available", "pip install resemblyzer  (optional)"),
]


@app.command()
def doctor() -> None:
    """Check system dependencies, models, and credentials."""
    _console().print("\n[bold]Shumary Doctor[/bold]\n")

    t = _table("Check", "Status", "Notes")

    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        t.add_row("ffprobe", "✅", "")
    except (FileNotFoundError, subprocess.CalledProcessError):
        t.add_row("ffprobe", "❌", "brew install ffmpeg")

    for label, module_name, note_fn, install_hint in _IMPORT_CHECKS:
        try:
            module = __import__(module_name, fromlist=["_"])
            t.add_row(label, "✅", note_fn(module))
        except ImportError:
            marker = "⚠ " if label == "resemblyzer" else "❌"
            t.add_row(label, marker, install_hint)

    from shumary.backends import ALL_BACKENDS, pick_backend

    try:
        chosen = pick_backend()
        chosen_name = chosen.name
    except Exception:
        chosen_name = None

    backend_install_hints = {
        "mlx": "pip install mlx-whisper",
        "faster": "pip install faster-whisper",
        "openai": "pip install openai-whisper",
    }
    for name, b in ALL_BACKENDS.items():
        available = b.available()
        marker = "✅" if available else "❌"
        note = backend_install_hints[name] if not available else ""
        if name == chosen_name:
            note = (note + "  " if note else "") + "← auto-picked"
        t.add_row(f"backend: {name}", marker, note)

    from shumary.config import HF_TOKEN_ENVVARS
    hf_token = next((os.environ.get(v, "").strip() for v in HF_TOKEN_ENVVARS if os.environ.get(v)), "")
    if hf_token:
        t.add_row("HF_TOKEN", "✅", f"{hf_token[:8]}…")
    else:
        t.add_row("HF_TOKEN", "⚠ ", "Using resemblyzer fallback")

    pyannote_cached = _pyannote_weights_cached()
    if pyannote_cached:
        if os.environ.get("HF_HUB_OFFLINE", "").strip() in ("1", "true", "True"):
            t.add_row("HF_HUB_OFFLINE", "✅", "set — no network calls")
        else:
            t.add_row(
                "HF_HUB_OFFLINE", "⚠ ",
                "pyannote weights are cached — set HF_HUB_OFFLINE=1 for guaranteed-offline runs",
            )

    _console().print(t)
    _console().print()

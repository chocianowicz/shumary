"""Shumary CLI — Typer-based entry point."""

import json
import os
import subprocess
import warnings
from functools import lru_cache

# Suppress noisy torchcodec/pyannote import-time warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from shumary import cache as cache_mod
from shumary.config import WHISPER_MODEL, SUPPORTED_FORMATS
from shumary.paths import INBOX, WIKI, REPO_ROOT, CACHE

app = typer.Typer(
    name="shumary",
    help="Personal LLM-maintained wiki fed by audio recordings.",
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


# ── shumary ingest ─────────────────────────────────────────────────────────────

@app.command()
def ingest(
    path: Annotated[Optional[Path], typer.Argument(help="Audio file to ingest, or omit for all of Inbox/")] = None,
    force: bool = typer.Option(False, "--force", "-f", help="Re-run all stages, ignoring cache."),
    no_diarize: bool = typer.Option(False, "--no-diarize", help="Skip speaker diarization."),
    model: str = typer.Option(WHISPER_MODEL, "--model", "-m", help="Whisper model size."),
) -> None:
    """Transcribe audio files and save rich markdown transcripts."""
    from shumary import pipeline

    if path is None:
        pipeline.process_inbox(force=force, no_diarize=no_diarize, model_name=model)
    else:
        if not path.is_absolute():
            path = REPO_ROOT / path
        pipeline.process_file(path, force=force, no_diarize=no_diarize, model_name=model)


# ── shumary status ─────────────────────────────────────────────────────────────

@app.command()
def status() -> None:
    """Show pending transcripts, wiki stats, and recent log entries."""
    from shumary import checks

    inbox_files = [
        f for f in INBOX.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ] if INBOX.exists() else []

    pending = checks.pending_transcripts()
    ingested = checks.ingested_transcripts()
    stats = checks.wiki_stats()

    console = _console()
    console.print("\n[bold]Shumary Status[/bold]\n")

    t = _table()
    t.show_header = False
    t.box = None
    t.padding = (0, 2)
    t.add_column("", style="dim")
    t.add_column("")
    t.add_row("Inbox", f"{len(inbox_files)} audio file(s) waiting")
    t.add_row("Pending ingest", f"{len(pending)} transcript(s)")
    t.add_row("Ingested", f"{len(ingested)} transcript(s)")
    t.add_row("Wiki pages", str(stats["total_pages"]))
    t.add_row("Source pages", str(stats["source_pages"]))
    console.print(t)

    if pending:
        console.print("\n[yellow]Pending transcripts:[/yellow]")
        for p in pending:
            console.print(f"  {p.name}")

    log = WIKI / "log.md"
    if log.exists():
        entries = [
            line for line in log.read_text(encoding="utf-8").splitlines()
            if line.startswith("## [")
        ]
        if entries:
            console.print("\n[dim]Recent activity:[/dim]")
            for entry in entries[-3:]:
                console.print(f"  {entry}")

    console.print()


# ── shumary lint ───────────────────────────────────────────────────────────────

@app.command()
def lint(
    as_json: bool = typer.Option(False, "--json", help="Output as JSON array."),
) -> None:
    """Run structural health checks on the wiki."""
    from shumary import checks

    items = checks.run_all()
    if as_json:
        print(json.dumps(items))
    else:
        _console().print("\n[bold]Wiki Lint[/bold]\n")
        for item in items:
            _console().print(item)
        _console().print()


# ── shumary open ──────────────────────────────────────────────────────────────

@app.command()
def open(
    target: str = typer.Argument("wiki", help="What to open: 'wiki' (default), a transcript path, or audio path."),
) -> None:
    """Open the wiki in Obsidian, or a specific file."""
    if target == "wiki":
        path = str(WIKI)
    else:
        p = Path(target)
        path = str(p if p.is_absolute() else REPO_ROOT / p)
    subprocess.run(["open", "-a", "Obsidian", path])


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

    t = _table("Slug", "whisper", "diarize", "align", "mood", "done")
    for s in slugs:
        status_map = cache_mod.stage_status(s)
        t.add_row(
            s,
            "✓" if status_map["whisper"] else "·",
            "✓" if status_map["diarize"] else "·",
            "✓" if status_map["align"] else "·",
            "✓" if status_map["mood"] else "·",
            "✓" if status_map["done"] else "·",
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

    try:
        import torch
        mps = torch.backends.mps.is_available()
        cuda = torch.cuda.is_available()
        device = "MPS ✅" if mps else ("CUDA ✅" if cuda else "CPU only")
        t.add_row("torch", "✅", f"{torch.__version__} — {device}")
    except ImportError:
        t.add_row("torch", "❌", "pip install torch")

    try:
        import whisper
        t.add_row("whisper", "✅", f"model cache: {WHISPER_MODEL}")
    except ImportError:
        t.add_row("whisper", "❌", "pip install openai-whisper")

    try:
        import pyannote.audio
        t.add_row("pyannote", "✅", pyannote.audio.__version__)
    except ImportError:
        t.add_row("pyannote", "❌", "pip install pyannote.audio")

    try:
        import resemblyzer
        t.add_row("resemblyzer", "✅", "tokenless fallback available")
    except ImportError:
        t.add_row("resemblyzer", "⚠ ", "pip install resemblyzer  (optional)")

    from shumary.config import HF_TOKEN_ENVVARS
    hf_token = next((os.environ.get(v, "").strip() for v in HF_TOKEN_ENVVARS if os.environ.get(v)), "")
    if hf_token:
        t.add_row("HF_TOKEN", "✅", f"{hf_token[:8]}…")
    else:
        t.add_row("HF_TOKEN", "⚠ ", "Using resemblyzer fallback")

    try:
        import frontmatter
        t.add_row("python-frontmatter", "✅", "")
    except ImportError:
        t.add_row("python-frontmatter", "❌", "pip install python-frontmatter")

    _console().print(t)
    _console().print()

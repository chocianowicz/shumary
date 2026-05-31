"""Code-side wiki health checks — the things that don't require LLM judgment."""

import re
from pathlib import Path

from shumary.paths import WIKI, RAW_TRANSCRIPTS
from shumary.utils import read_frontmatter


def pending_transcripts() -> list[Path]:
    return _transcripts_with_status("pending-ingest")


def ingested_transcripts() -> list[Path]:
    return _transcripts_with_status("ingested")


def _transcripts_with_status(status: str) -> list[Path]:
    if not RAW_TRANSCRIPTS.exists():
        return []
    result = []
    for p in sorted(RAW_TRANSCRIPTS.glob("*.md")):
        try:
            if read_frontmatter(p).get("status") == status:
                result.append(p)
        except Exception:
            pass
    return result


def orphan_pages(wiki: Path = WIKI) -> list[Path]:
    """Pages that have no inbound [[WikiLinks]] from any other wiki page."""
    pages = list(wiki.rglob("*.md"))
    inbound: dict[str, int] = {p.stem: 0 for p in pages}
    link_re = re.compile(r"\[\[([^\]|#]+)")
    for page in pages:
        text = page.read_text(encoding="utf-8", errors="ignore")
        for m in link_re.finditer(text):
            target = m.group(1).strip()
            if target in inbound:
                inbound[target] += 1
    meta = {"index", "log", "overview"}
    return [p for p in pages if inbound.get(p.stem, 0) == 0 and p.stem not in meta]


def index_dangling_refs(wiki: Path = WIKI) -> list[str]:
    """Links in index.md that point to pages that don't exist on disk."""
    index = wiki / "index.md"
    if not index.exists():
        return []
    text = index.read_text(encoding="utf-8", errors="ignore")
    link_re = re.compile(r"\[\[([^\]|#]+)")
    existing = {p.stem for p in wiki.rglob("*.md")}
    dangling = []
    for m in link_re.finditer(text):
        target = m.group(1).strip()
        if target not in existing:
            dangling.append(target)
    return sorted(set(dangling))


def transcripts_missing_log_entries() -> list[Path]:
    """Ingested transcripts whose slug doesn't appear in wiki/log.md."""
    ingested = ingested_transcripts()
    if not ingested:
        return []
    log = WIKI / "log.md"
    if not log.exists():
        return ingested
    log_text = log.read_text(encoding="utf-8", errors="ignore")
    return [p for p in ingested if p.stem not in log_text]


def wiki_stats(wiki: Path = WIKI) -> dict:
    pages = list(wiki.rglob("*.md"))
    sources = [p for p in pages if _get_type(p) == "source"]
    return {
        "total_pages": len(pages),
        "source_pages": len(sources),
        "pending_transcripts": len(pending_transcripts()),
        "ingested_transcripts": len(ingested_transcripts()),
    }


def run_all() -> list[str]:
    """Run all code-side checks and return a markdown checklist of findings."""
    items: list[str] = []

    pending = pending_transcripts()
    if pending:
        items.append(f"- [ ] **{len(pending)} transcript(s) pending ingest**: "
                     + ", ".join(f"`{p.name}`" for p in pending))
    else:
        items.append("- [x] No pending transcripts")

    orphans = orphan_pages()
    if orphans:
        items.append(f"- [ ] **{len(orphans)} orphan page(s)** (no inbound links): "
                     + ", ".join(f"`{p.name}`" for p in orphans))
    else:
        items.append("- [x] No orphan pages")

    dangling = index_dangling_refs()
    if dangling:
        items.append(f"- [ ] **{len(dangling)} dangling ref(s) in index.md**: "
                     + ", ".join(f"`{r}`" for r in dangling))
    else:
        items.append("- [x] No dangling index references")

    missing_log = transcripts_missing_log_entries()
    if missing_log:
        items.append(f"- [ ] **{len(missing_log)} ingested transcript(s) missing log entry**: "
                     + ", ".join(f"`{p.name}`" for p in missing_log))
    else:
        items.append("- [x] All ingested transcripts have log entries")

    items.append("")
    items.append("*Run in Claude Code for LLM-side checks: contradictions, stale claims, "
                 "missing cross-references, restructuring suggestions.*")

    return items


def _get_type(path: Path) -> str | None:
    try:
        return read_frontmatter(path).get("type")
    except Exception:
        return None

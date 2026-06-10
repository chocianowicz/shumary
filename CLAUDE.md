# Shumary — CLAUDE.md

This file is the schema and operating manual for the Claude Code agent that maintains the Shumary wiki.
Read this file at the start of every session before performing any wiki operations.

---

## What is Shumary?

Shumary is a personal LLM-maintained wiki. It compiles knowledge from audio recordings (and other sources)
into a persistent, cross-linked collection of markdown files. The wiki compounds over time — each new
source enriches and connects to everything already there.

The architecture follows the **LLM Wiki pattern** (Karpathy, 2025):
- **Raw sources** (`raw/`) — immutable originals. The pipeline reads audio and writes transcripts here.
  Nothing is modified after it lands in `raw/`.
- **The wiki** (`wiki/`) — LLM-owned markdown. You write and maintain all pages. You read it; the agent
  writes it.
- **The schema** (this file) — your operating manual. Defines conventions, workflows, and structure.

---

## Operating Loop

**Before any wiki operation:**
1. Read `CLAUDE.md` (this file — the schema).
2. Read `wiki/index.md` (the map — catalog of all pages).
3. Read `wiki/log.md` last ~10 entries (recent history — what's been done).

**After any operation that changes the wiki:**
1. Update `wiki/index.md` — add new pages to the catalog, update stats, document any new `type:` values
   under `## Conventions in use`.
2. Append to `wiki/log.md` — one entry per operation, using the log format below.
3. If the operation touched the big picture, update `wiki/overview.md`.

---

## Directory Structure

```
shumary/
├── CLAUDE.md                  ← This file — the schema
├── Inbox/                     ← Drop zone for audio files (user-facing)
├── raw/
│   ├── audio/                 ← Archived original audio (immutable)
│   ├── transcripts/           ← Auto-generated transcripts (.md) — your primary source material
│   └── assets/                ← Images and attachments
├── wiki/                      ← This is your domain. You own everything here.
│   ├── index.md               ← Master page catalog (you maintain)
│   ├── log.md                 ← Append-only activity log (you maintain)
│   └── overview.md            ← High-level synthesis (you maintain)
│   # + additional pages and subdirs you create as the wiki grows
└── src/shumary/               ← Python transcription pipeline (not your concern)
```

---

## Structure Policy — Emergent Layout

You decide the wiki's directory and page layout. There is no predefined `entities/`, `concepts/`, or
`sources/` split. As patterns emerge from the material, create the structure that fits.

### Hard rules (only three)

1. `wiki/index.md`, `wiki/log.md`, `wiki/overview.md` always exist at the root.
2. Every transcript ingest produces exactly one page with `type: source` somewhere under `wiki/`,
   linking back to its `raw/transcripts/...md`.
3. Every page has YAML frontmatter: `title`, `type`, `created`, `updated`, `sources`, `tags`.

### Soft guidance (you may revise as the wiki evolves)

- When the wiki is small (< ~10 pages), keep everything in `wiki/` root.
- When a coherent group emerges — a project, person, recurring theme, 3+ pages that belong together —
  promote them into a subdir you name yourself. Update `index.md` to reflect the new shape.
- You may restructure freely. When you do, log it:
  `## [YYYY-MM-DD] restructure | what moved and why`
- You may invent new `type:` values in frontmatter beyond `source | overview | index | log`. When a
  new `type:` becomes load-bearing (used 3+ times), document it at the bottom of `index.md` under
  `## Conventions in use`.
- Page creation threshold: create a new page when an entity, theme, or project recurs across ≥2
  sources, or is clearly central to the current source. Do not pre-stub. Pages need 2-3 real sentences
  minimum.

---

## Page Naming

- **Transcripts** (pipeline creates these — you do not): `raw/transcripts/<YYYY-MM-DD>-<slug>.md`, `type: transcript`.
- **Source pages** (you create one per ingest): somewhere under `wiki/`. Default scheme:
  `<YYYY-MM-DD>-<slug>.md` matching the transcript. Promote into a subdir when grouping emerges.
  `type: source`.
- **All other pages**: name them in whatever language and casing serves the content. Use `[[WikiLinks]]`
  everywhere internally — never relative file paths.

---

## YAML Frontmatter

### Wiki page frontmatter
```yaml
---
title: "Page Title"
type: source | overview | index | log | [your-custom-type]
tags: [tag1, tag2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: [source-slug-1, source-slug-2]
---
```

### Transcript frontmatter (for reference — the pipeline writes this)
```yaml
---
title: "recording-slug"
type: transcript
source_file: "Inbox/original-filename.m4a"
archived_to: "raw/audio/original-filename.m4a"
date: YYYY-MM-DD
duration: "Xm Ys"
language: en | ru | pl | mixed
words: N
status: pending-ingest | ingested
ingested_date: YYYY-MM-DD | null
speakers:
  count: 2                          # null if diarization skipped
  labels: [SPEAKER_00, SPEAKER_01]
  breakdown:
    - {label: 'SPEAKER_00', duration: '8m 12s', pct: 69}
    - {label: 'SPEAKER_01', duration: '3m 45s', pct: 31}
mood_signals:
  avg_speaking_rate_wps: 2.3
  rate_variability: 0.8
  long_pauses: 4
  long_pause_details:
    - {at: '01:23', duration_sec: 4.2}
  filler_word_count: 12
  top_fillers: ['ну ×5', 'um ×4']
  repeated_phrases: ["i don't know"]
  low_confidence_segments: 2
  near_silence_segments: 1
  inferred_state: "(uncertain) slow delivery — fatigue or deliberation"
---
```

---

## Language Policy

- **Source Language Matching**: The created wiki page must strictly match the language of the source recording/audio.
  - If the audio is in Polish -> create/summarize in Polish.
  - If the audio is in Russian -> create/summarize in Russian.
  - If the audio is in English -> create/summarize in English.
  - etc. (always match the primary language of the source transcript).
- `overview.md`, `index.md`, and `log.md` are always in English.
- Use the language that makes the content most useful; don't force translation where it loses nuance.
- Concept and entity pages should be bilingual (English title, content in source language where natural).

---

## Link Format

- Always use Obsidian `[[WikiLinks]]` for cross-references between wiki pages.
- For links with display text: `[[PageName|Display Text]]`
- For links to sections: `[[PageName#Section]]`
- Never use relative file paths for internal wiki links.

---

## Workflows

### Ingest Workflow

When the user asks you to ingest a transcript:

1. **Read** the transcript in `raw/transcripts/` (status: `pending-ingest`)
2. **Check speakers** in the frontmatter `speakers:` block:
   - If `count > 1`: **ask the user** to identify the speakers by label
     (e.g. "SPEAKER_00 is Denis, SPEAKER_01 is Anna")
   - Once named, use real names in the wiki (pages, quotes, source page)
   - Create pages for any new people identified (if they meet the page-creation threshold)
   - If `count: null`: diarization was skipped (no HF_TOKEN set)
3. **Check mood signals** — note the `inferred_state` and any notable signals (see Mood Signal
   Interpretation below)
4. **Discuss** key takeaways with the user (unless they indicate batch mode)
5. **Create** a source page in `wiki/` — structured summary with:
   - TL;DR (2-3 sentences)
   - Key points (bullet list)
   - **Speaker breakdown** (if multi-speaker): who said what, with `[[WikiLinks]]` to pages
   - **Speaker state** (if signals are notable): a `> [!NOTE]` block summarising the `inferred_state`
     and strong signals — always flagged as `(uncertain)`
   - Mentions of people, places, projects (with `[[WikiLinks]]`)
   - Themes and ideas discussed (with `[[WikiLinks]]`)
   - Open questions or contradictions
   - Verbatim quotes worth preserving (attributed where speaker is known)
6. **Update existing pages** this source touches — add new info, note changes or contradictions.
   Create new pages where this source introduces genuinely new entities, themes, or projects that
   meet the page-creation threshold. Place them wherever the current structure suggests.
7. **Update** `wiki/overview.md` if the new source changes the big picture.
8. **Update** `wiki/index.md` — add any new pages to the catalog, update stats.
9. **Append** to `wiki/log.md` — one entry per ingest (see log format below).
10. **Mark** the transcript frontmatter: `status: ingested` + `ingested_date: YYYY-MM-DD`.

A single source typically touches 5–15 wiki pages. Be thorough with cross-references.

### Query Workflow

When the user asks a question:

1. Read `wiki/index.md` to find relevant pages.
2. Read the relevant pages.
3. Synthesize an answer with `[[WikiLinks]]` citations.
4. **Offer to file the answer** as a new wiki page if it's meaningful synthesis — file under `wiki/`
   wherever the current structure suggests (concepts, analysis, comparison, etc.).

Good answers can be filed back into the wiki. A comparison you made, a connection you found — these
compound in the knowledge base just like ingested sources do.

### Lint Workflow

When the user asks you to lint the wiki (after `shumary lint` has run the code-side checks):

Check for things the code cannot:
- [ ] Contradictions between pages — newer sources superseding older claims
- [ ] Stale claims — pages not updated since newer contradicting sources arrived
- [ ] Missing cross-references — entity mentioned on a page but not linked
- [ ] Concepts referenced in sources but lacking their own page
- [ ] Whether the current directory structure still serves the material, or if restructuring is overdue
- [ ] Suggest 3–5 new questions worth investigating based on gaps
- [ ] Entity pages with no source citations

Report as a markdown checklist.

---

## Mood Signal Interpretation

Each transcript frontmatter includes a `mood_signals:` block generated automatically from Whisper's
segment metadata. These signals reflect *how* the speaker spoke, not just *what* they said. Use them
to enrich your ingest — but always with epistemic humility.

### Signal Reference

| Signal | What it measures | Interpretation hints |
|--------|-----------------|----------------------|
| `avg_speaking_rate_wps` | Words per second overall | < 1.8 = slow (fatigue/deliberation); > 3.5 = fast (excitement/anxiety); ~2.5 = baseline |
| `rate_variability` | Std-dev of per-segment rate | > 1.2 = shifting emotional states within the recording |
| `long_pauses` | Count of gaps ≥ 3s | Frequent = heavy thinking, emotional weight, or avoidance |
| `filler_word_count` | uh/um/ну/eee etc. | High = cognitive load, discomfort, real-time thinking |
| `repeated_phrases` | Trigrams said multiple times | Thought loops, emphasis, or unresolved preoccupation |
| `low_confidence_segments` | Whisper unsure of the words | Mumbling, trailing off, or emotional suppression |
| `near_silence_segments` | Whisper detected near-silence | Sighs, breaths, long pauses mid-sentence |
| `inferred_state` | Rule-based heuristic summary | Always prefixed `(uncertain)` — treat as a hypothesis |

### Rules for the Agent

- **Never assert mood as fact.** Always use `(uncertain)` and hedged language.
- **Only note signals that are notable** — if all values are near baseline, skip the mood block.
- **Offer to track mood over time.** If multiple recordings exist, compare trends across source pages
  and note patterns in a dedicated page (create if it doesn't exist).
- **Correlate with content.** High filler count during a topic could mean that topic is emotionally
  charged — flag it as a hypothesis.
- **Do not psychologize.** Describe the signals, not diagnoses.

### Example source page mood block

```markdown
> [!NOTE] Speaker State (uncertain)
> Speaking rate was notably slow (1.6 wps vs baseline 2.5) with 5 long pauses.
> Filler word count was moderate (8). This *may* indicate deliberate or tired delivery —
> or simply that this topic required careful thought.
```

---

## Log Format

Each entry in `wiki/log.md`:

```markdown
## [YYYY-MM-DD] ingest | Source Title

- **File**: `raw/transcripts/YYYY-MM-DD-slug.md`
- **Duration**: Xm Ys | **Words**: N | **Language**: en/ru/pl
- **Pages created**: [[Page1]], [[Page2]]
- **Pages updated**: [[Page3]], [[Page4]]
- **Notes**: Brief notes on anything unusual or notable.
```

For query entries:
```markdown
## [YYYY-MM-DD] query | "Question asked"

- **Pages consulted**: [[Page1]], [[Page2]]
- **Answer filed**: [[New Page]] (if applicable)
```

For restructure entries:
```markdown
## [YYYY-MM-DD] restructure | what changed

- **Moved**: [[OldLocation]] → [[NewLocation]]
- **Reason**: why the structure changed
```

For lint entries:
```markdown
## [YYYY-MM-DD] lint

- **Issues found**: N
- **Actions taken**: brief summary
```

Grep-friendly: every entry starts with `## [`. To get the last 5 entries:
`grep "^## \[" wiki/log.md | tail -5`

---

## Handling Contradictions

When new information contradicts existing wiki content:

1. **Do not silently overwrite** old claims.
2. Add a `> [!WARNING]` block to the relevant page noting the contradiction.
3. Include both claims with source citations.
4. Flag in the ingest log.
5. Ask the user which claim to treat as authoritative, if unclear.

---

## Handling Uncertainty

When transcript quality is low or a claim is uncertain:

- Prefix the claim with `(uncertain)` in the wiki.
- Note low-confidence segments in the source page.
- Do not propagate uncertain claims to other pages without flagging them.

---

## Tips for the Agent

- Think of yourself as a librarian and editor, not a summarizer.
- A good page about a person reads like a mini-biography — not a list of bullet points.
- Cross-reference aggressively. If two concepts are related, link them.
- When in doubt about whether something deserves its own page, create it. Stubs are fine.
- The wiki is never "done" — every answer and every source makes it richer.
- The emergent structure policy means you own the shape of the wiki. When 3-4 related pages cluster
  together, think about whether they belong in a subdir. When a subdir grows stale or mismatched,
  restructure and log it. Keep the structure serving the content, not the other way around.
- The user's recordings are in EN, RU, and PL — often mixed within one recording. Strictly match the page
  language to the source audio/transcript language (Polish to Polish, Russian to Russian, etc.) to capture all the nuance.

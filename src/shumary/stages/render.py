"""Render final markdown transcript and archive audio.

All YAML emitted via yaml.safe_dump — no hand-rolled repr() escaping.
"""

import shutil
from datetime import datetime
from pathlib import Path

from shumary.config import LOW_CONF_THRESHOLD, HIGH_SILENCE_PROB
from shumary.models import AlignedSegment, SpeakerStats, MoodSignals, TranscriptionResult
from shumary.paths import RAW_AUDIO, archived_audio_path_for, relative_to_repo
from shumary.utils import fmt_time, fmt_duration, slugify


def save_transcript(
    audio: Path,
    transcription: TranscriptionResult,
    segments: list[AlignedSegment],
    mood: MoodSignals,
    speaker_stats: SpeakerStats | None,
    date_str: str,
    slug: str,
    duration_sec: float | None,
    out_path: Path,
) -> Path:
    content = _build_document(
        audio=audio,
        transcription=transcription,
        segments=segments,
        mood=mood,
        speaker_stats=speaker_stats,
        date_str=date_str,
        slug=slug,
        duration_sec=duration_sec,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def archive_audio(audio: Path) -> Path:
    dest = archived_audio_path_for(audio)
    RAW_AUDIO.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        ts = datetime.now().strftime("%H%M%S")
        dest = dest.with_stem(f"{dest.stem}-{ts}")
    shutil.move(str(audio), dest)
    return dest


# ── Document builder ───────────────────────────────────────────────────────────

def _build_document(
    audio: Path,
    transcription: TranscriptionResult,
    segments: list[AlignedSegment],
    mood: MoodSignals,
    speaker_stats: SpeakerStats | None,
    date_str: str,
    slug: str,
    duration_sec: float | None,
) -> str:
    archived_path = archived_audio_path_for(audio)
    fm = _build_frontmatter(
        audio=audio,
        archived_path=archived_path,
        transcription=transcription,
        mood=mood,
        speaker_stats=speaker_stats,
        date_str=date_str,
        duration_sec=duration_sec,
        slug=slug,
    )

    parts = [f"---\n{fm}---\n"]
    parts.append(_render_header(slug, date_str, transcription, duration_sec, archived_path))

    if speaker_stats:
        parts.append(_render_speaker_section(speaker_stats))

    parts.append(_render_mood_section(mood))
    parts.append(_render_transcript_body(segments))
    parts.append(_render_plain_text(transcription.text))

    return "\n".join(parts)


def _build_frontmatter(
    audio: Path,
    archived_path: Path,
    transcription: TranscriptionResult,
    mood: MoodSignals,
    speaker_stats: SpeakerStats | None,
    date_str: str,
    duration_sec: float | None,
    slug: str,
) -> str:
    if speaker_stats:
        speakers_block = {
            "count": speaker_stats.count,
            "labels": [s.label for s in speaker_stats.speakers],
            "breakdown": [
                {"label": s.label, "duration": fmt_duration(s.duration_sec), "pct": s.pct}
                for s in speaker_stats.speakers
            ],
        }
    else:
        speakers_block = {
            "count": None,
            "labels": None,
            "breakdown": None,
            "note": "diarization skipped — no HF_TOKEN or model unavailable",
        }

    mood_block = {
        "avg_speaking_rate_wps": mood.avg_speaking_rate_wps,
        "rate_variability": mood.rate_variability,
        "long_pauses": mood.long_pauses,
        "long_pause_details": [
            {"at": p.at, "duration_sec": p.duration_sec} for p in mood.long_pause_details
        ],
        "filler_word_count": mood.filler_word_count,
        "top_fillers": list(mood.top_fillers),
        "repeated_phrases": list(mood.repeated_phrases),
        "low_confidence_segments": mood.low_confidence_segments,
        "near_silence_segments": mood.near_silence_segments,
        "inferred_state": mood.inferred_state,
    }

    data = {
        "title": slug,
        "type": "transcript",
        "source_file": f"Inbox/{audio.name}",
        "archived_to": str(relative_to_repo(archived_path)),
        "date": date_str,
        "duration": fmt_duration(duration_sec),
        "language": transcription.language,
        "words": transcription.word_count,
        "status": "pending-ingest",
        "ingested_date": None,
        "speakers": speakers_block,
        "mood_signals": mood_block,
    }

    import yaml
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)


# ── Sections ───────────────────────────────────────────────────────────────────

def _render_header(
    slug: str,
    date_str: str,
    transcription: TranscriptionResult,
    duration_sec: float | None,
    archived_path: Path,
) -> str:
    return (
        f"# Transcript: {slug}\n\n"
        f"> **Date**: {date_str}  \n"
        f"> **Duration**: {fmt_duration(duration_sec)}  \n"
        f"> **Language**: {transcription.language}  \n"
        f"> **Words**: {transcription.word_count}  \n"
        f"> **Original file**: `{archived_path.name}`\n"
    )


def _render_speaker_section(stats: SpeakerStats) -> str:
    lines = [
        "---\n",
        "## Speakers\n",
        f"**{stats.count} speaker(s) detected** by pyannote.audio\n",
        "| Label | Time | Share |",
        "|-------|------|-------|",
    ]
    for s in stats.speakers:
        lines.append(f"| `{s.label}` | {fmt_duration(s.duration_sec)} | {s.pct}% |")
    lines.append(
        "\n> **Label these speakers during wiki ingest**: "
        "tell the agent who SPEAKER_00, SPEAKER_01, etc. are.\n"
    )
    return "\n".join(lines) + "\n"


def _render_mood_section(mood: MoodSignals) -> str:
    lines = [
        "---\n",
        "## Behavioral / Mood Signals\n",
        "> Extracted from Whisper segment metadata.  \n"
        "> All inferences are **(uncertain)** — hypotheses, not diagnoses.\n",
        "| Signal | Value |",
        "|--------|-------|",
        f"| Speaking rate | {mood.avg_speaking_rate_wps} wps (baseline ~2.5) |",
        f"| Rate variability | {mood.rate_variability} |",
        f"| Long pauses (≥{3.0}s) | {mood.long_pauses} |",
        f"| Filler words | {mood.filler_word_count} |",
        f"| Low-confidence segments | {mood.low_confidence_segments} |",
        f"| Near-silence segments | {mood.near_silence_segments} |",
    ]
    if mood.top_fillers:
        lines.append(f"| Top fillers | {', '.join(mood.top_fillers)} |")

    lines.append(f"\n**Inferred state**: {mood.inferred_state}\n")

    if mood.long_pause_details:
        lines.append("**Long pauses at**:")
        for p in mood.long_pause_details:
            lines.append(f"- `[{p.at}]` — {p.duration_sec}s")
        lines.append("")

    if mood.repeated_phrases:
        lines.append("**Repeated phrases**:")
        for phrase in mood.repeated_phrases:
            lines.append(f'- "{phrase}"')
        lines.append("")

    return "\n".join(lines) + "\n"


def _render_transcript_body(segments: list[AlignedSegment]) -> str:
    lines = ["---\n", "## Full Transcript\n"]
    current_speaker: str | None = None

    for seg in segments:
        speaker = seg.speaker or "SPEAKER_??"
        if speaker != current_speaker:
            lines.append(f"\n**{speaker}**")
            current_speaker = speaker

        annotations: list[str] = []
        if seg.avg_logprob <= LOW_CONF_THRESHOLD:
            annotations.append("⚠ low-confidence")
        if seg.no_speech_prob >= HIGH_SILENCE_PROB:
            annotations.append("🔇 near-silence")

        # Check if there's a pause before this segment
        annotation_str = "  `" + " · ".join(annotations) + "`" if annotations else ""
        lines.append(f"[{fmt_time(seg.start)}] {seg.text.strip()}{annotation_str}")

    return "\n".join(lines) + "\n"


def _render_plain_text(text: str) -> str:
    return f"---\n\n## Plain Text\n\n{text.strip()}\n"

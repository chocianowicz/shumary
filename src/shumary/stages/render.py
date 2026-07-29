"""Render the final markdown transcript.

All YAML emitted via yaml.safe_dump — no hand-rolled repr() escaping.
"""

import re
from pathlib import Path

from shumary.config import LOW_CONF_THRESHOLD, HIGH_SILENCE_PROB, LONG_PAUSE_SEC, LATENCY_CAP_SEC
from shumary.models import (
    AlignedSegment,
    InteractionMetrics,
    ProsodyResult,
    SpeakerStats,
    MoodSignals,
    TranscriptionResult,
)
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
    diarize_backend: str | None = None,
    interaction: InteractionMetrics | None = None,
    prosody: ProsodyResult | None = None,
    plain_text: bool = True,
) -> Path:
    content = _build_document(
        audio=audio,
        transcription=transcription,
        segments=segments,
        mood=mood,
        speaker_stats=speaker_stats,
        diarize_backend=diarize_backend,
        interaction=interaction,
        prosody=prosody,
        date_str=date_str,
        slug=slug,
        duration_sec=duration_sec,
        plain_text=plain_text,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    return out_path


def save_transcript_json(
    audio: Path,
    transcription: TranscriptionResult,
    segments: list[AlignedSegment],
    mood: MoodSignals,
    speaker_stats: SpeakerStats | None,
    date_str: str,
    slug: str,
    duration_sec: float | None,
    out_path: Path,
    diarize_backend: str | None = None,
    interaction: InteractionMetrics | None = None,
    prosody: ProsodyResult | None = None,
) -> Path:
    """Write the machine-readable twin of the Markdown transcript.

    Same analysis, structured rather than formatted: the Markdown is for reading
    five minutes after a meeting, this is for everything built on top of it.
    """
    import json

    payload = _frontmatter_data(
        audio=audio,
        transcription=transcription,
        mood=mood,
        speaker_stats=speaker_stats,
        date_str=date_str,
        duration_sec=duration_sec,
        slug=slug,
        diarize_backend=diarize_backend,
        interaction=interaction,
        prosody=prosody,
    )
    payload["segments"] = [
        {
            "start": seg.start,
            "end": seg.end,
            "speaker": seg.speaker,
            "text": seg.text.strip(),
            "avg_logprob": seg.avg_logprob,
            "no_speech_prob": seg.no_speech_prob,
        }
        for seg in segments
    ]
    payload["text"] = transcription.text

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_path


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
    diarize_backend: str | None = None,
    interaction: InteractionMetrics | None = None,
    prosody: ProsodyResult | None = None,
    plain_text: bool = True,
) -> str:
    fm = _build_frontmatter(
        audio=audio,
        transcription=transcription,
        mood=mood,
        speaker_stats=speaker_stats,
        date_str=date_str,
        duration_sec=duration_sec,
        slug=slug,
        diarize_backend=diarize_backend,
        interaction=interaction,
        prosody=prosody,
    )

    parts = [f"---\n{fm}---\n"]
    parts.append(_render_header(slug, date_str, transcription, duration_sec, audio))

    if speaker_stats:
        parts.append(_render_speaker_section(speaker_stats, diarize_backend))

    parts.append(_render_mood_section(mood))

    if interaction is not None:
        parts.append(_render_interaction_section(interaction))

    if prosody is not None and prosody.speakers:
        parts.append(_render_prosody_section(prosody))

    parts.append(_render_transcript_body(segments))

    if plain_text:
        parts.append(_render_plain_text(transcription.text))

    return "\n".join(parts)


def _frontmatter_data(
    audio: Path,
    transcription: TranscriptionResult,
    mood: MoodSignals,
    speaker_stats: SpeakerStats | None,
    date_str: str,
    duration_sec: float | None,
    slug: str,
    diarize_backend: str | None = None,
    interaction: InteractionMetrics | None = None,
    prosody: ProsodyResult | None = None,
) -> dict:
    """The structured summary shared by the YAML frontmatter and the JSON transcript."""
    if speaker_stats:
        speakers_block = {
            "count": speaker_stats.count,
            "labels": [s.label for s in speaker_stats.speakers],
            "backend": diarize_backend or "unknown",
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
        "per_speaker": [_speaker_mood_block(sm) for sm in mood.per_speaker],
    }

    data = {
        "title": slug,
        "source_file": str(audio),
        "date": date_str,
        "duration": fmt_duration(duration_sec),
        "language": transcription.language,
        "words": transcription.word_count,
        "speakers": speakers_block,
        "mood_signals": mood_block,
    }

    if interaction is not None:
        data["interaction"] = _interaction_block(interaction)

    if prosody is not None and prosody.speakers:
        data["prosody"] = [_speaker_prosody_block(sp) for sp in prosody.speakers]

    return data


def _build_frontmatter(**kwargs) -> str:
    import yaml
    return yaml.safe_dump(
        _frontmatter_data(**kwargs),
        allow_unicode=True, sort_keys=False, default_flow_style=False,
    )


def _speaker_prosody_block(sp) -> dict:
    return {
        "speaker": sp.speaker,
        "f0_baseline_hz": sp.f0_baseline_hz,
        "f0_variability_st": sp.f0_variability_st,
        "energy_mean_db": sp.energy_mean_db,
        "energy_dynamics_db": sp.energy_dynamics_db,
        "jitter_local": sp.jitter_local,
        "shimmer_local": sp.shimmer_local,
        "f0_trend_by_third_st": list(sp.f0_trend_by_third_st),
    }


def _speaker_mood_block(sm) -> dict:
    return {
        "speaker": sm.speaker,
        "speaking_rate_wps": sm.speaking_rate_wps,
        "rate_variability": sm.rate_variability,
        "intra_pauses": sm.intra_pauses,
        "intra_pause_details": [
            {"at": p.at, "duration_sec": p.duration_sec} for p in sm.intra_pause_details
        ],
        "hesitations": sm.hesitations,
        "discourse_markers": sm.discourse_markers,
        "top_fillers": list(sm.top_fillers),
        "repeated_phrases": list(sm.repeated_phrases),
        "low_confidence_segments": sm.low_confidence_segments,
        "inferred_state": sm.inferred_state,
    }


def _interaction_block(interaction: InteractionMetrics) -> dict:
    return {
        "backend": interaction.backend,
        "overlap_count": interaction.overlap_count,
        "overlap_total_sec": interaction.overlap_total_sec,
        "long_gaps": interaction.long_gaps,
        "speakers": [
            {
                "speaker": s.speaker,
                "turn_count": s.turn_count,
                "avg_turn_sec": s.avg_turn_sec,
                "median_response_latency_sec": s.median_response_latency_sec,
                "p90_response_latency_sec": s.p90_response_latency_sec,
                "interruptions_made": s.interruptions_made,
                "interruptions_received": s.interruptions_received,
                "talk_share_by_third": list(s.talk_share_by_third),
            }
            for s in interaction.speakers
        ],
    }


# ── Sections ───────────────────────────────────────────────────────────────────

def _render_header(
    slug: str,
    date_str: str,
    transcription: TranscriptionResult,
    duration_sec: float | None,
    audio: Path,
) -> str:
    lines = (
        f"# Transcript: {slug}\n\n"
        f"> **Date**: {date_str}  \n"
        f"> **Duration**: {fmt_duration(duration_sec)}  \n"
        f"> **Language**: {transcription.language}  \n"
        f"> **Words**: {transcription.word_count}  \n"
        f"> **Original file**: `{audio.name}`\n"
    )
    if transcription.hallucinations_filtered > 0:
        n = transcription.hallucinations_filtered
        plural = "s" if n != 1 else ""
        lines += (
            f"> **Quality notes**: {n} hallucination-suspect segment{plural} "
            f"filtered during transcription\n"
        )
    return lines


_BACKEND_LABELS = {
    "pyannote": "pyannote.audio",
    "resemblyzer": "resemblyzer",
    "unknown": "an unknown backend",
}


def _render_speaker_section(stats: SpeakerStats, diarize_backend: str | None = None) -> str:
    backend_label = _BACKEND_LABELS.get(diarize_backend or "unknown", diarize_backend or "an unknown backend")
    lines = [
        "---\n",
        "## Speakers\n",
        f"**{stats.count} speaker(s) detected** by {backend_label}\n",
        "| Label | Time | Share |",
        "|-------|------|-------|",
    ]
    for s in stats.speakers:
        lines.append(f"| `{s.label}` | {fmt_duration(s.duration_sec)} | {s.pct}% |")
    lines.append(
        "\n> Speaker labels are assigned automatically — map SPEAKER_00, "
        "SPEAKER_01, … to real names as needed.\n"
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
        f"| Long pauses (≥{LONG_PAUSE_SEC}s) | {mood.long_pauses} |",
        f"| Filler words (hesitations + discourse markers) | {mood.filler_word_count} |",
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

    if mood.per_speaker:
        lines.append(_render_per_speaker_mood_table(mood.per_speaker))

    return "\n".join(lines) + "\n"


def _render_per_speaker_mood_table(per_speaker: tuple) -> str:
    """One column per speaker — hesitations and discourse markers shown
    separately (B6)."""
    lines = ["### Per-Speaker Signals\n"]

    header = "| Signal | " + " | ".join(f"`{sm.speaker}`" for sm in per_speaker) + " |"
    sep = "|--------|" + "|".join("------" for _ in per_speaker) + "|"
    lines += [header, sep]

    lines.append(
        "| Speaking rate (wps) | "
        + " | ".join(f"{sm.speaking_rate_wps}" for sm in per_speaker) + " |"
    )
    lines.append(
        "| Rate variability | "
        + " | ".join(f"{sm.rate_variability}" for sm in per_speaker) + " |"
    )
    lines.append(
        f"| Intra-speaker pauses (≥{LONG_PAUSE_SEC}s) | "
        + " | ".join(f"{sm.intra_pauses}" for sm in per_speaker) + " |"
    )
    lines.append(
        "| Hesitations (uh, um, ...) | "
        + " | ".join(f"{sm.hesitations}" for sm in per_speaker) + " |"
    )
    lines.append(
        "| Discourse markers (like, ну, ...) | "
        + " | ".join(f"{sm.discourse_markers}" for sm in per_speaker) + " |"
    )
    lines.append(
        "| Low-confidence segments | "
        + " | ".join(f"{sm.low_confidence_segments}" for sm in per_speaker) + " |"
    )
    if any(sm.top_fillers for sm in per_speaker):
        lines.append(
            "| Top fillers | "
            + " | ".join(", ".join(sm.top_fillers) or "—" for sm in per_speaker) + " |"
        )

    lines.append("")
    for sm in per_speaker:
        lines.append(f"**{sm.speaker} inferred state**: {sm.inferred_state}\n")

    return "\n".join(lines)


def _render_interaction_section(interaction: InteractionMetrics) -> str:
    lines = [
        "---\n",
        "## Interaction Dynamics\n",
        "> Turn-taking metrics derived from speaker diarization.  \n"
        "> All inferences are **(uncertain)** — hypotheses, not diagnoses.\n",
    ]

    speakers = interaction.speakers
    header = "| Metric | " + " | ".join(f"`{s.speaker}`" for s in speakers) + " |"
    sep = "|--------|" + "|".join("------" for _ in speakers) + "|"
    lines += [header, sep]

    lines.append(
        "| Turn count | " + " | ".join(str(s.turn_count) for s in speakers) + " |"
    )
    lines.append(
        "| Avg turn length (s) | " + " | ".join(f"{s.avg_turn_sec}" for s in speakers) + " |"
    )
    lines.append(
        "| Median response latency (s) | "
        + " | ".join(_fmt_optional(s.median_response_latency_sec) for s in speakers) + " |"
    )
    lines.append(
        "| p90 response latency (s) | "
        + " | ".join(_fmt_optional(s.p90_response_latency_sec) for s in speakers) + " |"
    )
    lines.append(
        "| Interruptions made | "
        + " | ".join(str(s.interruptions_made) for s in speakers) + " |"
    )
    lines.append(
        "| Interruptions received | "
        + " | ".join(str(s.interruptions_received) for s in speakers) + " |"
    )
    lines.append(
        "| Talk share by third | "
        + " | ".join(
            "/".join(f"{x:.0%}" for x in s.talk_share_by_third) for s in speakers
        ) + " |"
    )

    lines.append(f"\n**Long gaps (> {LATENCY_CAP_SEC}s, excluded from latency stats)**: {interaction.long_gaps}\n")

    if interaction.overlap_count is not None:
        lines.append(
            f"**Overlapping speech**: {interaction.overlap_count} overlap(s), "
            f"{interaction.overlap_total_sec}s total\n"
        )
    else:
        lines.append(
            "**Overlapping speech**: not available (resemblyzer turns cannot overlap)\n"
        )

    lines.append(
        "\n> **(uncertain)** Frequent interruptions or consistently high response "
        "latency for one speaker *may* indicate a power asymmetry, distraction, "
        "or processing time — not necessarily disengagement. Talk-share trends "
        "across thirds *may* reflect a shift in who is leading the conversation.\n"
    )

    return "\n".join(lines) + "\n"


def _render_prosody_section(prosody: ProsodyResult) -> str:
    speakers = prosody.speakers
    lines = [
        "---\n",
        "## Voice / Prosody\n",
        "> Acoustic features extracted from the audio (pitch, energy, voice "
        "quality) — independent of the words spoken.  \n"
        "> These are **descriptive acoustics, not emotional diagnoses**. All "
        "inferences are **(uncertain)**.\n",
    ]

    header = "| Signal | " + " | ".join(f"`{sp.speaker}`" for sp in speakers) + " |"
    sep = "|--------|" + "|".join("------" for _ in speakers) + "|"
    lines += [header, sep]

    lines.append(
        "| F0 baseline (Hz) | "
        + " | ".join(f"{sp.f0_baseline_hz}" for sp in speakers) + " |"
    )
    lines.append(
        "| F0 variability (semitones) | "
        + " | ".join(f"{sp.f0_variability_st}" for sp in speakers) + " |"
    )
    lines.append(
        "| Energy mean (dB) | "
        + " | ".join(f"{sp.energy_mean_db}" for sp in speakers) + " |"
    )
    lines.append(
        "| Energy dynamics (dB) | "
        + " | ".join(f"{sp.energy_dynamics_db}" for sp in speakers) + " |"
    )
    lines.append(
        "| Jitter (local) | "
        + " | ".join(_fmt_optional(sp.jitter_local) for sp in speakers) + " |"
    )
    lines.append(
        "| Shimmer (local) | "
        + " | ".join(_fmt_optional(sp.shimmer_local) for sp in speakers) + " |"
    )
    lines.append(
        "| F0 trend by third (semitones) | "
        + " | ".join(
            "/".join(f"{x:+.1f}" for x in sp.f0_trend_by_third_st) for sp in speakers
        ) + " |"
    )

    lines.append(
        "\n> **(uncertain)** A flattening F0 trend across thirds *may* reflect "
        "fatigue, deflation, or simply settling into a topic. Reduced F0/energy "
        "variability *may* indicate monotone or subdued delivery — or just a "
        "calmer register. Elevated jitter/shimmer *may* accompany vocal tension "
        "or strain, but are also strongly affected by microphone, distance, and "
        "recording quality — treat all of these as hypotheses, not diagnoses.\n"
    )

    return "\n".join(lines) + "\n"


def _fmt_optional(value: float | None) -> str:
    return f"{value}" if value is not None else "—"


# Paragraphs are broken at a sentence boundary once they reach roughly this
# many words.
PARAGRAPH_MAX_WORDS = 80

_SENTENCE_END_RE = re.compile(r'[.!?…][”"\')\]]?$')


def _segment_annotation(seg: AlignedSegment) -> str:
    annotations: list[str] = []
    if seg.avg_logprob <= LOW_CONF_THRESHOLD:
        annotations.append("⚠ low-confidence")
    if seg.no_speech_prob >= HIGH_SILENCE_PROB:
        annotations.append("🔇 near-silence")
    return "  `" + " · ".join(annotations) + "`" if annotations else ""


def _ends_at_sentence_boundary(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.strip()))


def _render_transcript_body(segments: list[AlignedSegment]) -> str:
    """Render the transcript as speaker paragraphs.

    Consecutive same-speaker segments are merged into prose paragraphs, with
    a timestamp only at the start of each paragraph. A new paragraph starts
    when the speaker changes, an inline pause marker is inserted (intra-
    speaker gap >= LONG_PAUSE_SEC), or the current paragraph has reached
    ~PARAGRAPH_MAX_WORDS at a sentence boundary. Inter-speaker gaps are not
    annotated — they're response latency, covered by Interaction Dynamics.
    """
    lines = ["---\n", "## Full Transcript\n"]

    current_speaker: str | None = None
    paragraph: list[str] = []
    paragraph_words = 0
    prev_seg: AlignedSegment | None = None

    def flush_paragraph() -> None:
        if paragraph:
            lines.append(" ".join(paragraph))

    for seg in segments:
        speaker = seg.speaker or "SPEAKER_??"
        text = seg.text.strip()
        annotation_str = _segment_annotation(seg)

        speaker_changed = speaker != current_speaker
        pause_marker: str | None = None
        if not speaker_changed and prev_seg is not None:
            gap = seg.start - prev_seg.end
            if gap >= LONG_PAUSE_SEC:
                pause_marker = f"*(pause {gap:.1f}s)*"

        start_new_paragraph = speaker_changed or pause_marker is not None

        if start_new_paragraph:
            flush_paragraph()
            if speaker_changed:
                lines.append(f"\n**{speaker}**")
                current_speaker = speaker
            if pause_marker:
                lines.append(pause_marker)
            paragraph = []
            paragraph_words = 0

        piece = text + annotation_str
        if not paragraph:
            paragraph.append(f"[{fmt_time(seg.start)}] {piece}")
        else:
            paragraph.append(piece)
        paragraph_words += len(text.split())

        if paragraph_words >= PARAGRAPH_MAX_WORDS and _ends_at_sentence_boundary(text):
            flush_paragraph()
            paragraph = []
            paragraph_words = 0

        prev_seg = seg

    flush_paragraph()

    return "\n".join(lines) + "\n"


def _render_plain_text(text: str) -> str:
    return f"---\n\n## Plain Text\n\n{text.strip()}\n"

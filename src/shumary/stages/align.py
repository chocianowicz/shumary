"""Align Whisper segments with speaker turns.

Two strategies, chosen per-segment:

1. **Word-level alignment** (Spec 4.1) — when a segment carries word timestamps
   (`Segment.words`) and speaker turns are present: each word is assigned to the
   speaker turn it overlaps most; consecutive same-speaker words are grouped into
   `AlignedSegment`s with start/end taken from the actual word times. This avoids
   misattributed quotes from naive char-proportional splitting.

2. **Char-proportional fallback** — for segments with no word timestamps (older
   caches, or backends that don't provide them), the previous behaviour is kept:
   a segment spanning multiple turns is split proportionally by character count.
"""

from shumary.models import Segment, Turn, AlignedSegment, Word

MIN_OVERLAP_SEC = 0.1


def run(segments: tuple[Segment, ...], turns: list[Turn] | None) -> list[AlignedSegment]:
    if not turns:
        return [
            AlignedSegment(
                start=s.start, end=s.end, text=s.text,
                avg_logprob=s.avg_logprob, no_speech_prob=s.no_speech_prob,
                speaker=None,
            )
            for s in segments
        ]

    result: list[AlignedSegment] = []
    for seg in segments:
        if seg.words:
            pieces = _split_segment_by_words(seg, turns)
        else:
            pieces = _split_segment(seg, turns)
        result.extend(pieces)
    return result


# ── Word-level alignment (Spec 4.1) ─────────────────────────────────────────────

def _split_segment_by_words(seg: Segment, turns: list[Turn]) -> list[AlignedSegment]:
    """Assign each word to its max-overlap turn, then group consecutive
    same-speaker words into AlignedSegments."""
    speakers = [_word_speaker(w, turns) for w in seg.words]

    # Fill zero-overlap words: inherit previous word's speaker; first word
    # falls back to the turn whose midpoint is nearest.
    for i, sp in enumerate(speakers):
        if sp is not None:
            continue
        if i > 0:
            speakers[i] = speakers[i - 1]
        else:
            speakers[i] = _nearest_turn_by_midpoint(seg.words[i], turns).speaker

    # Group consecutive same-speaker words.
    pieces: list[AlignedSegment] = []
    group_words: list[Word] = []
    group_speaker: str | None = None

    for word, speaker in zip(seg.words, speakers):
        if group_speaker is None:
            group_speaker = speaker
            group_words = [word]
        elif speaker == group_speaker:
            group_words.append(word)
        else:
            pieces.append(_make_piece(group_words, group_speaker, seg))
            group_speaker = speaker
            group_words = [word]

    if group_words:
        pieces.append(_make_piece(group_words, group_speaker, seg))

    return pieces


def _make_piece(words: list[Word], speaker: str | None, seg: Segment) -> AlignedSegment:
    return AlignedSegment(
        start=words[0].start,
        end=words[-1].end,
        text=" ".join(w.text for w in words),
        avg_logprob=seg.avg_logprob,
        no_speech_prob=seg.no_speech_prob,
        speaker=speaker,
    )


def _word_speaker(word: Word, turns: list[Turn]) -> str | None:
    """Return the speaker of the turn with the greatest overlap with this word,
    or None if no turn overlaps it at all."""
    best_turn: Turn | None = None
    best_overlap = 0.0
    for t in turns:
        ov = _overlap(word.start, word.end, t.start, t.end)
        if ov > best_overlap:
            best_overlap = ov
            best_turn = t
    return best_turn.speaker if best_turn is not None else None


def _nearest_turn_by_midpoint(word: Word, turns: list[Turn]) -> Turn:
    word_mid = (word.start + word.end) / 2
    return min(turns, key=lambda t: abs(((t.start + t.end) / 2) - word_mid))


# ── Char-proportional fallback (no word timestamps) ─────────────────────────────

def _split_segment(seg: Segment, turns: list[Turn]) -> list[AlignedSegment]:
    """Split one Whisper segment into AlignedSegments at speaker boundaries."""
    overlapping = [
        t for t in turns
        if _overlap(seg.start, seg.end, t.start, t.end) >= MIN_OVERLAP_SEC
    ]

    if not overlapping:
        # No speaker turn covers this segment
        return [AlignedSegment(
            start=seg.start, end=seg.end, text=seg.text,
            avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob,
            speaker="SPEAKER_??",
        )]

    if len(overlapping) == 1:
        return [AlignedSegment(
            start=seg.start, end=seg.end, text=seg.text,
            avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob,
            speaker=overlapping[0].speaker,
        )]

    # Multiple speakers — split at boundaries within the segment's time range
    pieces: list[AlignedSegment] = []
    seg_dur = seg.end - seg.start
    text = seg.text
    total_chars = max(len(text), 1)

    for i, turn in enumerate(overlapping):
        piece_start = max(seg.start, turn.start)
        if i + 1 < len(overlapping):
            piece_end = min(seg.end, overlapping[i + 1].start)
        else:
            piece_end = seg.end

        if piece_end <= piece_start:
            continue

        # Distribute text proportionally by duration
        piece_dur = piece_end - piece_start
        frac = piece_dur / seg_dur if seg_dur > 0 else 1.0
        char_count = max(1, round(total_chars * frac))
        piece_text = text[:char_count].strip()
        text = text[char_count:]

        pieces.append(AlignedSegment(
            start=piece_start, end=piece_end, text=piece_text,
            avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob,
            speaker=turn.speaker,
        ))

    # Attach any leftover text to the last piece
    if text.strip() and pieces:
        last = pieces[-1]
        pieces[-1] = AlignedSegment(
            start=last.start, end=last.end, text=(last.text + " " + text.strip()).strip(),
            avg_logprob=last.avg_logprob, no_speech_prob=last.no_speech_prob,
            speaker=last.speaker,
        )

    return pieces if pieces else [AlignedSegment(
        start=seg.start, end=seg.end, text=seg.text,
        avg_logprob=seg.avg_logprob, no_speech_prob=seg.no_speech_prob,
        speaker=overlapping[0].speaker,
    )]


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

"""Align Whisper segments with pyannote speaker turns.

Rewrite of the old merge.py: instead of assigning a whole Whisper segment to the
dominant speaker, we split segments at speaker boundaries when a segment spans multiple
turns. Text is distributed proportionally by character count (word timestamps are not
available in Whisper's default mode).
"""

from shumary.models import Segment, Turn, AlignedSegment

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
        pieces = _split_segment(seg, turns)
        result.extend(pieces)
    return result


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

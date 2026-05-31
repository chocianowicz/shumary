"""Whisper transcription stage."""

from pathlib import Path

from shumary.models import Segment, TranscriptionResult
from shumary.modelhub import whisper_model
from shumary.config import WHISPER_MODEL


def run(audio: Path, model_name: str = WHISPER_MODEL, *, fast: bool = False) -> TranscriptionResult:
    model = whisper_model(model_name)

    if fast:
        # Greedy decode (beam_size=1) + single temperature pass.
        # ~25-35% faster than default on large model. Tiny quality trade-off on clean speech.
        result = model.transcribe(
            str(audio),
            verbose=False,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=True,
        )
    else:
        result = model.transcribe(str(audio), verbose=False)

    language = result.get("language", "unknown")
    text = result.get("text", "")
    word_count = len(text.split())

    segments = tuple(
        Segment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            avg_logprob=s.get("avg_logprob", 0.0),
            no_speech_prob=s.get("no_speech_prob", 0.0),
        )
        for s in result.get("segments", [])
    )

    return TranscriptionResult(
        language=language,
        word_count=word_count,
        text=text,
        segments=segments,
    )

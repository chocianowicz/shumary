"""Whisper transcription stage."""

from pathlib import Path

from shumary.models import Segment, TranscriptionResult
from shumary.modelhub import whisper_model
from shumary.config import WHISPER_MODEL


def run(audio: Path, model_name: str = WHISPER_MODEL) -> TranscriptionResult:
    model = whisper_model(model_name)
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

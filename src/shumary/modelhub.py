"""Singleton-loaded AI models — load once per process, reuse across all files in a batch."""

import os
import warnings
from contextlib import contextmanager
from functools import lru_cache

from shumary.config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    DIARIZATION_MODEL,
    HF_TOKEN_ENVVARS,
)


@contextmanager
def _suppress_pyannote_warnings():
    """Silence pyannote.audio's noisy import-/load-time UserWarnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
        yield


@lru_cache(maxsize=None)
def whisper_model(name: str = WHISPER_MODEL):
    try:
        import whisper  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError(
            "openai-whisper is not installed. Run: pip install openai-whisper"
        )

    device = WHISPER_DEVICE or _autodetect_device()
    print(f"  Loading Whisper '{name}' on {device}…")
    return whisper.load_model(name, device=device)


@lru_cache(maxsize=None)
def resemblyzer_encoder():
    """Return VoiceEncoder or None if resemblyzer is not installed."""
    try:
        from resemblyzer import VoiceEncoder  # type: ignore[import-untyped]
    except ImportError:
        print("  ⚠  resemblyzer not installed — skipping diarization.")
        return None
    print("  Loading resemblyzer VoiceEncoder…")
    return VoiceEncoder()


@lru_cache(maxsize=None)
def diarization_pipeline():
    """Return pyannote Pipeline or None (gracefully) if unavailable."""
    token = _hf_token()
    if not token:
        print("  ⚠  No HF_TOKEN — skipping diarization.")
        return None

    try:
        with _suppress_pyannote_warnings():
            from pyannote.audio import Pipeline  # type: ignore[import-untyped]
    except ImportError:
        print("  ⚠  pyannote.audio not installed — skipping diarization.")
        return None

    try:
        with _suppress_pyannote_warnings():
            pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
        _move_to_device(pipeline)
        return pipeline
    except Exception as exc:
        print(f"  ⚠  Could not load diarization model: {exc}")
        return None


# ── Internal helpers ────────────────────────────────────────────────────────────

def _hf_token() -> str | None:
    for var in HF_TOKEN_ENVVARS:
        token = os.environ.get(var, "").strip()
        if token:
            return token
    return None


def _autodetect_device() -> str:
    # Note: MPS is intentionally NOT used for openai-whisper. Since Phase 3,
    # word_timestamps=True is mandatory (Spec 3), and PyTorch's MPS backend
    # cannot run the DTW alignment step it requires — `.double()` on an MPS
    # tensor raises `TypeError: Cannot convert a MPS Tensor to float64 dtype`
    # (observed on torch 2.12.0 / macOS, even with
    # PYTORCH_ENABLE_MPS_FALLBACK=1). CPU is fully supported; CUDA supports
    # float64 so it's used when available.
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _move_to_device(pipeline) -> None:
    try:
        import torch
        if torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
        elif torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception:
        pass

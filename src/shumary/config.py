"""Runtime configuration — tunables and multilingual constants."""

import os

# ── Whisper ────────────────────────────────────────────────────────────────────

WHISPER_MODEL: str = os.environ.get("SHUMARY_WHISPER_MODEL", "large")
WHISPER_DEVICE: str | None = os.environ.get("SHUMARY_WHISPER_DEVICE")

# ── Pyannote ───────────────────────────────────────────────────────────────────

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN_ENVVARS = ("HF_TOKEN", "HUGGINGFACE_TOKEN")

# ── Diarization — single-speaker decision rule (Spec 4.3) ─────────────────────
# Resemblyzer path: if the silhouette score for a forced 2-way split is below
# this threshold, treat the recording as a single speaker instead of forcing
# a (spurious) k>=2 split.
#
# Tuned 2026-06-10 against real recordings (same embedding + silhouette(k=2)
# code path as the production rule):
#   - solo sample (~5.18 min, single speaker, turns >=3s cut+concatenated
#     from a real recording): sil2 = 0.1423
#   - duo sample (~58 min, two speakers): sil2 = 0.3153
#
# The default 0.10 failed the solo<threshold check (0.1423 >= 0.10), so per
# spec it was raised to the geometric mean of the two observed scores:
# sqrt(0.1423 * 0.3153) ≈ 0.2118. NOTE: at this threshold the duo recording
# (0.3153) does not clear the spec's 2x margin (>= 0.4237) — the two
# observed scores are only ~2.2x apart, not the required 4x. Geometric mean
# is the spec's prescribed adjustment for a failed margin; a wider margin
# would require either a different feature/clustering approach or more
# diverse tuning samples (out of scope for this task).

SOLO_SILHOUETTE: float = 0.2118

# ── Dual-source recordings (mic on the left channel, system audio on the right) ──
# For recordings made by `shumary record` the two channels are separate sources,
# so "who is speaking" is a channel question, not a clustering one. The mic
# channel is gated by energy and the system channel is diarized normally.

# RMS envelope framing for the mic channel's speech gate.
DUAL_FRAME_SEC: float = 0.03
DUAL_HOP_SEC: float = 0.01

# The mic must exceed its own noise floor by this much to *stay* in a turn.
DUAL_SPEECH_MARGIN_DB: float = 10.0

# ...but a turn is only admitted if it also clears this higher bar somewhere
# inside it, and holds it for DUAL_MIN_ONSET_SEC. Two thresholds rather than one
# because the release margin above is easy to clear with things that are not
# speech: on a headset in a quiet room the noise floor sits near silence, so
# breathing, a fan, or a keypress all pass +10 dB. Sustained speech does not just
# cross that line, it stays well above it — while a keystroke is a spike a few
# tens of milliseconds wide.
#
# Deliberately low enough that short backchannel ("mhm", "да", "ok") is kept:
# those are quiet but they are still voiced, and they are what makes the
# turn-taking and response-latency metrics mean anything.
DUAL_ONSET_MARGIN_DB: float = 18.0
DUAL_MIN_ONSET_SEC: float = 0.06

# ...and must also exceed the system channel by this much, which is what rejects
# speaker bleed: audio played out loud and picked up by the mic arrives
# attenuated, so it never wins against the system channel it came from.
# Headphones make this moot; this margin is what makes speakers survivable.
DUAL_ECHO_MARGIN_DB: float = 6.0

# Envelope smoothing: gaps shorter than this are bridged, and runs shorter than
# this are dropped, so a turn is not shredded by ordinary pauses between words.
DUAL_MIN_GAP_SEC: float = 0.35
DUAL_MIN_SPEECH_SEC: float = 0.25

# Percentile of the mic's RMS envelope treated as its noise floor.
DUAL_NOISE_FLOOR_PCT: float = 20.0

# Total transcribed speech a label needs before it is reported as a participant.
# Diarization occasionally emits a cluster holding a single low-confidence word;
# presenting that as an extra person corrupts the speaker count, which is the
# headline figure of the whole transcript. Real participants clear this by orders
# of magnitude — on a measured 47-minute five-person call the quietest genuine
# speaker held 3m 36s while the phantom cluster held 1.4s.
#
# Someone who truly said only a word or two is dropped from the speaker table,
# which is the intended trade: their words stay in the transcript either way.
MIN_SPEAKER_SPEECH_SEC: float = 2.0

# A system channel whose peak never clears this (relative to full scale) is
# treated as silent — nothing was playing, so there is no remote side.
DUAL_SILENT_CHANNEL_PEAK: float = 1e-4

# ── Audio ──────────────────────────────────────────────────────────────────────

SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".webm", ".aac"}
)

# ── Mood signal thresholds ─────────────────────────────────────────────────────

LONG_PAUSE_SEC: float = 3.0
LOW_CONF_THRESHOLD: float = -1.0
HIGH_SILENCE_PROB: float = 0.6
SLOW_RATE: float = 1.8
FAST_RATE: float = 3.5
BASELINE_RATE: float = 2.5

# ── Interaction metrics (Spec 5 / 5.2) ──────────────────────────────────────────
# Consecutive same-speaker turns separated by less than this are merged into
# one utterance before computing turn/latency/talk-share metrics.
UTTERANCE_MERGE_GAP: float = 1.0

# Minimum raw-turn intersection (seconds) between different speakers to count
# as an overlap (pyannote backend only).
OVERLAP_MIN: float = 0.1

# B "yields" an interruption to A if A's turn ends within this many seconds
# after B's turn starts (and B started strictly inside A's turn).
INTERRUPT_YIELD_SEC: float = 2.0

# Response latencies above this are treated as topic breaks / silence
# ("long_gaps") rather than a response, and excluded from the
# median/p90 latency stats.
LATENCY_CAP_SEC: float = 30.0

# ── Hallucination post-filter (Spec 3-VAD) ───────────────────────────────────────
# Applies to the openai and mlx backends (faster-whisper has built-in vad_filter).
# A segment is dropped when no_speech_prob >= this threshold AND its normalized
# text repeats the previous segment's AND its duration >= 2.0s.

HALLUCINATION_NO_SPEECH: float = 0.5
HALLUCINATION_MIN_DURATION_SEC: float = 2.0

# ── Prosody extraction (Spec 7 / Phase 7.1) ──────────────────────────────────────
# Segments shorter than this are skipped entirely (too short for stable pitch
# / voice-quality estimates).
PROSODY_MIN_SEGMENT_SEC: float = 1.0

# librosa.pyin speech-band parameters.
PROSODY_F0_FMIN: float = 65.0
PROSODY_F0_FMAX: float = 400.0
PROSODY_FRAME_LENGTH: int = 2048

# A segment needs at least this many voiced pyin frames for its F0 fields to
# be populated; otherwise they're None.
PROSODY_MIN_VOICED_FRAMES: int = 10

# ── Filler words (finding B6) ───────────────────────────────────────────────────
# Split into two lexicons per language:
#  - HESITATIONS: pure non-lexical fillers (uh, um, er, эм, eee, ...) — never
#    ordinary words, always indicate a pause/hesitation when spoken.
#  - DISCOURSE_MARKERS: lexical items that *can* function as fillers/hedges
#    (like, you know, ну, вот, это, no, właśnie, ...) but are also ordinary
#    words — counted separately so they don't get conflated with hesitations.
#
# Per finding B7 (mixed-language recordings), all three languages (en, ru, pl)
# are always merged regardless of the detected language — see mood.py's
# `_filler_pattern` / `_hesitation_pattern` / `_discourse_marker_pattern`.

# Thresholds for the two filler signals. They are deliberately different kinds
# of claim: hesitations carry the "cognitive load or discomfort" reading, while
# discourse markers are reported as *style* and never as distress.
#
# A merged threshold made the verdict trivially reachable in Russian and Polish,
# where the marker lexicons are ordinary conversational vocabulary — a measured
# 47-minute Russian call scored 0 hesitations and 271 markers, and was reported
# as showing discomfort purely on the strength of ordinary speech.
HESITATION_HIGH: int = 15
DISCOURSE_MARKER_HIGH: int = 40

HESITATIONS: dict[str, set[str]] = {
    "en": {
        "uh", "um", "uhh", "umm", "er", "err", "hmm", "hm", "ugh",
    },
    "ru": {
        "эм", "эхм", "ммм",
    },
    "pl": {
        "eee", "yyy", "ehh", "yhm",
    },
}

DISCOURSE_MARKERS: dict[str, set[str]] = {
    "en": {
        "like", "basically", "literally", "actually", "you know",
    },
    # Inclusion criterion: the word must *predominantly* function as a filler in
    # speech. A high-frequency function word fails it even though it can be used
    # as a filler, because the false positives swamp the real ones.
    #
    # `это` was removed for exactly that reason: it is an ordinary demonstrative
    # pronoun ("this"), and on a measured Russian call it alone accounted for 138
    # of 271 marker hits — over half the signal from one of the most common words
    # in the language.
    "ru": {
        "ну", "вот", "значит", "короче",
        "как бы", "типа", "собственно", "в общем",
    },
    "pl": {
        "no", "właśnie", "jakby",
        "znaczy", "no to",
    },
}

# Backward/continuity alias: global `filler_word_count` keeps meaning
# hesitations + discourse markers combined. Some code/tests may still refer
# to the merged lexicon by language.
FILLER_WORDS: dict[str, set[str]] = {
    lang: HESITATIONS.get(lang, set()) | DISCOURSE_MARKERS.get(lang, set())
    for lang in set(HESITATIONS) | set(DISCOURSE_MARKERS)
}

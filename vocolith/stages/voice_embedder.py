# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Stage 4b: Extract voice d-vector embeddings per speaker using resemblyzer."""
from __future__ import annotations
import logging
from pathlib import Path

import librosa
import numpy as np

from vocolith.models.transcript import DiarizedTranscript

log = logging.getLogger(__name__)

# Minimum audio seconds per SEGMENT to include in an embedding
_MIN_SEGMENT_SECONDS = 3.0
# Minimum TOTAL audio seconds across all segments to produce a reliable embedding.
# Resemblyzer needs sufficient audio to average out noise.  Embeddings from less
# than this threshold are too noisy for reliable voice matching and are discarded.
_MIN_TOTAL_EMBEDDING_SECONDS = 5.0
# Maximum seconds to use per speaker.  Accuracy plateaus at ~30s (EER ~3-5%);
# using more audio provides negligible improvement at 2× the compute cost.
_MAX_SECONDS_PER_SPEAKER = 30.0


def compute_speaker_embeddings(
    audio_path: Path,
    transcript: DiarizedTranscript,
    sample_rate: int = 16000,
) -> dict[str, np.ndarray]:
    """
    Compute a mean d-vector embedding for each diarized speaker.

    Aggregates audio segments belonging to each speaker_label, then
    encodes them via resemblyzer's VoiceEncoder.

    Args:
        audio_path:   Path to the (denoised) WAV file.
        transcript:   Diarized transcript with speaker_label on segments.
        sample_rate:  Audio sample rate (default 16000 Hz).

    Returns:
        dict mapping speaker_label -> 256-dim numpy array.
        Empty dict if resemblyzer is not installed or no labelled speakers.
    """
    try:
        from resemblyzer import VoiceEncoder
    except ImportError:
        log.warning("resemblyzer not installed — skipping voice embeddings.")
        return {}

    # Collect speaker -> list of (start_sample, end_sample)
    speaker_segments: dict[str, list[tuple[int, int]]] = {}
    for seg in transcript.segments:
        label = seg.speaker_label
        if not label:
            continue
        start_s = int(seg.start * sample_rate)
        end_s = int(seg.end * sample_rate)
        if end_s - start_s < int(_MIN_SEGMENT_SECONDS * sample_rate):
            continue
        speaker_segments.setdefault(label, []).append((start_s, end_s))

    if not speaker_segments:
        log.debug("No labelled speaker segments for voice embedding.")
        return {}

    log.info("Loading audio for voice embeddings: %s", Path(audio_path).name)
    try:
        # Load raw audio without whole-file VAD trimming.  resemblyzer's
        # preprocess_wav() runs webrtcvad over the ENTIRE recording, which
        # can trim 30-40% of meeting audio (WebEx codec fools the VAD into
        # marking speech as silence).  This shifts all sample indices so
        # speakers appearing in the second half of the meeting get empty
        # chunks and produce degenerate identical d-vectors.
        # The diarizer has already given us precise per-speaker timestamps,
        # so we slice directly and apply only volume normalisation per chunk.
        wav_raw, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    except Exception as exc:
        log.warning("Could not load audio for resemblyzer: %s", exc)
        return {}

    encoder = VoiceEncoder()
    embeddings: dict[str, np.ndarray] = {}
    max_samples = int(_MAX_SECONDS_PER_SPEAKER * sample_rate)

    for label, segs in speaker_segments.items():
        # Concatenate up to _MAX_SECONDS_PER_SPEAKER of audio for this speaker.
        # Slice from the raw (untrimmed) wav so timestamps are exact, then
        # apply per-chunk normalisation via preprocess_wav on the small slice.
        chunks = []
        total = 0
        for start_s, end_s in segs:
            chunk_raw = wav_raw[start_s:end_s]
            if len(chunk_raw) < int(_MIN_SEGMENT_SECONDS * sample_rate):
                # Slice came up short (segment near end of file) — skip
                continue
            # Normalise volume to a consistent level; no VAD trimming needed
            # since diarizer timestamps already exclude silence.
            chunk = librosa.util.normalize(chunk_raw)
            chunks.append(chunk)
            total += len(chunk)
            if total >= max_samples:
                break

        if not chunks:
            continue

        combined = np.concatenate(chunks)[:max_samples]
        total_secs = len(combined) / sample_rate

        try:
            embedding: np.ndarray = encoder.embed_utterance(combined)  # type: ignore[assignment]
            embeddings[label] = embedding
            log.debug(
                "Voice embedding for %s: %.1f s audio -> 256-dim vector (norm=%.3f)",
                label, total_secs, float(np.linalg.norm(embedding)),
            )
        except Exception as exc:
            log.warning("Failed to embed speaker %s: %s", label, exc)

    log.info(
        "Voice embeddings computed for %d/%d speaker(s)",
        len(embeddings), len(speaker_segments),
    )
    return embeddings


def embed_segments(
    audio_path: Path,
    segment_times: list[tuple[float, float]],
    sample_rate: int = 16000,
    min_duration_s: float | None = None,
) -> np.ndarray | None:
    """
    Compute a single mean d-vector from a specific list of (start_s, end_s) time ranges.

    Used after a segment split: each person gets the embedding of THEIR audio
    slices only, not the full mixed-label audio.

    Args:
        audio_path:     Path to the WAV file.
        segment_times:  List of (start_seconds, end_seconds) pairs.
        sample_rate:    Audio sample rate (default 16000 Hz).
        min_duration_s: Minimum total audio required.  None uses
                        _MIN_TOTAL_EMBEDDING_SECONDS (default quality gate).
                        Pass 0.0 to store whatever is available — appropriate
                        when this is the only audio for a person from a split.

    Returns:
        256-dim numpy array, or None if resemblyzer unavailable or too little audio.
    """
    try:
        from resemblyzer import VoiceEncoder
    except ImportError:
        log.warning("resemblyzer not installed — cannot compute split embedding.")
        return None

    try:
        wav_raw, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    except Exception as exc:
        log.warning("Could not load audio for split embedding: %s", exc)
        return None

    max_samples = int(_MAX_SECONDS_PER_SPEAKER * sample_rate)
    chunks = []
    total = 0
    for start_s, end_s in segment_times:
        s = int(start_s * sample_rate)
        e = int(end_s * sample_rate)
        if e - s < int(_MIN_SEGMENT_SECONDS * sample_rate):
            continue
        chunk_raw = wav_raw[s:e]
        if len(chunk_raw) < int(_MIN_SEGMENT_SECONDS * sample_rate):
            continue
        chunk = librosa.util.normalize(chunk_raw)
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_samples:
            break

    if not chunks:
        log.debug("embed_segments: no usable audio in provided segment times.")
        return None

    combined = np.concatenate(chunks)[:max_samples]
    total_secs = len(combined) / sample_rate

    # Require minimum total audio.  Caller can pass min_duration_s=0.0 to
    # store whatever is available (e.g. when this is the only audio for a
    # person identified in a split — noisy embedding beats no embedding).
    effective_min = _MIN_TOTAL_EMBEDDING_SECONDS if min_duration_s is None else min_duration_s
    if effective_min > 0 and total_secs < effective_min:
        log.info(
            "embed_segments: only %.1fs of audio — below %.1fs minimum for reliable "
            "embedding; skipping (transcript assignment still applied).",
            total_secs, effective_min,
        )
        return None

    try:
        encoder = VoiceEncoder()
        embedding: np.ndarray = encoder.embed_utterance(combined)  # type: ignore[assignment]
        log.debug(
            "Split embedding: %.1fs audio → 256-dim (norm=%.3f)",
            total_secs, float(np.linalg.norm(embedding)),
        )
        return embedding
    except Exception as exc:
        log.warning("Failed to compute split embedding: %s", exc)
        return None

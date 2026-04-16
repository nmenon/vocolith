# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Stage 4: Speaker diarization via WhisperX + pyannote."""
from __future__ import annotations
import logging
from pathlib import Path

from vocolith.models.transcript import DiarizedTranscript, TranscriptSegment
from vocolith.utils.gpu import get_device

log = logging.getLogger(__name__)

# Guard: pyannote warning filters applied at most once per process.
_pyannote_warnings_suppressed = False

# pyannote/speaker-diarization-community-1 is the open-access model (no license gate).
# pyannote/speaker-diarization-3.1 requires accepting terms on HuggingFace.
_DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"


def _is_model_cached(model_id: str) -> bool:
    """Return True if the model is present in the local HuggingFace cache."""
    from pathlib import Path
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    # HF stores models as models--<org>--<name>
    slug = "models--" + model_id.replace("/", "--")
    model_dir = cache_root / slug
    # A downloaded model has a 'snapshots' subdirectory with at least one entry
    snapshots = model_dir / "snapshots"
    if snapshots.is_dir() and any(snapshots.iterdir()):
        return True
    # Also check HF_HOME / HUGGINGFACE_HUB_CACHE env overrides
    import os
    for env_var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        alt = os.environ.get(env_var)
        if alt:
            alt_dir = Path(alt) / "hub" / slug / "snapshots"
            if alt_dir.is_dir() and any(alt_dir.iterdir()):
                return True
    return False


def diarize(
    audio_path: Path,
    transcript: DiarizedTranscript,
    hf_token: str | None = None,
    min_speakers: int = 1,
    max_speakers: int = 10,
    suppress_warnings: bool = True,
) -> DiarizedTranscript:
    """
    Run speaker diarization and assign speaker labels to transcript segments.

    Uses WhisperX's DiarizationPipeline (backed by pyannote).
    Requires a HuggingFace token for model download.

    Gracefully degrades (returns transcript unchanged) if the token is missing
    or diarization fails.

    Args:
        audio_path:   Path to WAV file.
        transcript:   Transcript from transcriber (no speaker labels yet).
        hf_token:     HuggingFace API token.
        min_speakers: Minimum expected speakers (hint, not enforced).
        max_speakers: Maximum expected speakers (hint, not enforced).

    Returns:
        DiarizedTranscript with speaker_label set on each segment.
    """
    # If no token supplied, check whether the model is already in the local cache.
    # pyannote ≥ 3.x accepts token=None for locally-cached models.
    effective_token = hf_token
    if not effective_token:
        if _is_model_cached(_DEFAULT_MODEL):
            log.info(
                "No HUGGINGFACE_TOKEN but model is cached locally — "
                "proceeding with local_files_only."
            )
            effective_token = None   # will pass local_files_only=True below
        else:
            log.warning(
                "No HUGGINGFACE_TOKEN and model not cached — diarization skipped. "
                "Set HUGGINGFACE_TOKEN and accept the licence at: "
                "https://huggingface.co/pyannote/speaker-diarization-community-1"
            )
            return transcript

    try:
        from whisperx.diarize import DiarizationPipeline, assign_word_speakers
    except ImportError as exc:
        log.warning("WhisperX diarization not available: %s", exc)
        return transcript

    device = get_device()
    log.info("Running speaker diarization on %s (device=%s)...", audio_path, device)

    global _pyannote_warnings_suppressed
    if suppress_warnings and not _pyannote_warnings_suppressed:
        import warnings
        warnings.filterwarnings(
            "ignore",
            message="std\\(\\).*degrees of freedom",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="TensorFloat-32",
        )
        _pyannote_warnings_suppressed = True

    try:
        diarize_kwargs: dict = {"device": device}
        if effective_token:
            diarize_kwargs["token"] = effective_token
        else:
            # Model is cached — token=None is sufficient; pyannote loads from
            # HF cache automatically.  local_files_only is not a supported
            # DiarizationPipeline kwarg so must NOT be passed here.
            diarize_kwargs["token"] = None
        diarize_model = DiarizationPipeline(**diarize_kwargs)
        from vocolith.utils.progress import add_task, update_task, complete_task
        diag_task = add_task("Diarizing speakers…", total=100)

        def _progress_cb(pct: float) -> None:
            update_task(diag_task, completed=pct)

        diarize_df = None
        try:
            kwargs: dict = {
                "progress_callback": _progress_cb,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }

            diarize_df = diarize_model(str(audio_path), **kwargs)
        finally:
            complete_task(diag_task)
            # Always free VRAM regardless of success/failure
            del diarize_model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    log.debug("GPU cache cleared after diarization.")
            except Exception:
                pass

        # Build the dict structure assign_word_speakers expects
        result_dict = {
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "text": s.text,
                    "words": [
                        {"word": w.word, "start": w.start,
                         "end": w.end, "score": w.score}
                        for w in s.words
                    ],
                }
                for s in transcript.segments
            ]
        }

        # Guard: diarize_df is None if diarization raised an exception above.
        # Log explicitly before the outer except swallows it.
        if diarize_df is None:
            log.warning("Diarization produced no output — continuing without speaker labels.")
            return transcript

        # diarize_model() without return_embeddings always returns a plain DataFrame;
        # pyannote type stubs show a union — extract DataFrame from tuple just in case.
        import pandas as _pd
        diarize_df_clean = diarize_df if isinstance(diarize_df, _pd.DataFrame) else diarize_df[0]
        assigned = assign_word_speakers(diarize_df_clean, result_dict)  # type: ignore[arg-type]
        assigned_segs = assigned.get("segments", [])

        # Collect unique speaker labels
        speaker_labels: set[str] = set()
        labeled_segments: list[TranscriptSegment] = []

        for i, seg in enumerate(assigned_segs):
            label = seg.get("speaker")
            if label:
                speaker_labels.add(label)

            orig = transcript.segments[i] if i < len(transcript.segments) else None
            if orig is None:
                log.debug(
                    "Diarization produced extra segment at index %d "
                    "(no original to match) — word timestamps unavailable for this segment.",
                    i,
                )
            labeled_segments.append(TranscriptSegment(
                segment_id=i,
                start=seg.get("start", orig.start if orig else 0.0),
                end=seg.get("end", orig.end if orig else 0.0),
                text=seg.get("text", orig.text if orig else "").strip(),
                words=orig.words if orig else [],
                speaker_label=label,
            ))

        n_speakers = len(speaker_labels)
        log.info("Diarization complete: %d speaker(s) detected", n_speakers)
        try:
            from vocolith.utils.progress import status as _status
            _status(f"[green]✓ Diarization[/green]  {n_speakers} speaker(s) detected")
        except Exception:
            pass

        result = DiarizedTranscript(
            segments=labeled_segments,
            language=transcript.language,
            duration_seconds=transcript.duration_seconds,
            speakers_detected=n_speakers,
        )

        return result

    except Exception as exc:
        log.warning("Diarization failed (%s). Continuing without speaker labels.", exc)
        return transcript

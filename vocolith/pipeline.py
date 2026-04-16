# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Pipeline orchestrator: wires stages together via PipelineContext."""
from __future__ import annotations
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from vocolith.config import AppConfig
from vocolith.models.transcript import DiarizedTranscript

log = logging.getLogger(__name__)


@dataclass
class StageError:
    stage: str
    message: str
    fatal: bool = False


@dataclass
class PipelineContext:
    """Shared state threaded through all pipeline stages."""
    video_path: Path
    output_dir: Path   # transcript.md, meeting_notes.md written here
    debug_dir: Path    # intermediate files: WAVs, sampled frames, diarization.json
    config: AppConfig

    # Populated by stages
    audio_path: Path | None = None
    denoised_path: Path | None = None
    av_offset: float | None = None
    transcript: DiarizedTranscript | None = None

    # Video analysis results
    frames: list[Any] = field(default_factory=list)            # list[VideoFrame]
    ocr_names: list[str] = field(default_factory=list)
    ocr_vocabulary: list[str] = field(default_factory=list)
    # Per-frame name map: {timestamp_s: [names visible at that timestamp]}
    # Used for temporal OCR correlation in speaker_resolver
    frame_name_map: dict[float, list[str]] = field(default_factory=dict)
    face_detections: list[Any] = field(default_factory=list)   # list[FaceDetection]

    # Known attendees supplied by the user (--attendees flag)
    attendees: list[str] = field(default_factory=list)

    # Speaker resolution
    speaker_map: dict[str, str] = field(default_factory=dict)   # label -> display_name
    # Populated by speaker_resolver; consumed by confirmation wizard
    pending_speakers: list[Any] = field(default_factory=list)   # list[PendingSpeaker]
    # Voice embeddings pending user confirmation: label -> 256-dim numpy array.
    # resolve_speakers populates this; wizard stores under confirmed name.
    pending_voice_embeddings: dict[str, Any] = field(default_factory=dict)
    # Labels explicitly confirmed as a SINGLE person by the user.
    # Splits and skips are excluded — their embeddings are mixed or unverified.
    # Only labels in this set get an embedding written to the database.
    confirmed_single_labels: set[str] = field(default_factory=set)
    # Per-person embeddings extracted from segment splits.
    # Each entry is (person_name, 256-dim numpy array) — one entry per
    # unique person identified during a segment-by-segment split.
    confirmed_split_embeddings: list[Any] = field(default_factory=list)
    # Labels for which resolve_speakers created a brand-new profile (strategies 3/4/6).
    # The profile is saved to SQLite immediately (FK constraint), but touch() is
    # deferred to _store_confirmed_embeddings so meeting_count is not incremented
    # twice (once pre-wizard, once post-wizard).
    pending_new_profiles: set[str] = field(default_factory=set)

    # Output
    meeting_notes: Any = None   # MeetingNotes
    errors: list[StageError] = field(default_factory=list)
    _errors_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # Metadata
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    def add_error(self, stage: str, msg: str, fatal: bool = False) -> None:
        """Thread-safe error accumulation (audio + video stages run concurrently)."""
        error = StageError(stage=stage, message=msg, fatal=fatal)
        with self._errors_lock:
            self.errors.append(error)
        level = logging.ERROR if fatal else logging.WARNING
        log.log(level, "[%s] %s", stage, msg)

    @property
    def effective_audio(self) -> Path | None:
        """Return denoised audio if available, else raw audio."""
        return self.denoised_path or self.audio_path


def run_pipeline(
    video_path: Path,
    output_dir: Path,
    config: AppConfig,
    debug_dir: Path | None = None,
    attendees: list[str] | None = None,
    dry_run: bool = False,
    template: str | None = None,
    no_faces: bool = False,
    no_ocr: bool = False,
) -> PipelineContext:
    """
    Execute the full meeting decoder pipeline.

    Stages 3+4 (audio) run first, then stages 5+6+7 (video) optionally in
    parallel, then speaker resolution and note generation.

    Args:
        video_path:  Input video file.
        output_dir:  Where to write transcript.md, meeting_notes.md etc.
        config:      Loaded AppConfig.
        dry_run:     If True, skip LLM summarization (transcription only).
        template:    Notes template name or path override.
        no_faces:    Skip face recognition.
        no_ocr:      Skip OCR name extraction.

    Returns:
        PipelineContext with all populated fields and any non-fatal errors.
    """
    from vocolith.stages.audio_extractor import extract_audio, AudioExtractionError
    from vocolith.stages.audio_denoiser import denoise_audio
    from vocolith.stages.transcriber import transcribe
    from vocolith.stages.diarizer import diarize

    resolved_output = Path(output_dir)
    resolved_debug  = Path(debug_dir) if debug_dir else resolved_output / "debug"

    ctx = PipelineContext(
        video_path=Path(video_path),
        output_dir=resolved_output,
        debug_dir=resolved_debug,
        config=config,
        attendees=attendees or [],
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    resolved_debug.mkdir(parents=True, exist_ok=True)

    from vocolith.utils.progress import pipeline_progress, add_task, advance_task, update_task, complete_task

    with pipeline_progress():
        # Overall pipeline stage tracker — 9 stages
        stages = [
            "1. Extract audio", "2. Denoise", "3. Transcribe",
            "4. Diarize", "5. Sample frames", "6. OCR",
            "7. Resolve speakers", "8. Confirm speakers", "9. Generate notes",
        ]
        overall = add_task(
            f"[bold]Pipeline[/bold] — {ctx.video_path.name}",
            total=len(stages),
        )

        def _stage(n: int, label: str) -> None:
            log.debug("=== Stage %d: %s ===", n, label)
            update_task(overall, description=f"[bold]{stages[n-1]}[/bold]")

        return _run_pipeline_inner(
            ctx, config, overall, _stage,
            dry_run=dry_run, template=template,
            no_faces=no_faces, no_ocr=no_ocr,
        )


def _run_pipeline_inner(
    ctx: "PipelineContext",
    config: "AppConfig",
    overall_task,
    stage_fn,
    dry_run: bool,
    template: str | None,
    no_faces: bool,
    no_ocr: bool,
) -> "PipelineContext":
    """Execute pipeline stages within the active progress context."""
    from vocolith.stages.audio_extractor import extract_audio, AudioExtractionError
    from vocolith.stages.audio_denoiser import denoise_audio
    from vocolith.utils.progress import advance_task, add_task, complete_task

    # ── Stage 1: Audio extraction ────────────────────────────────────────────
    stage_fn(1, "Audio Extraction")
    try:
        ctx.audio_path, ctx.av_offset = extract_audio(
            ctx.video_path, ctx.debug_dir,
            sample_rate=config.audio.sample_rate,
        )
        advance_task(overall_task)
    except AudioExtractionError as exc:
        ctx.add_error("audio_extractor", str(exc), fatal=True)
        return ctx

    # ── Stage 2: Denoise ─────────────────────────────────────────────────────
    stage_fn(2, "Audio Denoising")
    if config.denoiser.enabled:
        try:
            ctx.denoised_path = denoise_audio(
                ctx.audio_path, ctx.debug_dir,
                stationary=config.denoiser.stationary,
            )
        except Exception as exc:
            ctx.add_error("denoiser", f"Denoising failed: {exc}. Using raw audio.")
    advance_task(overall_task)

    # ── Scale frame sampling interval by video duration ─────────────────────
    # A 1h meeting at 5s interval → ~740 frames → hour-long OCR run.
    # Scale up the interval so we keep at most ~200 frames regardless of length.
    effective_interval = config.frame_sampling.interval_seconds
    if ctx.effective_audio:
        try:
            import soundfile as _sf
            info = _sf.info(str(ctx.effective_audio))
            duration_s = info.duration
            max_frames = 200
            min_interval = config.frame_sampling.interval_seconds
            scaled = max(min_interval, duration_s / max_frames)
            if scaled > min_interval:
                log.info(
                    "Auto-scaling frame interval: %.0fs video → %.0fs interval (~%d frames max)",
                    duration_s, scaled, max_frames,
                )
                effective_interval = scaled
        except Exception:
            pass

    # ── Stage 3: Transcription ───────────────────────────────────────────────
    stage_fn(3, "Transcription")
    _run_transcription(ctx, config)
    advance_task(overall_task)   # 3/9 — transcription done

    # ── Stage 4: Diarization ─────────────────────────────────────────────────
    stage_fn(4, "Diarization")
    _run_diarization(ctx, config)
    advance_task(overall_task)   # 4/9 — diarization done

    # Abort early if transcription failed — no point running OCR
    if ctx.transcript is None:
        ctx.add_error("transcriber", "Transcription produced no output.", fatal=True)
        return ctx

    # Release WhisperX + pyannote models and clear CUDA cache so OCR has headroom
    _flush_gpu("audio pipeline (post transcribe+diarize)")

    # ── Stages 5+6: Video pipeline (frame sampling + OCR) ────────────────────
    # Advance overall AFTER the work is done — not before.
    stage_fn(5, "Frame Sampling")
    if not (no_ocr and no_faces):
        _run_video_stages(ctx, config, no_faces=no_faces, no_ocr=no_ocr,
                          frame_interval_override=effective_interval)
    advance_task(overall_task)  # stage 5 complete
    stage_fn(6, "OCR complete")
    advance_task(overall_task)  # stage 6 complete

    # ── Stage 7b: Apply OCR terminology correction ───────────────────────────
    if ctx.ocr_vocabulary and ctx.transcript:
        log.debug("=== Stage 7b: Terminology Post-correction ===")
        from vocolith.utils.text import correct_transcript_terminology
        ctx.transcript.segments = correct_transcript_terminology(
            ctx.transcript.segments, ctx.ocr_vocabulary
        )

    # ── Stage 7: Speaker resolution ──────────────────────────────────────────
    stage_fn(7, "Speaker Resolution")
    try:
        from vocolith.stages.speaker_resolver import resolve_speakers
        ctx = resolve_speakers(ctx)
    except Exception as exc:
        ctx.add_error("speaker_resolver", f"Speaker resolution error: {exc}")
    advance_task(overall_task)

    # ── Stage 8: Confirmation wizard (if enabled) ────────────────────────────
    stage_fn(8, "Confirm Speakers")
    if config.speaker_resolution.confirm_auto_identified:
        try:
            from vocolith.stages.identifier_wizard import run_confirmation_wizard
            run_confirmation_wizard(ctx, config=config)
        except Exception as exc:
            ctx.add_error("confirmation_wizard", f"Confirmation wizard error: {exc}")

    advance_task(overall_task)

    # ── Write transcript.md ───────────────────────────────────────────────────
    _write_transcript(ctx)

    # Release OCR (EasyOCR) and any remaining GPU tensors before LLM stage.
    # This gives local Ollama models (--local) free VRAM without a two-pass workflow.
    _flush_gpu("pre-LLM (post OCR/face)")

    # ── Stage 9: Note generation ─────────────────────────────────────────────
    stage_fn(9, "Note Generation")
    if not dry_run:
        log.debug("=== Stage 9: Note Generation ===")
        try:
            from vocolith.stages.note_generator import generate_notes
            ctx = generate_notes(ctx, template=template)
        except Exception as exc:
            ctx.add_error("note_generator", f"Note generation failed: {exc}")
    else:
        log.info("Dry-run mode: skipping LLM note generation.")
    advance_task(overall_task)

    log.debug("Pipeline complete. Output: %s", ctx.output_dir)
    if ctx.errors:
        log.warning("%d non-fatal issue(s) encountered during processing.", len(ctx.errors))

    return ctx


# ─── Internal helpers ────────────────────────────────────────────────────────

def _run_transcription(ctx: PipelineContext, config: AppConfig) -> None:
    """Stage 3: Transcribe audio."""
    from vocolith.stages.transcriber import transcribe

    audio = ctx.effective_audio
    if not audio:
        ctx.add_error("transcriber", "No audio available.", fatal=True)
        return

    initial_prompt: str | None = None
    if ctx.ocr_vocabulary:
        initial_prompt = "Technical terms: " + ", ".join(ctx.ocr_vocabulary[:60])

    log.debug("=== Stage 3: Transcription ===")
    try:
        tc = config.transcription
        ctx.transcript = transcribe(
            audio_path=audio,
            model_size=tc.model_size,
            compute_type=tc.compute_type,
            batch_size=tc.batch_size,
            language=tc.language,
            initial_prompt=initial_prompt,
        )
    except Exception as exc:
        ctx.add_error("transcriber", str(exc), fatal=True)
        return


def _run_diarization(ctx: PipelineContext, config: AppConfig) -> None:
    """Stage 4: Diarize audio — assign speaker labels to transcript segments."""
    from vocolith.stages.diarizer import diarize

    if not config.diarization.enabled or ctx.transcript is None:
        return

    audio = ctx.effective_audio
    if not audio:
        return

    log.debug("=== Stage 4: Diarization ===")
    hf_token = (
        config.transcription.huggingface_token
        or __import__("os").environ.get("HUGGINGFACE_TOKEN")
    )
    # When attendees are provided use their count as a LOWER BOUND hint, not
    # an upper bound ceiling.  Capping at len(attendees) when the actual meeting
    # has more speakers causes pyannote to merge distinct voices into one label.
    # Instead take the larger of the attendee count and the config maximum.
    effective_max_speakers = max(
        len(ctx.attendees) if ctx.attendees else 0,
        config.diarization.max_speakers,
    )
    log.info(
        "Diarization max_speakers=%d (from %s)",
        effective_max_speakers,
        "attendee list" if ctx.attendees else "config",
    )
    ctx.transcript = diarize(
        audio_path=audio,
        transcript=ctx.transcript,
        hf_token=hf_token,
        min_speakers=config.diarization.min_speakers,
        max_speakers=effective_max_speakers,
        suppress_warnings=config.diarization.suppress_warnings,
    )


def _run_video_stages(ctx: PipelineContext, config: AppConfig,
                       no_faces: bool = False, no_ocr: bool = False,
                       frame_interval_override: float | None = None) -> None:
    """Stages 5+6: Frame sampling, OCR, terminology, face detection."""
    interval = frame_interval_override or config.frame_sampling.interval_seconds
    log.debug("=== Stage 5: Frame Sampling ===")
    try:
        from vocolith.stages.frame_sampler import sample_frames
        ctx.frames = sample_frames(
            ctx.video_path,
            ctx.debug_dir / "frames",
            interval_seconds=interval,
            change_threshold=config.frame_sampling.change_threshold,
            top_strip_pct=config.frame_sampling.top_strip_pct,
            bottom_strip_pct=config.frame_sampling.bottom_strip_pct,
            av_offset=ctx.av_offset or 0.0,
            min_frames=config.frame_sampling.min_frames,
        )
    except Exception as exc:
        ctx.add_error("frame_sampler", f"Frame sampling failed: {exc}")
        return  # no frames = skip OCR+face

    if not no_ocr and ctx.frames:
        log.debug("=== Stage 6a: OCR Name + Terminology Extraction ===")
        try:
            from vocolith.stages.ocr_extractor import extract_ocr
            ctx.ocr_names, ctx.ocr_vocabulary, ctx.frame_name_map = extract_ocr(
                ctx.frames,
                languages=config.ocr.languages,
                confidence_threshold=config.ocr.confidence_threshold,
                min_name_freq=config.ocr.min_name_freq,
                ui_blacklist=config.ocr.ui_blacklist,
                extract_terminology=config.ocr.extract_terminology,
                ocr_workers=config.ocr.ocr_workers,
                ocr_gpu_worker=config.ocr.ocr_gpu_worker,
                ocr_noise_freq_threshold=config.ocr.ocr_noise_freq_threshold,
                ocr_gpu_frame_ratio=config.ocr.ocr_gpu_frame_ratio,
            )
            if ctx.ocr_names:
                log.info("OCR names found: %s", ctx.ocr_names)
        except Exception as exc:
            ctx.add_error("ocr_extractor", f"OCR failed: {exc}")

    if not no_faces and config.pipeline.enable_face_recognition and ctx.frames:
        log.debug("=== Stage 6b: Face Identification ===")
        try:
            from vocolith.stages.face_identifier import identify_faces
            ctx.face_detections = identify_faces(
                ctx.frames,
                tolerance=config.face_recognition.tolerance,
                model=config.face_recognition.model,
            )
        except Exception as exc:
            ctx.add_error("face_identifier", f"Face recognition failed: {exc}")


def _flush_gpu(label: str = "") -> None:
    """
    Force-collect Python objects and release PyTorch CUDA cache.

    Called between heavy GPU stages (WhisperX → OCR → LLM) so each stage
    starts with maximum available VRAM.  gc.collect() must precede
    empty_cache() — otherwise model tensors still referenced by Python
    objects won't be freed and the cache flush has no effect.
    """
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.debug("GPU cache flushed: %s", label or "pipeline")
    except Exception:
        pass


def _write_transcript(ctx: PipelineContext) -> None:
    """Write transcript.md to the output directory."""
    if not ctx.transcript:
        return
    from vocolith.utils.text import render_transcript_md
    import json

    meta = {
        "filename": ctx.video_path.name,
        "date": ctx.session_id[:8],
        "duration": ctx.transcript.duration_seconds,
        "language": ctx.transcript.language,
    }

    md = render_transcript_md(ctx.transcript.segments, meta)
    transcript_path = ctx.output_dir / "transcript.md"
    transcript_path.write_text(md, encoding="utf-8")
    log.info("Transcript written: %s", transcript_path)

    # Write diarization manifest — used by `vocolith identify` wizard
    debug_json = ctx.debug_dir / "diarization.json"
    manifest = {
        "meta": {
            "video_path": str(ctx.video_path.resolve()),
            "audio_path": str(ctx.effective_audio.resolve()) if ctx.effective_audio else None,
            "output_dir": str(ctx.output_dir.resolve()),
            "debug_dir": str(ctx.debug_dir.resolve()),
            "session_id": ctx.session_id,
            "duration_seconds": ctx.transcript.duration_seconds,
            "language": ctx.transcript.language,
            "speakers_detected": ctx.transcript.speakers_detected,
        },
        "segments": [s.model_dump() for s in ctx.transcript.segments],
    }
    debug_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

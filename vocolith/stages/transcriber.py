# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Stage 3: Transcription using WhisperX with word-level timestamps."""
from __future__ import annotations
import logging
import threading
import time
from pathlib import Path

from vocolith.models.transcript import DiarizedTranscript, TranscriptSegment, WordTimestamp
from vocolith.utils.gpu import get_whisper_config

log = logging.getLogger(__name__)


def transcribe(
    audio_path: Path,
    model_size: str = "auto",
    compute_type: str = "auto",
    batch_size: int | str = "auto",
    language: str | None = None,
    initial_prompt: str | None = None,
) -> DiarizedTranscript:
    """
    Transcribe audio using WhisperX with word-level forced alignment.

    Args:
        audio_path:     Path to the (denoised) WAV file.
        model_size:     "auto" (recommended) or explicit size.
        compute_type:   "auto", "float16", or "int8".
        batch_size:     "auto" or integer.
        language:       ISO 639-1 code or None for auto-detect.
        initial_prompt: Optional context string to bias transcription
                        (e.g. domain terminology extracted from OCR).

    Returns:
        DiarizedTranscript with segments and word timestamps. Speaker
        labels are NOT set here — that happens in diarizer.py.
    """
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX not installed. Run: "
            "pip install git+https://github.com/m-bain/whisperX.git"
        ) from exc

    hw = get_whisper_config(model_size, compute_type,
                             None if str(batch_size) == "auto" else int(batch_size))
    device = hw["device"]
    effective_model = hw["model_size"]
    effective_compute = hw["compute_type"]
    effective_batch = hw["batch_size"]

    # Flush any stale VRAM allocations before loading the model
    if device == "cuda":
        try:
            import torch
            torch.cuda.empty_cache()
            free_gb = (torch.cuda.get_device_properties(0).total_memory
                       - torch.cuda.memory_reserved(0)) / 1e9
            log.info(
                "GPU: %.2f GB free before loading model '%s' (compute=%s batch=%d)",
                free_gb, effective_model, effective_compute, effective_batch,
            )
        except Exception:
            pass
    else:
        log.info(
            "Loading WhisperX model '%s' on %s (compute=%s batch=%d)...",
            effective_model, device, effective_compute, effective_batch,
        )

    model = whisperx.load_model(
        effective_model,
        device,
        compute_type=effective_compute,
        language=language,
    )

    log.debug("Transcribing %s...", Path(audio_path).name)
    audio = whisperx.load_audio(str(audio_path))

    # Get audio duration for the progress description
    _dur_s: float = 0.0
    try:
        import soundfile as _sf
        _dur_s = _sf.info(str(audio_path)).duration
        _dur_str = f"{int(_dur_s // 3600)}h{int((_dur_s % 3600) // 60)}m" \
                   if _dur_s >= 3600 else f"{int(_dur_s // 60)}m{int(_dur_s % 60)}s"
    except Exception:
        _dur_str = "?"

    from vocolith.utils.progress import add_task, update_task, complete_task

    # RTF (real-time factor) estimate: observed ~0.28 on this GPU for large-v2.
    # Use 0.30 as a conservative estimate so the bar reaches ~100% slightly early.
    # Bar shows X/N virtual 5-second chunks — readable N/M instead of 0/? spinner.
    _RTF = 0.30
    _VIRTUAL_CHUNKS = max(10, int(_dur_s / 5))
    tx_task = add_task(
        f"Transcribing {_dur_str} audio  [{effective_model} {effective_compute} batch={effective_batch}]",
        total=_VIRTUAL_CHUNKS,
    )

    def _start_phase_ticker(total_chunks: int, est_s: float,
                             sleep_s: float = 3.0, task_id=None):
        """Start an RTF-estimate progress ticker. Returns (stop_event, thread).
        Each call creates independent state — no shared mutable variables."""
        target_task = task_id if task_id is not None else tx_task
        stop_ev = threading.Event()
        t0 = time.monotonic()  # captured in closure by value at call time

        def _tick() -> None:
            while not stop_ev.is_set():
                elapsed = time.monotonic() - t0
                done = min(int(elapsed / est_s * total_chunks), total_chunks - 1)
                update_task(target_task, completed=done)
                time.sleep(sleep_s)

        t = threading.Thread(target=_tick, daemon=True)
        t.start()
        return stop_ev, t

    def _stop_phase_ticker(stop_ev: threading.Event, t: threading.Thread) -> None:
        stop_ev.set()
        t.join(timeout=5)

    # Start transcription phase ticker.
    # Always stop it in a finally block so the thread never leaks, even when a
    # non-OOM exception escapes the degradation ladder and re-raises.
    _trans_est_s = max(_dur_s * _RTF, 10.0)
    _trans_stop, _trans_ticker = _start_phase_ticker(_VIRTUAL_CHUNKS, _trans_est_s)

    transcribe_kwargs: dict = {"batch_size": effective_batch}
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt
        log.debug("Using initial_prompt for terminology boost (%d chars)", len(initial_prompt))

    # Degradation ladder for OOM:
    #   1. cuda / configured compute / configured batch
    #   2. cuda / int8 / batch=1          (reduce precision + serialise)
    #   3. cpu  / int8 / small model      (guaranteed to work, slow)
    def _try_transcribe(mdl, kwargs: dict) -> dict:
        """Attempt transcription; return raw result dict on success."""
        return mdl.transcribe(audio, **kwargs)

    def _is_oom(exc: Exception) -> bool:
        return "out of memory" in str(exc).lower() or "cuda failed" in str(exc).lower()

    def _clear_gpu() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    result = None

    # Degradation ladder wrapped in try/finally so the ticker thread is always
    # stopped — even when a non-OOM exception escapes the ladder and re-raises.
    try:
        # Attempt 1: as configured (cuda / int8 / batch=4 for ~4GB cards)
        _clear_gpu()
        try:
            result = _try_transcribe(model, transcribe_kwargs)
        except Exception as exc:
            if not _is_oom(exc):
                raise
            from vocolith.utils.progress import status as _status
            _status(f"[yellow]⚠ OOM[/yellow] compute={effective_compute} batch={effective_batch} → retrying batch=1")
            log.warning("OOM at compute=%s batch=%d — trying int8 batch=1",
                        effective_compute, effective_batch)
            update_task(tx_task,
                        description=f"Transcribing {_dur_str} audio  [OOM → retrying batch=1]")
        finally:
            del model
            _clear_gpu()

        # Attempt 2: same GPU, int8, batch=1 (serialise inference to minimise activation memory)
        if result is None and device == "cuda":
            model2 = None
            _clear_gpu()
            update_task(tx_task,
                        description=f"Transcribing {_dur_str} audio  [{effective_model} int8 batch=1]")
            try:
                log.info("Retrying: %s int8 batch=1 on CUDA...", effective_model)
                model2 = whisperx.load_model(
                    effective_model, device,
                    compute_type="int8", language=language,
                )
                retry_kwargs: dict = {"batch_size": 1}
                if initial_prompt:
                    retry_kwargs["initial_prompt"] = initial_prompt
                result = _try_transcribe(model2, retry_kwargs)
            except Exception as exc2:
                if not _is_oom(exc2):
                    raise
                from vocolith.utils.progress import status as _status
                _status("[yellow]⚠ OOM at batch=1[/yellow] — falling back to [bold]CPU whisper-small[/bold] (slow)")
                log.warning("OOM at batch=1 — falling back to CPU with small model (slow but guaranteed)")
                update_task(tx_task,
                            description=f"Transcribing {_dur_str} audio  [OOM → CPU fallback]")
            finally:
                if model2 is not None:
                    del model2
                _clear_gpu()

        # Attempt 3: CPU, small model, int8 (no VRAM needed — always works)
        if result is None:
            update_task(tx_task,
                        description=f"Transcribing {_dur_str} audio  [whisper-small CPU — slow]")
            model_cpu = None   # initialise for safe finally reference
            _clear_gpu()
            log.warning("Loading whisper-small on CPU — transcription will be slower...")
            try:
                model_cpu = whisperx.load_model(
                    "small", "cpu",
                    compute_type="int8", language=language,
                )
                from vocolith.utils.gpu import get_whisper_config as _gwc
                cpu_kwargs: dict = {"batch_size": _gwc("small", "int8")["batch_size"]}
                if initial_prompt:
                    cpu_kwargs["initial_prompt"] = initial_prompt
                result = _try_transcribe(model_cpu, cpu_kwargs)
                log.info("CPU transcription complete.")
            finally:
                if model_cpu is not None:
                    del model_cpu
                _clear_gpu()

        # Guard: all three attempts failed (should never happen, but be safe)
        if result is None:
            raise RuntimeError(
                "Transcription failed: all three attempts (cuda/batch, cuda/batch=1, cpu/small) "
                "returned no result."
            )
    finally:
        _stop_phase_ticker(_trans_stop, _trans_ticker)

    detected_lang = result.get("language", language or "en")
    n_raw = len(result.get("segments", []))
    log.info("Detected language: %s  (%d raw segments)", detected_lang, n_raw)

    # Stop transcription ticker and hide the transcription bar — it is done.
    complete_task(tx_task, description=f"Transcribed {_dur_str} → {n_raw} segments")

    # Fresh bar for alignment so the total doesn't jump from 742→N on the same row.
    _ALIGN_CHUNKS = max(5, n_raw)
    _align_est_s = max(_dur_s * 0.06, 5.0)
    align_task = add_task(
        f"Aligning word timestamps  [{n_raw} segments, lang={detected_lang}]",
        total=_ALIGN_CHUNKS,
    )
    _align_stop, _align_ticker = _start_phase_ticker(
        _ALIGN_CHUNKS, _align_est_s, sleep_s=2.0, task_id=align_task
    )

    # Force word-level alignment; delete alignment model immediately after to free VRAM
    log.debug("Aligning word timestamps...")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=device
        )
        try:
            result = whisperx.align(
                result["segments"], model_a, metadata, audio, device,
                return_char_alignments=False,
            )
        finally:
            del model_a
            _clear_gpu()
    except Exception as exc:
        log.warning("Word alignment failed (%s); using segment-level timestamps.", exc)

    _stop_phase_ticker(_align_stop, _align_ticker)

    # Convert to our model.  Always complete align_task even if conversion
    # raises so the progress bar doesn't linger on an exception path.
    segments = []
    duration = 0.0  # initialise before try so log.info below is never unbound
    try:
        raw_segs = result.get("segments", [])

        # Estimate total duration from last segment
        duration = raw_segs[-1]["end"] if raw_segs else 0.0

        for i, seg in enumerate(raw_segs):
            words = [
                WordTimestamp(
                    word=w.get("word", "").strip(),
                    start=w.get("start", seg["start"]),
                    end=w.get("end", seg["end"]),
                    score=w.get("score", 1.0),
                )
                for w in seg.get("words", [])
            ]
            segments.append(TranscriptSegment(
                segment_id=i,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                words=words,
            ))
    finally:
        complete_task(align_task,
                      description=f"Aligned {len(segments)} segments  ({_dur_str})")

    log.info(
        "Transcription complete: %d segments, %.0fs duration",
        len(segments), duration,
    )
    from vocolith.utils.progress import status as _status
    _status(f"[green]✓ Transcription[/green]  {len(segments)} segments  {_dur_str}  lang={detected_lang}")

    # GPU cache is already cleared by the degradation ladder's finally blocks.
    # One final flush in case any alignment model residue remains.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.debug("GPU cache cleared after transcription.")
    except Exception:
        pass

    return DiarizedTranscript(
        segments=segments,
        language=detected_lang,
        duration_seconds=duration,
        speakers_detected=0,  # set by diarizer
    )

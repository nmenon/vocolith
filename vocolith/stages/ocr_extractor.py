# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""
Stage 6a+6b: OCR extraction — attendee names AND domain terminology from video frames.

Two outputs:
  - ocr_names: list of likely participant display names
  - ocr_vocabulary: list of domain terms for Whisper hotword injection
"""
from __future__ import annotations
import logging
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vocolith.stages.frame_sampler import SampledFrame

# ── Per-worker process state ──────────────────────────────────────────────────
# Each worker process holds its own EasyOCR reader so the model is loaded once
# per worker (not once per frame).  CPU-only to avoid GPU contention between
# parallel workers; the main process reader uses GPU if available.
_worker_reader = None


def _worker_init(languages: list[str]) -> None:
    """Initialise a CPU-only EasyOCR reader in each worker process."""
    global _worker_reader
    import easyocr  # noqa: PLC0415
    _worker_reader = easyocr.Reader(languages, gpu=False, verbose=False)


def _process_one_frame(args: tuple) -> tuple[int, list[str], dict, dict, float]:
    """
    Worker entry point: OCR one frame's strips + full image.

    Returns (frame_idx, frame_names, name_counts, term_counts, timestamp_s).
    All heavy numpy data lives in-process; only the small result dicts cross
    the process boundary back to the main process.
    """
    (frame_idx, top_strip, bottom_strip, full_image,
     timestamp_s, confidence_threshold, blacklist_tuple,
     extract_terminology) = args

    if _worker_reader is None:
        return frame_idx, [], {}, {}, timestamp_s

    blacklist = frozenset(blacklist_tuple)
    name_counts: dict[str, int] = {}
    frame_names: list[str] = []

    for strip in (top_strip, bottom_strip):
        if strip is None or strip.size == 0:
            continue
        enhanced = _enhance_for_ocr(strip)
        try:
            results = _worker_reader.readtext(enhanced, detail=1)
        except Exception as _exc:
            log.debug("OCR worker strip error (frame %d): %s", frame_idx, _exc)
            continue
        for (_, text, conf) in results:
            if conf < confidence_threshold:
                continue
            cleaned = _clean_name_candidate(text, blacklist)
            if cleaned:
                name_counts[cleaned] = name_counts.get(cleaned, 0) + 1
                frame_names.append(cleaned)

    term_counts: dict[str, int] = {}
    if extract_terminology and full_image is not None and full_image.size > 0:
        h, w = full_image.shape[:2]
        import cv2 as _cv2  # noqa: PLC0415
        scale = min(1.0, 1280 / max(w, 1))
        small = _cv2.resize(full_image, (int(w * scale), int(h * scale))) if scale < 1.0 else full_image
        try:
            results = _worker_reader.readtext(small, detail=1)
        except Exception as _exc:
            log.debug("OCR worker term error (frame %d): %s", frame_idx, _exc)
            results = []
        for (_, text, conf) in results:
            if conf < confidence_threshold:
                continue
            for word in text.split():
                if _is_domain_term(word):
                    term_counts[word] = term_counts.get(word, 0) + 1

    return frame_idx, frame_names, name_counts, term_counts, timestamp_s

log = logging.getLogger(__name__)

# Terms that are clearly UI chrome, not names or domain vocabulary
_DEFAULT_BLACKLIST = frozenset({
    "mute", "unmute", "video", "participants", "chat", "share",
    "record", "leave", "end", "reactions", "more", "settings",
    "raise hand", "view", "pin", "spotlight", "host", "waiting room",
    "stop", "start", "join", "audio", "camera", "screen", "present",
    "meeting", "webex", "zoom", "teams", "meet",
})

# Patterns that look like domain terminology (not common English words)
# Deliberately conservative to reduce false positives
_TERM_PATTERNS = [
    # All-caps acronyms: 2-10 chars (LPDDR, ADAS, BMS, TDA4, WOODY, JEDEC)
    # Excludes single-char initials and overly short noise
    r'^[A-Z]{2,10}[0-9]{0,4}$',
    # CamelCase identifiers (min 2 humps): WhisperX, PyAnnote, OpenCV
    r'^[A-Z][a-z]{2,}[A-Z][a-zA-Z0-9]{2,}$',
    # Part numbers with hyphen: TDA4VH-Q1, STM32-F4, LPDDR5-6400
    r'^[A-Z]{2,8}[0-9]{1,5}[A-Z]{0,3}-[A-Z0-9]{1,8}$',
    # Mixed alphanumeric ending in SI units: 32GB, 4KHz, 1080p, 400MHz
    r'^\d+[A-Z]{2,4}$',
    # Version-like: v2.3, v1.5.4
    r'^v\d+\.\d+',
]
_TERM_RE = [re.compile(p) for p in _TERM_PATTERNS]

# Common English words that match our patterns but are not domain terms
_TERM_STOPWORDS = frozenset({
    "THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "HAVE",
    "WILL", "BEEN", "THEN", "WHEN", "INTO", "THEY", "YOUR",
    "EACH", "WERE", "MORE", "OVER", "ALSO", "SOME", "AFTER",
})


def extract_ocr(
    frames: list["SampledFrame"],
    languages: list[str] | None = None,
    confidence_threshold: float = 0.5,
    min_name_freq: int = 2,
    ui_blacklist: list[str] | None = None,
    extract_terminology: bool = True,
    ocr_workers: int = 0,
    ocr_gpu_worker: bool = True,
    ocr_noise_freq_threshold: float = 0.8,
    ocr_gpu_frame_ratio: float = 0.0,
) -> tuple[list[str], list[str], dict[float, list[str]]]:
    """
    Run OCR on sampled frame strips and return (names, vocabulary).

    Names are extracted from the top/bottom strips (participant overlays).
    Vocabulary is extracted from the full frame (slides, documents, chat).

    Args:
        frames:               List of SampledFrame objects from frame_sampler.
        languages:            EasyOCR language list (default: ["en"]).
        confidence_threshold: Minimum EasyOCR confidence score (0-1).
        min_name_freq:        Name must appear in N+ frames to be counted.
        ui_blacklist:         Strings to ignore (UI labels).
        extract_terminology:  Also extract domain terms for Whisper hotword injection.

    Returns:
        Tuple (ocr_names, ocr_vocabulary).
        ocr_names: deduplicated participant names sorted by frequency.
        ocr_vocabulary: deduplicated domain terms sorted by frequency.
    """
    if not frames:
        return [], [], {}

    try:
        import easyocr
    except ImportError:
        log.warning("easyocr not installed — skipping OCR. pip install easyocr")
        return [], [], {}

    languages = languages or ["en"]
    blacklist = frozenset(b.lower() for b in (ui_blacklist or [])) | _DEFAULT_BLACKLIST

    total = len(frames)

    # GPU path: one reader, sequential but ~7× faster per frame than CPU.
    # Takes priority over worker pool when CUDA is available and not disabled.
    use_gpu = ocr_gpu_worker and _has_gpu()

    if use_gpu:
        # GPU reader runs sequentially in the main process (~1 s/frame).
        # CPU workers run in parallel subprocesses on the remaining frames
        # (~7 s/frame each), overlapping with the GPU phase.
        # Frame split: x_gpu / 1 ≈ (total - x_gpu) * 7 / n_cpu  → balance wall time.
        if ocr_workers == 1:
            n_cpu_workers = 0   # explicit: GPU-only, no CPU parallelism
        elif ocr_workers > 1:
            n_cpu_workers = ocr_workers - 1
        else:
            # auto with GPU present: GPU-only by default.
            # CPU workers take 60-120s/frame to initialise the model and risk
            # hitting the drain timeout before producing any results.
            # Pass --ocr-workers N (N>=2) to opt in to CPU worker parallelism.
            n_cpu_workers = 0

        if n_cpu_workers > 0:
            if ocr_gpu_frame_ratio > 0.0:
                # Explicit user override: pin GPU fraction
                x_gpu = max(1, min(int(total * ocr_gpu_frame_ratio), total - 1))
            else:
                # Auto-balance: equalise GPU and CPU wall time.
                # GPU: 1 s/frame; each CPU worker: 7 s/frame.
                # x_gpu / 1 = (total - x_gpu) * 7 / n_cpu_workers
                x_gpu = max(1, min(int(total * 7 / (n_cpu_workers + 7)), total - 1))
        else:
            x_gpu = total

        n_workers = 1  # GPU path marker — sequential GPU below
        log.info(
            "OCR: GPU handles %d frames, %d CPU worker(s) handle %d frames (parallel)",
            x_gpu, n_cpu_workers, total - x_gpu,
        )
    else:
        n_cpu_workers = 0
        x_gpu = 0
        # CPU-only path: parallel worker processes.
        # Each worker loads its own EasyOCR model (~300 MB RAM) so cap at 6.
        if ocr_workers > 0:
            n_workers = min(ocr_workers, total)
        else:
            n_workers = min(max(1, (os.cpu_count() or 2) // 2), total, 6)
        log.info("OCR: %d frames, %d CPU worker(s) (parallel)", total, n_workers)

    from vocolith.utils.progress import add_task, advance_task, complete_task, update_task
    ocr_task = add_task(f"OCR {total} frames", total=total)

    name_counter: Counter[str] = Counter()
    term_counter: Counter[str] = Counter()
    # Per-frame name map: {timestamp_s: [names visible in that frame]}
    # Used by speaker_resolver for temporal correlation
    frame_name_map: dict[float, list[str]] = {}

    _report_every = max(1, total // 10)
    blacklist_tuple = tuple(blacklist)

    completed = 0

    def _advance() -> None:
        """Advance progress bar and log/update description every ~10%."""
        nonlocal completed
        completed += 1
        advance_task(ocr_task)
        if completed % _report_every == 0 or completed == total:
            seen = list(name_counter.keys())[:5]
            log.info(
                "OCR progress: %d/%d frames (%d%%)  names so far: %s",
                completed, total, completed * 100 // total,
                ", ".join(seen) if seen else "none yet",
            )
            if seen:
                update_task(ocr_task,
                            description=f"OCR {total} frames — found: {', '.join(seen[:3])}")

    def _collect_future(future) -> None:
        """Merge one worker future result into shared counters."""
        try:
            _, frame_names, name_counts, term_counts, timestamp = future.result()
        except Exception as exc:
            log.debug("OCR worker error: %s", exc)
            return
        name_counter.update(name_counts)
        term_counter.update(term_counts)
        if frame_names:
            frame_name_map[timestamp] = frame_names

    # ── CPU-only parallel path ────────────────────────────────────────────────
    if not use_gpu and n_workers > 1:
        try:
            frame_args = [
                (idx, sf.top_strip, sf.bottom_strip,
                 sf.full_image if extract_terminology else None,
                 sf.frame.timestamp_s, confidence_threshold,
                 blacklist_tuple, extract_terminology)
                for idx, sf in enumerate(frames)
            ]
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_worker_init,
                initargs=(languages,),
            ) as executor:
                futures = {executor.submit(_process_one_frame, arg): arg[0]
                           for arg in frame_args}
                for future in as_completed(futures):
                    _collect_future(future)
                    _advance()
        except Exception as pool_exc:
            log.warning("OCR CPU pool failed (%s) — falling back to sequential CPU.", pool_exc)
            # Sequential CPU fallback
            reader_cpu = easyocr.Reader(languages, gpu=False, verbose=False)
            for _, sf in enumerate(frames):
                frame_names_seq: list[str] = []
                for strip in [sf.top_strip, sf.bottom_strip]:
                    if strip is None or strip.size == 0:
                        continue
                    frame_names_seq.extend(
                        _process_strip_for_names(strip, reader_cpu, confidence_threshold,
                                                 blacklist, name_counter) or [])
                if frame_names_seq:
                    frame_name_map[sf.frame.timestamp_s] = frame_names_seq
                if extract_terminology:
                    _process_frame_for_terms(sf.full_image, reader_cpu,
                                             confidence_threshold, term_counter)
                _advance()

    # ── GPU path (sequential) with optional CPU workers on remaining frames ───
    else:
        reader = easyocr.Reader(languages, gpu=use_gpu, verbose=False)

        # Bug fix: when use_gpu=False (CPU-only sequential, ended up here because
        # n_workers<=1), all frames must go through the single reader below.
        if not use_gpu:
            x_gpu = total

        # Smoke-test: catch GPU failures BEFORE splitting and launching workers.
        # Fixes double-processing: if smoke-test fails we reset x_gpu BEFORE
        # computing the frame split — CPU workers are not yet submitted.
        if use_gpu and frames:
            _test_strip = next(
                (sf.top_strip for sf in frames[:5]
                 if sf.top_strip is not None and sf.top_strip.size > 0),
                None,
            )
            if _test_strip is not None:
                try:
                    reader.readtext(_enhance_for_ocr(_test_strip), detail=1)
                    log.debug("OCR GPU reader smoke-test passed.")
                except Exception as _smoke_exc:
                    log.warning(
                        "OCR GPU reader failed smoke-test (%s) — reinitialising with CPU.",
                        _smoke_exc,
                    )
                    reader = easyocr.Reader(languages, gpu=False, verbose=False)
                    use_gpu = False
                    x_gpu = total    # single reader handles all frames; n_cpu_workers ignored

        # Split frames now (after smoke-test may have adjusted x_gpu).
        gpu_frames = frames[:x_gpu]
        cpu_frames = frames[x_gpu:]

        # Launch CPU workers AFTER smoke-test so x_gpu is finalised.
        # Fix: use 'spawn' start method so workers start with a clean process
        # state and do NOT inherit the parent's CUDA context.  'fork' (Linux
        # default) copies the active CUDA state into the child which deadlocks
        # PyTorch BLAS operations in the worker — workers never complete.
        cpu_executor = None
        cpu_futures: dict = {}
        if n_cpu_workers > 0 and cpu_frames:
            try:
                import multiprocessing as _mp
                cpu_args = [
                    (x_gpu + idx, sf.top_strip, sf.bottom_strip,
                     sf.full_image if extract_terminology else None,
                     sf.frame.timestamp_s, confidence_threshold,
                     blacklist_tuple, extract_terminology)
                    for idx, sf in enumerate(cpu_frames)
                ]
                cpu_executor = ProcessPoolExecutor(
                    max_workers=n_cpu_workers,
                    initializer=_worker_init,
                    initargs=(languages,),
                    mp_context=_mp.get_context("spawn"),  # no CUDA fork deadlock
                )
                cpu_futures = {cpu_executor.submit(_process_one_frame, arg): i
                               for i, arg in enumerate(cpu_args)}
            except Exception as exc:
                log.warning("OCR: could not start CPU workers (%s) — GPU-only.", exc)
                cpu_executor = None
                cpu_futures = {}
                gpu_frames = frames       # GPU handles everything
                cpu_frames = []

        # GPU sequential loop — poll cpu_futures between frames so CPU results
        # appear in the progress bar in real-time rather than in a burst after
        # the GPU loop.  wait(timeout=0) is non-blocking: it returns any futures
        # that are already done without delaying the GPU loop.
        _remaining_cpu = set(cpu_futures.keys())

        for _, sf in enumerate(gpu_frames):
            frame_names_seq: list[str] = []
            for strip in [sf.top_strip, sf.bottom_strip]:
                if strip is None or strip.size == 0:
                    continue
                frame_names_seq.extend(
                    _process_strip_for_names(strip, reader, confidence_threshold,
                                             blacklist, name_counter) or [])
            if frame_names_seq:
                frame_name_map[sf.frame.timestamp_s] = frame_names_seq
            if extract_terminology:
                _process_frame_for_terms(sf.full_image, reader,
                                         confidence_threshold, term_counter)
            _advance()

            # Non-blocking harvest of any CPU futures that finished while GPU
            # was processing this frame.
            if _remaining_cpu:
                from concurrent.futures import wait as _cf_wait, FIRST_COMPLETED
                _done, _remaining_cpu = _cf_wait(
                    _remaining_cpu, timeout=0, return_when=FIRST_COMPLETED
                )
                for _f in _done:
                    _collect_future(_f)
                    _advance()

        # Drain any CPU futures still running after the GPU loop finishes.
        # Timeout = 30 s/frame × remaining frames (generous ceiling for slow CPU).
        # TimeoutError means workers are stuck (e.g. CUDA fork deadlock) — log
        # and move on rather than blocking the pipeline indefinitely.
        if _remaining_cpu:
            import concurrent.futures as _cf_mod
            _cpu_timeout_s = max(120, len(_remaining_cpu) * 120)
            try:
                for future in as_completed(_remaining_cpu, timeout=_cpu_timeout_s):
                    _collect_future(future)
                    _advance()
            except _cf_mod.TimeoutError:
                log.warning(
                    "OCR CPU workers timed out after %ds — %d frame(s) skipped.",
                    _cpu_timeout_s, len(_remaining_cpu),
                )

        if cpu_executor:
            import sys as _sys
            if _sys.version_info >= (3, 9):
                cpu_executor.shutdown(wait=False, cancel_futures=True)
            else:
                cpu_executor.shutdown(wait=False)

    # ── Static-noise filter ───────────────────────────────────────────────────
    # Names present in >threshold fraction of ALL frames are always-on UI chrome
    # (browser tabs, window titles, persistent overlays) — not participant names.
    # Real participant names only appear when that person's panel is visible.
    if ocr_noise_freq_threshold < 1.0 and total > 0:
        # Count distinct frames each name appears in (use frame_name_map not
        # name_counter, because name_counter inflates counts via top+bottom strips).
        name_frame_count: Counter[str] = Counter()
        for names_in_frame in frame_name_map.values():
            for name in set(names_in_frame):
                name_frame_count[name] += 1

        noise_names = {
            name for name, fcount in name_frame_count.items()
            if fcount / total > ocr_noise_freq_threshold
        }
        if noise_names:
            log.info(
                "OCR static-noise filter removed %d name(s) seen in >%.0f%% of frames: %s",
                len(noise_names), ocr_noise_freq_threshold * 100, sorted(noise_names),
            )
            for name in noise_names:
                name_counter.pop(name, None)
            # Also scrub from frame_name_map so _try_ocr_match never sees them
            for ts in frame_name_map:
                frame_name_map[ts] = [n for n in frame_name_map[ts] if n not in noise_names]

    # Filter names by minimum frequency
    ocr_names = [
        name for name, count in name_counter.most_common()
        if count >= min_name_freq
    ]

    # Sort terms by frequency
    ocr_vocabulary = [
        term for term, _ in term_counter.most_common(80)
    ]

    complete_task(ocr_task)
    if ocr_names:
        log.info("OCR names detected: %s", ocr_names[:10])
    if ocr_vocabulary:
        log.info("OCR vocabulary: %d terms extracted", len(ocr_vocabulary))

    return ocr_names, ocr_vocabulary, frame_name_map


def _enhance_for_ocr(strip: np.ndarray) -> np.ndarray:
    """Apply contrast enhancement to improve OCR accuracy on name overlays."""
    if cv2 is None:
        return strip
    if strip.ndim == 3:
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    else:
        gray = strip
    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    # Convert back to BGR for EasyOCR
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _process_strip_for_names(
    strip: np.ndarray,
    reader,
    confidence_threshold: float,
    blacklist: frozenset,
    counter: Counter,
) -> list[str]:
    """Run OCR on a strip, accumulate names in counter, and return names found."""
    enhanced = _enhance_for_ocr(strip)
    found: list[str] = []
    try:
        results = reader.readtext(enhanced, detail=1)
    except Exception as exc:
        log.warning("OCR strip error: %s", exc)
        return found

    for (_, text, conf) in results:
        if conf < confidence_threshold:
            continue
        cleaned = _clean_name_candidate(text, blacklist)
        if cleaned:
            counter[cleaned] += 1
            found.append(cleaned)
    return found


def _process_frame_for_terms(
    frame: np.ndarray,
    reader,
    confidence_threshold: float,
    counter: Counter,
) -> None:
    """Run OCR on the full frame and accumulate domain terminology."""
    if cv2 is None:
        return
    # Downscale for faster processing (terms are usually large text on slides)
    h, w = frame.shape[:2]
    scale = min(1.0, 1280 / max(w, 1))
    if scale < 1.0:
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        small = frame

    try:
        results = reader.readtext(small, detail=1)
    except Exception as exc:
        log.debug("OCR full frame error: %s", exc)
        return

    for (_, text, conf) in results:
        if conf < confidence_threshold:
            continue
        for word in text.split():
            if _is_domain_term(word):
                counter[word] += 1


def _clean_name_candidate(text: str, blacklist: frozenset) -> str | None:
    """
    Heuristic filter: return cleaned name if text looks like a person's name,
    else None.
    """
    from vocolith.utils.text import clean_ocr_name
    cleaned = clean_ocr_name(text)
    if not cleaned:
        return None
    if cleaned.lower() in blacklist:
        return None
    # Must have at least one alphabetic character
    if not any(c.isalpha() for c in cleaned):
        return None
    return cleaned


def _is_domain_term(word: str) -> bool:
    """Return True if the word looks like a domain-specific technical term."""
    if len(word) < 2 or len(word) > 20:
        return False
    if word.upper() in _TERM_STOPWORDS:
        return False
    for pattern in _TERM_RE:
        if pattern.match(word):
            return True
    return False


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# Deferred import to avoid circular import.
# cv2 = None sentinel ensures NameError is replaced with a clear None-guard
# in _enhance_for_ocr and _process_frame_for_terms.
cv2 = None
try:
    import cv2  # type: ignore[assignment]
except ImportError:
    log.warning("opencv-python not installed — frame processing unavailable.")

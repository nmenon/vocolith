# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""
Stage 6b: Optional face recognition from video frames.

Requires: pip install cmake dlib face-recognition
Enable via: config.yaml -> pipeline.enable_face_recognition: true

This is a supporting (low-confidence) signal. Since cameras are
frequently off in real meetings, face recognition is intentionally
given lower priority than OCR and voice embedding matching.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vocolith.stages.frame_sampler import SampledFrame
    from vocolith.models.meeting import FaceDetection

log = logging.getLogger(__name__)


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def identify_faces(
    frames: list["SampledFrame"],
    tolerance: float = 0.6,
    model: str | None = None,   # None = auto (cnn if GPU, hog if CPU)
    min_face_height_px: int = 30,
) -> list["FaceDetection"]:
    """
    Detect and encode faces in sampled video frames.

    Args:
        frames:             Sampled frames from frame_sampler.
        tolerance:          Face comparison tolerance (lower = stricter).
        model:              "hog" (CPU-fast) or "cnn" (GPU-accurate).
        min_face_height_px: Minimum face height in pixels to consider.

    Returns:
        List of FaceDetection objects with face locations and encodings.
        Returns empty list if face_recognition is not installed (graceful).
    """
    try:
        import face_recognition
    except ImportError:
        log.debug(
            "face_recognition not installed — face identification skipped. "
            "Install with: pip install cmake dlib face-recognition"
        )
        return []

    from vocolith.models.meeting import FaceDetection
    import numpy as np

    detections: list[FaceDetection] = []

    for sf in frames:
        img = sf.full_image
        if img is None or img.size == 0:
            continue

        # face_recognition expects RGB, OpenCV gives BGR
        import cv2
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Auto-select model: CNN is more accurate for small faces but needs GPU
        effective_model = model or ("cnn" if _has_gpu() else "hog")
        try:
            locations = face_recognition.face_locations(rgb, model=effective_model)
        except Exception as exc:
            log.debug("face_locations failed on frame %s: %s",
                      sf.frame.timestamp_s, exc)
            continue

        if not locations:
            continue

        # Filter by minimum face size
        valid_locations = []
        for loc in locations:
            top, _right, bottom, _left = loc
            face_h = bottom - top
            if face_h >= min_face_height_px:
                valid_locations.append(loc)

        if not valid_locations:
            continue

        try:
            encodings = face_recognition.face_encodings(rgb, valid_locations)
        except Exception as exc:
            log.debug("face_encodings failed: %s", exc)
            continue

        for loc, enc in zip(valid_locations, encodings):
            detections.append(FaceDetection(
                frame_timestamp_s=sf.frame.timestamp_s,
                face_location=loc,
                face_encoding=enc.tolist(),
            ))

    log.info("Face detection: %d face(s) found in %d frames",
             len(detections), len(frames))
    return detections

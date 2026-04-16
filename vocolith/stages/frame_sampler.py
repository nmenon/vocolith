# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Stage 5: Sample video frames at regular intervals for OCR and face analysis."""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vocolith.models.meeting import VideoFrame

log = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    """A sampled video frame with both the full frame and name-strip crops."""
    frame: VideoFrame
    full_image: np.ndarray              # full frame (H, W, 3) BGR
    top_strip: np.ndarray | None        # top N% of frame
    bottom_strip: np.ndarray | None     # bottom N% of frame


def sample_frames(
    video_path: Path,
    output_dir: Path | None = None,
    interval_seconds: float = 5.0,
    change_threshold: float = 0.002,
    top_strip_pct: float = 0.25,
    bottom_strip_pct: float = 0.15,
    av_offset: float = 0.0,
    save_debug_frames: bool = False,
    min_frames: int = 6,
) -> list[SampledFrame]:
    """
    Sample frames from a video at regular intervals.

    Applies a change-detection filter to skip frames that are visually
    similar to the previous sampled frame (avoids redundant OCR on
    long static screen-sharing segments).

    Crops top and bottom strips where video conferencing UIs typically
    show participant name overlays.

    Args:
        video_path:       Input video file.
        output_dir:       Optional directory to save sampled frames as JPEG.
        interval_seconds: Target sampling interval (default: 5s).
        change_threshold: Minimum mean absolute difference (0-1) between
                          consecutive frames to include a new sample.
        top_strip_pct:    Fraction of frame height for the top name strip.
        bottom_strip_pct: Fraction of frame height for the bottom name strip.
        av_offset:        A/V time offset in seconds (applied to timestamps).
        save_debug_frames: If True, save sampled frames as JPEG to output_dir.

    Returns:
        List of SampledFrame objects (may be empty if video has no frames).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        log.warning("Could not open video: %s", video_path.name)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    frame_interval = max(1, int(fps * interval_seconds))

    expected = max(1, total_frames // frame_interval)
    log.info(
        "Sampling frames: %.0fs duration, %.1f fps, interval=%ds (~%d frames)",
        duration, fps, interval_seconds, expected,
    )

    from vocolith.utils.progress import add_task, advance_task
    fs_task = add_task(f"Sampling frames (~{expected})", total=expected)

    if output_dir and save_debug_frames:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    sampled: list[SampledFrame] = []
    prev_gray: np.ndarray | None = None
    frame_idx = 0
    sample_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_s = (frame_idx / fps) + av_offset
                # Advance for every candidate frame checked (not just kept ones)
                # so the bar reaches 100% regardless of change-detection filtering.
                advance_task(fs_task)

                # Change detection: compare with previous sampled frame.
                # Skip if too similar, BUT guarantee min_frames by bypassing the
                # filter when we have few samples relative to total duration.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None:
                    diff = np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0
                    expected_samples = max(min_frames, int(duration / interval_seconds))
                    too_similar = diff < change_threshold
                    have_enough = len(sampled) >= min_frames
                    if too_similar and have_enough and len(sampled) >= expected_samples * 0.5:
                        prev_gray = gray  # update so next compare uses latest seen frame
                        frame_idx += 1
                        continue

                prev_gray = gray
                h = frame.shape[0]

                # Crop name-overlay strips
                top_h = max(1, int(h * top_strip_pct))
                bottom_h = max(1, int(h * bottom_strip_pct))
                top_strip = frame[:top_h, :, :]
                bottom_strip = frame[h - bottom_h:, :, :]

                vf = VideoFrame(
                    timestamp_s=timestamp_s,
                    frame_index=frame_idx,
                    source_region="sampled",
                )
                sf = SampledFrame(
                    frame=vf,
                    full_image=frame.copy(),
                    top_strip=top_strip.copy(),
                    bottom_strip=bottom_strip.copy(),
                )
                sampled.append(sf)

                if output_dir and save_debug_frames:
                    out_path = output_dir / f"frame_{sample_idx:04d}_{timestamp_s:.1f}s.jpg"
                    cv2.imwrite(str(out_path), frame)

                sample_idx += 1

            frame_idx += 1
    finally:
        cap.release()

    from vocolith.utils.progress import complete_task
    complete_task(fs_task, description=f"Frames sampled: {len(sampled)}/{expected} kept")
    log.info("Frame sampling complete: %d frames sampled", len(sampled))
    return sampled

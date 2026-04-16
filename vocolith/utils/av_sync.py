# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Audio-video sync detection via ffprobe."""
from __future__ import annotations
import json
import logging
import subprocess

log = logging.getLogger(__name__)


def get_av_offset_seconds(video_path: str) -> float | None:
    """
    Use ffprobe to detect the start time offset between audio and video streams.
    Returns offset in seconds (positive = audio starts after video), or None if
    ffprobe fails or video has a single stream.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            log.debug("ffprobe failed: %s", result.stderr[:200])
            return None

        data = json.loads(result.stdout)
        streams = data.get("streams", [])

        audio_start = None
        video_start = None
        for stream in streams:
            codec_type = stream.get("codec_type")
            start = stream.get("start_time")
            if start is not None:
                start = float(start)
                if codec_type == "audio":
                    audio_start = start
                elif codec_type == "video":
                    video_start = start

        if audio_start is not None and video_start is not None:
            offset = audio_start - video_start
            if abs(offset) > 0.05:
                log.warning(
                    "A/V sync offset detected: %.3fs (audio leads/lags video). "
                    "Frame timestamps will be adjusted.",
                    offset,
                )
            return offset

    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
        log.debug("A/V sync check skipped: %s", exc)

    return None

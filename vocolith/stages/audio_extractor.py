# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Stage 1: Extract mono 16kHz WAV from video file using ffmpeg."""
from __future__ import annotations
import logging
import subprocess
from pathlib import Path

from vocolith.utils.av_sync import get_av_offset_seconds

log = logging.getLogger(__name__)


class AudioExtractionError(RuntimeError):
    pass


def extract_audio(video_path: Path, output_dir: Path,
                   sample_rate: int = 16000) -> tuple[Path, float | None]:
    """
    Extract mono WAV audio from a video file.

    Returns:
        (wav_path, av_offset_seconds) where av_offset is the audio-video
        start time difference (or None if ffprobe unavailable).

    Raises:
        AudioExtractionError: if ffmpeg fails or video has no audio stream.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = output_dir / "audio_raw.wav"

    log.info("Extracting audio from %s", video_path.name)

    # First check if the file has an audio stream
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True)
    if "audio" not in probe.stdout:
        raise AudioExtractionError(
            f"No audio stream found in {video_path.name}. "
            "Cannot transcribe a video without audio."
        )

    # Check A/V sync before extraction
    av_offset = get_av_offset_seconds(str(video_path))

    # Extract audio
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ac", "1",               # mono
        "-ar", str(sample_rate),  # 16 kHz
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise AudioExtractionError(
            f"ffmpeg failed (exit {result.returncode}): {result.stderr[-500:]}"
        )

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise AudioExtractionError("ffmpeg produced empty output file.")

    size_mb = wav_path.stat().st_size / (1024 ** 2)
    log.debug("Audio extracted: %.1f MB -> %s", size_mb, wav_path.name)
    return wav_path, av_offset

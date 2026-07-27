# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""ffmpeg-based meeting recorder.

Records system audio (and optionally screen) to a file, then stops cleanly
when the user presses Ctrl+C or a duration limit is reached.

Output formats:
    audio-only  →  16 kHz mono PCM WAV (no re-extraction needed by the pipeline)
    with screen →  WebM (VP9 video + Opus audio, directly processable by vocolith)
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .devices import CaptureDevice

# ---------------------------------------------------------------------------
# Video encoder auto-detection
# ---------------------------------------------------------------------------

def _best_video_encoder() -> tuple[str, list[str]]:
    """Return (encoder_name, extra_opts) for the best available video encoder.

    Preference: GPU-accelerated (NVENC/VAAPI) > libx264 > mpeg4.
    Extra opts tune for low-motion screen-share content (meetings).
    """
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-encoders"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return "libx264", ["-preset", "ultrafast", "-crf", "28"]

    if "h264_nvenc" in out:
        return "h264_nvenc", ["-preset", "p1", "-rc", "vbr", "-cq", "28"]
    if "h264_vaapi" in out:
        return "h264_vaapi", ["-qp", "28"]
    if "libx264" in out:
        return "libx264", ["-preset", "ultrafast", "-crf", "28"]
    return "mpeg4", ["-q:v", "6"]


@dataclass
class RecordingResult:
    output_path: Path
    duration_seconds: float
    has_video: bool


class Recorder:
    """Manage a single ffmpeg capture process.

    Usage::

        rec = Recorder()
        rec.start(Path("/tmp/meeting.wav"), audio_device)
        # ... user presses Ctrl+C ...
        result = rec.stop()
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._output_path: Optional[Path] = None
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        output_path: Path,
        audio: CaptureDevice,
        screen: Optional[CaptureDevice] = None,
        max_duration_secs: Optional[float] = None,
    ) -> None:
        """Launch ffmpeg and begin recording.

        Args:
            output_path: Where to write the recording (.wav for audio-only,
                         .webm for screen+audio).
            audio: Audio capture device (from discover_devices()).
            screen: Optional screen capture device.  If None, audio-only.
            max_duration_secs: Hard time limit passed to ffmpeg via -t.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path = output_path
        cmd = self._build_cmd(output_path, audio, screen, max_duration_secs)
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._start_time = time.monotonic()

    def stop(self) -> RecordingResult:
        """Gracefully stop ffmpeg and return recording metadata.

        Sends 'q\\n' to ffmpeg stdin (clean shutdown: flushes all buffers and
        writes a valid file trailer before exiting).  Falls back to SIGTERM
        after 10 s if ffmpeg ignores the quit command.
        """
        if self._proc is None:
            raise RuntimeError("Recorder is not running")

        elapsed = time.monotonic() - self._start_time

        # 'q' is ffmpeg's interactive quit key — triggers a clean shutdown.
        try:
            if self._proc.stdin:
                self._proc.stdin.write(b"q\n")
                self._proc.stdin.flush()
        except BrokenPipeError:
            pass  # process already exited

        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

        assert self._output_path is not None
        return RecordingResult(
            output_path=self._output_path,
            duration_seconds=elapsed,
            has_video=self._output_path.suffix != ".wav",
        )

    def is_running(self) -> bool:
        """Return True while the ffmpeg process is alive."""
        return self._proc is not None and self._proc.poll() is None

    def elapsed_seconds(self) -> float:
        """Seconds since recording started (0 if not started)."""
        if not self._start_time:
            return 0.0
        return time.monotonic() - self._start_time

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_cmd(
        self,
        output_path: Path,
        audio: CaptureDevice,
        screen: Optional[CaptureDevice],
        max_duration_secs: Optional[float],
    ) -> list[str]:
        """Build the ffmpeg command list for this OS and device combination."""
        cmd: list[str] = ["ffmpeg", "-y"]

        if screen is not None:
            cmd = self._add_screen_input(cmd, screen)

        cmd = self._add_audio_input(cmd, audio)

        # ── Output mapping and encoding ────────────────────────────────────────
        if screen is not None:
            # screen = input 0 (video), audio = input 1
            cmd += ["-map", "0:v", "-map", "1:a"]
            # Cap width at 1920px (meeting content; stays under NVENC 4096px limit
            # and keeps file size sane).  scale='min(1920,iw)' with -2 ensures
            # width and height are both divisible by 2 (required by most encoders).
            cmd += ["-vf", "scale='min(1920,iw)':-2"]
            venc, venc_opts = _best_video_encoder()  # noqa: E501 defined at module level above Recorder
            cmd += ["-c:v", venc] + venc_opts
            cmd += ["-c:a", "pcm_s16le", "-ac", "1", "-ar", "16000"]  # lossless audio
        else:
            # Single audio input → 16 kHz mono WAV (pipeline reads this directly)
            cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]

        # -t as an OUTPUT duration limit (placed last, before the filename)
        # affects the encoder — stops all output after N seconds regardless
        # of how many inputs are running.
        if max_duration_secs is not None:
            cmd += ["-t", str(int(max_duration_secs))]

        cmd.append(str(output_path))
        return cmd

    @staticmethod
    def _add_screen_input(
        cmd: list[str],
        screen: CaptureDevice,
    ) -> list[str]:
        """Append screen capture arguments to *cmd* and return it.

        Correct ffmpeg order: -f <fmt> <format-specific-options...> -i <id>
        extra_args (e.g. -framerate, -video_size) are format-specific and
        must come AFTER -f, not before it.
        """
        if screen.fmt == "avfoundation":
            # macOS: screen-only device as "screen_idx:none"
            cmd += ["-f", "avfoundation"] + screen.extra_args + ["-i", f"{screen.id}:none"]
        else:
            cmd += ["-f", screen.fmt] + screen.extra_args + ["-i", screen.id]
        return cmd

    @staticmethod
    def _add_audio_input(
        cmd: list[str],
        audio: CaptureDevice,
    ) -> list[str]:
        """Append audio capture arguments to *cmd* and return it.

        extra_args (e.g. -loopback for WASAPI) are format-specific and
        must come AFTER -f.
        """
        if audio.fmt == "avfoundation":
            # macOS: audio-only device as "none:audio_idx"
            cmd += ["-f", "avfoundation"] + audio.extra_args + ["-i", f"none:{audio.id}"]
        else:
            cmd += ["-f", audio.fmt] + audio.extra_args + ["-i", audio.id]
        return cmd


# ---------------------------------------------------------------------------
# Duration string parser  (e.g.  "90m", "1h30m", "3600s", "01:30:00")
# ---------------------------------------------------------------------------

def parse_duration(s: str) -> float:
    """Parse a human duration string to seconds.

    Supported formats:
        "3600"        → raw seconds (numeric string)
        "3600s"       → seconds
        "90m"         → minutes
        "1h30m"       → hours + minutes
        "1h30m45s"    → hours + minutes + seconds
        "01:30:00"    → HH:MM:SS
        "90:00"       → MM:SS
    """
    s = s.strip().lower()

    # HH:MM:SS or MM:SS
    colon = re.fullmatch(r"(\d+):(\d+)(?::(\d+))?", s)
    if colon:
        a, b, c = colon.group(1), colon.group(2), colon.group(3)
        if c is not None:
            return int(a) * 3600 + int(b) * 60 + int(c)
        return int(a) * 60 + int(b)

    # Raw number → treat as seconds
    if re.fullmatch(r"\d+", s):
        return float(s)

    # Component form: 1h30m45s
    total = 0.0
    for value, unit in re.findall(r"(\d+)\s*([hms])", s):
        v = int(value)
        if unit == "h":
            total += v * 3600
        elif unit == "m":
            total += v * 60
        else:
            total += v

    if total == 0.0:
        raise ValueError(
            f"Cannot parse duration {s!r}. "
            "Use formats like '90m', '1h30m', '3600s', or '01:30:00'."
        )
    return total

# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""OS-agnostic capture device discovery.

Supports Linux (PulseAudio/PipeWire + X11/XWayland/Wayland),
macOS (AVFoundation), and Windows (WASAPI + DirectShow + GDI).

All device discovery is done via ffmpeg's built-in -list_devices mechanism
or platform tools (pactl on Linux), so no extra Python dependencies are needed.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CaptureDevice:
    """A single capture source understood by ffmpeg."""

    id: str              # value passed to ffmpeg -i (or the audio half of "screen:audio")
    name: str            # human-readable label shown to the user
    kind: str            # "audio_system" | "audio_mic" | "screen"
    fmt: str             # ffmpeg -f format string (pulse, avfoundation, wasapi, ...)
    is_loopback: bool = False          # True = captures system/playback audio
    extra_args: list[str] = field(default_factory=list)  # ffmpeg args inserted before -i

    def __str__(self) -> str:
        tag = " [loopback]" if self.is_loopback else ""
        return f"{self.kind:<14} {self.fmt:<14} {self.id!r:<32}  {self.name}{tag}"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def discover_devices() -> list[CaptureDevice]:
    """Discover available capture devices on the current OS.

    Returns audio (system loopback + mic) and screen devices.
    Never raises; returns a safe minimal list on any discovery failure.
    """
    os_name = platform.system()
    if os_name == "Linux":
        return _discover_linux()
    if os_name == "Darwin":
        return _discover_macos()
    if os_name == "Windows":
        return _discover_windows()
    raise RuntimeError(f"Unsupported OS for capture: {os_name!r}")


def get_default_audio(devices: list[CaptureDevice]) -> Optional[CaptureDevice]:
    """Return the best audio device: prefer system loopback, fall back to mic."""
    for d in devices:
        if d.is_loopback:
            return d
    for d in devices:
        if d.kind in ("audio_mic", "audio_system"):
            return d
    return None


def get_default_screen(devices: list[CaptureDevice]) -> Optional[CaptureDevice]:
    """Return the first screen device found, or None."""
    for d in devices:
        if d.kind == "screen":
            return d
    return None


# ---------------------------------------------------------------------------
# Linux
# ---------------------------------------------------------------------------

def _discover_linux() -> list[CaptureDevice]:
    devices: list[CaptureDevice] = []

    # ── Audio via PulseAudio / PipeWire (pactl) ───────────────────────────────
    #
    # PipeWire ships a PulseAudio-compatible layer so pactl works for both.
    # We query sources; *.monitor entries are loopback (capture what you hear).
    #
    audio_found = False
    try:
        out = subprocess.check_output(
            ["pactl", "list", "sources", "short"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            src = parts[1].strip()
            is_monitor = src.endswith(".monitor")
            devices.append(CaptureDevice(
                id=src,
                name=f"{src} ({'system audio' if is_monitor else 'microphone'})",
                kind="audio_system" if is_monitor else "audio_mic",
                fmt="pulse",
                is_loopback=is_monitor,
            ))
            audio_found = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    if not audio_found:
        # Fallback: try ffmpeg pulse device listing
        try:
            out = _ffmpeg_list_devices("pulse")
            for m in re.finditer(r'"([^"]+)"', out):
                src = m.group(1)
                is_mon = src.endswith(".monitor")
                devices.append(CaptureDevice(
                    id=src, name=src,
                    kind="audio_system" if is_mon else "audio_mic",
                    fmt="pulse", is_loopback=is_mon,
                ))
                audio_found = True
        except Exception:
            pass

    if not audio_found:
        # Last resort: hardcode well-known PulseAudio/PipeWire names
        devices += [
            CaptureDevice(
                id="default",
                name="default (microphone)",
                kind="audio_mic",
                fmt="pulse",
            ),
            CaptureDevice(
                id="default.monitor",
                name="default.monitor (system audio loopback)",
                kind="audio_system",
                fmt="pulse",
                is_loopback=True,
            ),
        ]

    # ── Screen capture ─────────────────────────────────────────────────────────
    #
    # Decision tree:
    #   $DISPLAY set            → x11grab per-monitor + full desktop fallback
    #   $WAYLAND_DISPLAY only   → pipewire screencast (portal dialog required)
    #
    display = os.environ.get("DISPLAY", "")
    wayland = os.environ.get("WAYLAND_DISPLAY", "")

    if display:
        devices.extend(_enumerate_x11_monitors(display))
    elif wayland:
        devices.append(CaptureDevice(
            id="0",
            name=f"Wayland display {wayland} (PipeWire — portal dialog required)",
            kind="screen",
            fmt="pipewire",
            extra_args=["-framerate", "5"],
        ))

    return devices


def _enumerate_x11_monitors(display: str) -> list[CaptureDevice]:
    """Return one CaptureDevice per physical monitor, plus one for the full desktop.

    Uses ``xrandr --listmonitors`` to get per-monitor geometry (resolution +
    offset within the combined X11 framebuffer).  x11grab captures a sub-region
    via ``-i DISPLAY+X,Y`` when an offset is specified.

    Monitor devices are listed first (primary monitor first), followed by the
    full combined desktop as the last entry.  ``get_default_screen()`` therefore
    returns the primary monitor by default.
    """
    monitors: list[CaptureDevice] = []
    full_w, full_h = 0, 0   # accumulated bounding box for the full-desktop fallback

    try:
        out = subprocess.check_output(
            ["xrandr", "--listmonitors"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        # Each monitor line looks like:
        #   0: +*DP-0 3440/800x1440/335+0+0  DP-0
        #              ^^^^     ^^^^  ^ ^
        #              px_w     px_h  x y
        _MON_RE = re.compile(
            r"^\s*(\d+):\s+\S+\s+(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)\s+(\S+)"
        )
        for line in out.splitlines():
            m = _MON_RE.match(line)
            if not m:
                continue
            idx   = int(m.group(1))
            pw    = int(m.group(2))   # pixel width
            ph    = int(m.group(3))   # pixel height
            ox    = int(m.group(4))   # X offset in framebuffer
            oy    = int(m.group(5))   # Y offset in framebuffer
            oname = m.group(6)        # output name e.g. DP-0

            # x11grab accepts :DISPLAY+X,Y — offset restricts the capture region.
            dev_id = f"{display}+{ox},{oy}" if (ox or oy) else display
            monitors.append(CaptureDevice(
                id=dev_id,
                name=f"Monitor {idx}: {oname} ({pw}x{ph})",
                kind="screen",
                fmt="x11grab",
                extra_args=["-framerate", "5", "-video_size", f"{pw}x{ph}"],
            ))
            full_w = max(full_w, ox + pw)
            full_h = max(full_h, oy + ph)
    except Exception:
        pass

    if len(monitors) <= 1:
        # Single monitor or discovery failed — just return the full desktop.
        size = _x11_desktop_size()
        return [CaptureDevice(
            id=display,
            name=f"X11 display {display} ({size})",
            kind="screen",
            fmt="x11grab",
            extra_args=["-framerate", "5", "-video_size", size],
        )]

    # "Full desktop" entry spanning all monitors.
    # Use explicit +0,0 offset so the ID ":0.0+0,0" is distinct from ":0.0"
    # (the primary monitor at that same offset) — lets --screen-device tell them apart.
    monitors.append(CaptureDevice(
        id=f"{display}+0,0",
        name=f"Full desktop ({full_w}x{full_h}, all monitors)",
        kind="screen",
        fmt="x11grab",
        extra_args=["-framerate", "5", "-video_size", f"{full_w}x{full_h}"],
    ))
    return monitors


def _x11_desktop_size() -> str:
    """Return 'WxH' for the combined X11 desktop, defaulting to 1920x1080."""
    for cmd in (["xdpyinfo"], ["xrandr", "--current"]):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            m = re.search(r"(\d{3,5})x(\d{3,5})", out)
            if m:
                return f"{m.group(1)}x{m.group(2)}"
        except Exception:
            continue
    return "1920x1080"


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------

def _discover_macos() -> list[CaptureDevice]:
    devices: list[CaptureDevice] = []

    # avfoundation -list_devices returns non-zero but writes to stderr.
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", "dummy"],
            text=True,
            capture_output=True,
        )
        out = result.stderr  # ffmpeg writes device list to stderr
    except FileNotFoundError:
        return _macos_fallback()

    in_video = False
    in_audio = False

    for line in out.splitlines():
        if "AVFoundation video devices" in line:
            in_video, in_audio = True, False
            continue
        if "AVFoundation audio devices" in line:
            in_audio, in_video = True, False
            continue

        m = re.search(r"\[(\d+)\]\s+(.+)", line)
        if not m:
            continue
        idx, name = m.group(1), m.group(2).strip()

        if in_video:
            if any(kw in name.lower() for kw in ("capture screen", "desktop", "display")):
                devices.append(CaptureDevice(
                    id=idx,
                    name=name,
                    kind="screen",
                    fmt="avfoundation",
                    extra_args=["-framerate", "5"],
                ))
        elif in_audio:
            is_loopback = any(
                kw in name.lower()
                for kw in ("blackhole", "loopback", "soundflower", "stereo mix", "virtual")
            )
            devices.append(CaptureDevice(
                id=idx,
                name=name,
                kind="audio_system" if is_loopback else "audio_mic",
                fmt="avfoundation",
                is_loopback=is_loopback,
            ))

    return devices if devices else _macos_fallback()


def _macos_fallback() -> list[CaptureDevice]:
    return [
        CaptureDevice(
            id="0",
            name="Built-in Microphone",
            kind="audio_mic",
            fmt="avfoundation",
        ),
        CaptureDevice(
            id="1",
            name="Capture screen 0",
            kind="screen",
            fmt="avfoundation",
            extra_args=["-framerate", "5"],
        ),
    ]


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _discover_windows() -> list[CaptureDevice]:
    devices: list[CaptureDevice] = []
    flags = 0x08000000  # CREATE_NO_WINDOW — suppress console flicker

    # ── WASAPI (preferred: loopback capture is built-in) ─────────────────────
    try:
        out = _ffmpeg_list_devices("wasapi", creationflags=flags)
        for m in re.finditer(r'"([^"]+)"', out):
            name = m.group(1)
            devices.append(CaptureDevice(
                id=name, name=f"{name} (WASAPI)",
                kind="audio_mic", fmt="wasapi",
            ))
    except Exception:
        pass

    # Add a dedicated WASAPI loopback entry (captures all system audio)
    # This uses -loopback 1 flag which works regardless of device name.
    devices.insert(0, CaptureDevice(
        id="default",
        name="System audio loopback (WASAPI)",
        kind="audio_system",
        fmt="wasapi",
        is_loopback=True,
        extra_args=["-loopback", "1"],
    ))

    # ── DirectShow fallback ───────────────────────────────────────────────────
    try:
        out = _ffmpeg_list_devices("dshow", creationflags=flags)
        for line in out.splitlines():
            if "(audio)" not in line.lower():
                continue
            m = re.search(r'"([^"]+)"', line)
            if not m:
                continue
            name = m.group(1)
            is_loopback = any(
                kw in name.lower()
                for kw in ("stereo mix", "what u hear", "wave out", "loopback")
            )
            if not any(d.name.startswith(name) for d in devices):
                devices.append(CaptureDevice(
                    id=f"audio={name}", name=name,
                    kind="audio_system" if is_loopback else "audio_mic",
                    fmt="dshow", is_loopback=is_loopback,
                ))
    except Exception:
        pass

    # ── Screen via GDI grab ───────────────────────────────────────────────────
    devices.append(CaptureDevice(
        id="desktop",
        name="Desktop (GDI grab)",
        kind="screen",
        fmt="gdigrab",
        extra_args=["-framerate", "5"],
    ))

    return devices


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ffmpeg_list_devices(fmt: str, creationflags: int = 0) -> str:
    """Run 'ffmpeg -f <fmt> -list_devices true -i dummy' and return combined output."""
    kwargs: dict = {"text": True, "capture_output": True}
    if creationflags:
        kwargs["creationflags"] = creationflags
    result = subprocess.run(
        ["ffmpeg", "-f", fmt, "-list_devices", "true", "-i", "dummy"],
        **kwargs,
    )
    return result.stdout + result.stderr

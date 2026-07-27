# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""
Parser for Microsoft Teams' generated meeting transcript (.vtt export).

Teams' transcript export is *not* standard WebVTT with ``<v Name>`` voice
tags. Cue blocks look like::

    WEBVTT

    1 "Nishanth Menon" (833673728)
    00:00:03.127 --> 00:00:21.630
    Alright, so today I was hoping to cover through the basic agenda here...

    2 "Srikanth Eswaran" (531405056)
    00:00:21.630 --> 00:00:42.064
    ...

i.e. cue index + quoted speaker name + parenthesised numeric participant id
on one line, a timestamp range on the next, then the caption text.

A fallback for standard ``<v Speaker Name>text</v>`` cues (older Teams/Zoom
exports) is also supported in case a different variant is fed in.
"""
from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_HEADER_RE = re.compile(r'^\d+\s+"([^"]+)"\s+\(\d+\)\s*$')
_TIMESTAMP_RE = re.compile(
    r'^(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})'
)
_VOICE_TAG_RE = re.compile(r'<v\s+([^>]+)>(.*?)(?:</v>)?$', re.IGNORECASE)


@dataclass
class ReferenceEntry:
    """One speaker turn from a reference transcript (e.g. Teams .vtt export)."""
    start: float
    end: float
    speaker: str
    text: str


def _parse_ts(ts: str) -> float:
    """Convert 'HH:MM:SS.mmm' to seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_teams_vtt(path: Path) -> list[ReferenceEntry]:
    """
    Parse a Microsoft Teams generated .vtt transcript into ReferenceEntry list.

    Never raises on a malformed cue block — such blocks are skipped and
    logged at debug level so one bad cue doesn't discard the whole file.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())

    entries: list[ReferenceEntry] = []
    for block in blocks:
        lines = [l for l in block.splitlines() if l.strip()]
        if not lines or lines[0].strip().upper() == "WEBVTT":
            continue

        entry = _parse_teams_block(lines)
        if entry is None:
            entry = _parse_voice_tag_block(lines)
        if entry is None:
            log.debug("Skipping unparseable vtt cue block: %r", block[:120])
            continue
        entries.append(entry)

    entries.sort(key=lambda e: e.start)
    log.info("Parsed %d reference transcript entries from %s", len(entries), path)
    return entries


def _parse_teams_block(lines: list[str]) -> ReferenceEntry | None:
    """Parse a Teams-style block: header line, timestamp line, text line(s)."""
    idx = 0
    header_match = _HEADER_RE.match(lines[idx].strip())
    if not header_match:
        return None
    speaker = header_match.group(1).strip()
    idx += 1

    if idx >= len(lines):
        return None
    ts_match = _TIMESTAMP_RE.match(lines[idx].strip())
    if not ts_match:
        return None
    idx += 1

    try:
        start = _parse_ts(ts_match.group(1))
        end = _parse_ts(ts_match.group(2))
    except ValueError:
        return None

    text = " ".join(l.strip() for l in lines[idx:]).strip()
    if not text:
        return None

    return ReferenceEntry(start=start, end=end, speaker=speaker, text=text)


def _parse_voice_tag_block(lines: list[str]) -> ReferenceEntry | None:
    """Fallback: standard WebVTT cue with an optional numeric index line,
    a timestamp line, and a <v Speaker>text</v> caption line."""
    idx = 0
    if idx < len(lines) and re.fullmatch(r"\d+", lines[idx].strip()):
        idx += 1  # optional cue index line

    if idx >= len(lines):
        return None
    ts_match = _TIMESTAMP_RE.match(lines[idx].strip())
    if not ts_match:
        return None
    idx += 1

    try:
        start = _parse_ts(ts_match.group(1))
        end = _parse_ts(ts_match.group(2))
    except ValueError:
        return None

    caption = " ".join(l.strip() for l in lines[idx:]).strip()
    voice_match = _VOICE_TAG_RE.match(caption)
    if not voice_match:
        return None

    speaker = voice_match.group(1).strip()
    caption_text = re.sub(r"</?v[^>]*>", "", voice_match.group(2)).strip()
    if not caption_text:
        return None

    return ReferenceEntry(start=start, end=end, speaker=speaker, text=caption_text)

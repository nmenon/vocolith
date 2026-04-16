# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""
Load an arbitrary transcript file into a DiarizedTranscript.

Supported formats (auto-detected by content):

1. Vocolith transcript.md  — produced by ``vocolith process``
   Line format:  [MM:SS] **Speaker Name**: text

2. WebVTT (.vtt) — Webex, Teams, Zoom caption exports
   Standard WebVTT with optional speaker-labelled cue identifiers:
     N "Speaker Name" (ID)
     HH:MM:SS.mmm --> HH:MM:SS.mmm
     text

   Also handles plain WebVTT (no speaker labels) and
   VTT with inline <v Speaker>text</v> tags.

3. Plain text / any other format
   The entire file is treated as a single unnamed speaker block so the
   LLM can still extract notes from it.
"""
from __future__ import annotations
import re
from pathlib import Path

from vocolith.models.transcript import DiarizedTranscript, TranscriptSegment

# Vocolith transcript.md:  [MM:SS] **Speaker**: text
_VOCOLITH_LINE = re.compile(
    r"^\[(\d{1,2}:\d{2})\]\s+\*\*([^*]+)\*\*:\s+(.+)$"
)

# VTT timestamp line:  HH:MM:SS.mmm --> HH:MM:SS.mmm  (or MM:SS.mmm)
_VTT_TIMESTAMP = re.compile(
    r"^(\d{1,2}:\d{2}:\d{2}[\.,]\d+|\d{2}:\d{2}[\.,]\d+)"
    r"\s+-->\s+"
    r"(\d{1,2}:\d{2}:\d{2}[\.,]\d+|\d{2}:\d{2}[\.,]\d+)"
)

# Webex/Teams cue ID line:  N "Speaker Name" (numeric-id)
_VTT_SPEAKER_CUE = re.compile(r'^[\d]+\s+"([^"]+)"\s+\(\d+\)\s*$')

# Inline voice tag:  <v Speaker Name>text</v>
_VTT_VOICE_TAG = re.compile(r"<v\s+([^>]+)>(.*?)</v>", re.DOTALL)

# Strip any remaining VTT tags (<c>, <00:00:00.000>, etc.)
_VTT_TAGS = re.compile(r"<[^>]+>")


def _vtt_ts_to_seconds(ts: str) -> float:
    """Convert VTT/SRT timestamp (HH:MM:SS.mmm or MM:SS.mmm) to float seconds."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    m, s = parts
    return float(m) * 60 + float(s)


def load_transcript_file(path: Path) -> DiarizedTranscript:
    """
    Parse a transcript file into a DiarizedTranscript.

    Format is auto-detected; falls back to plain-text wrapping when no
    known format is recognised.
    """
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    # Try VTT first when extension matches or file starts with WEBVTT
    if suffix == ".vtt" or text.lstrip().startswith("WEBVTT"):
        segments = _parse_vtt(text)
        if segments:
            return _make_transcript(segments)

    # Try vocolith transcript.md format
    segments = _parse_vocolith_format(text)
    if segments:
        return _make_transcript(segments)

    # Fall back: entire file as one unnamed segment
    return DiarizedTranscript(
        segments=[
            TranscriptSegment(
                segment_id=0,
                start=0.0,
                end=0.0,
                text=text.strip(),
                speaker_name=None,
            )
        ],
        duration_seconds=0.0,
        speakers_detected=0,
    )


def _make_transcript(segments: list[TranscriptSegment]) -> DiarizedTranscript:
    duration = max((s.end for s in segments), default=0.0)
    speakers = len({s.speaker_name for s in segments if s.speaker_name})
    return DiarizedTranscript(
        segments=segments,
        duration_seconds=duration,
        speakers_detected=speakers,
    )


def _parse_vtt(text: str) -> list[TranscriptSegment]:
    """
    Parse WebVTT content into TranscriptSegments.

    Handles three common VTT flavours:
    - Webex: cue ID line is  N "Speaker Name" (numeric-id)
    - Teams/Zoom: <v Speaker>text</v> inline voice tags
    - Plain VTT: no speaker info — speaker_name left None
    """
    segments: list[TranscriptSegment] = []
    lines = text.splitlines()
    i = 0

    current_speaker: str | None = None
    current_start: float = 0.0
    current_end: float = 0.0

    while i < len(lines):
        line = lines[i].strip()

        # Skip WEBVTT header, NOTE blocks, STYLE blocks, blank lines
        if not line or line.startswith("WEBVTT") or line.startswith("NOTE") \
                or line.startswith("STYLE") or line.startswith("REGION"):
            i += 1
            continue

        # Webex-style cue identifier:  N "Speaker Name" (id)
        m_spk = _VTT_SPEAKER_CUE.match(line)
        if m_spk:
            current_speaker = m_spk.group(1).strip()
            i += 1
            continue

        # Timestamp line
        m_ts = _VTT_TIMESTAMP.match(line)
        if m_ts:
            current_start = _vtt_ts_to_seconds(m_ts.group(1))
            current_end = _vtt_ts_to_seconds(m_ts.group(2))
            # Collect cue text lines until blank line
            i += 1
            cue_lines: list[str] = []
            while i < len(lines) and lines[i].strip():
                cue_lines.append(lines[i])
                i += 1
            cue_text = " ".join(cue_lines)

            # Check for inline <v Speaker> tags — a single cue may contain
            # multiple voice spans (e.g. two people in the same timestamp block).
            # Emit one segment per voice span; fall back to current_speaker if none.
            voice_matches = _VTT_VOICE_TAG.findall(cue_text)
            if voice_matches:
                for spk, span_text in voice_matches:
                    cleaned = _VTT_TAGS.sub("", span_text).strip()
                    if cleaned:
                        segments.append(TranscriptSegment(
                            segment_id=len(segments),
                            start=current_start,
                            end=current_end,
                            text=cleaned,
                            speaker_name=spk.strip(),
                        ))
                continue  # skip the single-segment fallback below

            # No voice tags — use cue-level speaker and strip all tags
            speaker = current_speaker
            cue_text = _VTT_TAGS.sub("", cue_text).strip()

            if cue_text:
                segments.append(TranscriptSegment(
                    segment_id=len(segments),
                    start=current_start,
                    end=current_end,
                    text=cue_text,
                    speaker_name=speaker,
                ))
            # Reset speaker after cue (Webex sets it fresh each cue)
            current_speaker = None
            continue

        # Plain numeric cue identifier (no speaker info) — skip
        if re.match(r"^\d+$", line):
            i += 1
            continue

        i += 1

    return segments


def _parse_vocolith_format(text: str) -> list[TranscriptSegment]:
    """
    Parse vocolith transcript.md lines into segments.
    Line format:  [MM:SS] **Speaker Name**: text
    """
    segments: list[TranscriptSegment] = []
    for line in text.splitlines():
        m = _VOCOLITH_LINE.match(line.strip())
        if not m:
            continue
        ts_str, speaker, content = m.groups()
        mm, ss = ts_str.split(":")
        start = float(mm) * 60 + float(ss)
        segments.append(
            TranscriptSegment(
                segment_id=len(segments),
                start=start,
                end=start,   # patched to next segment's start below
                text=content.strip(),
                speaker_name=speaker.strip(),
            )
        )
    # Patch end times: each segment ends where the next begins.
    # The last segment gets a 1s nominal duration (no following segment to anchor on).
    for i in range(len(segments) - 1):
        segments[i] = TranscriptSegment(
            segment_id=segments[i].segment_id,
            start=segments[i].start,
            end=segments[i + 1].start,
            text=segments[i].text,
            speaker_name=segments[i].speaker_name,
        )
    if segments:
        last = segments[-1]
        segments[-1] = TranscriptSegment(
            segment_id=last.segment_id,
            start=last.start,
            end=last.start + 1.0,
            text=last.text,
            speaker_name=last.speaker_name,
        )
    return segments

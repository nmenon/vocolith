# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Transcript formatting and OCR-guided terminology correction."""
from __future__ import annotations
import difflib
import re
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vocolith.models.transcript import TranscriptSegment

log = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Convert float seconds to [HH:MM:SS] string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def render_transcript_md(segments: list[TranscriptSegment],
                          meta: dict) -> str:
    """Render a diarized transcript as a Markdown string."""
    lines = [
        "# Meeting Transcript",
        f"**File**: {meta.get('filename', 'unknown')}",
        f"**Date**: {meta.get('date', 'unknown')}",
        f"**Duration**: {format_timestamp(meta.get('duration', 0))}",
        f"**Language**: {meta.get('language', 'unknown')}",
    ]

    speakers = sorted({s.speaker_name or s.speaker_label or 'Unknown'
                        for s in segments if s.speaker_name or s.speaker_label})
    # Map speaker_name → resolution method so the lookup actually hits
    resolution_map: dict[str, str] = {}
    for s in segments:
        name = s.speaker_name or s.speaker_label
        if name and s.resolution_method:
            resolution_map.setdefault(name, s.resolution_method)

    # Methods that add no useful information for a reader
    _NOISE_METHODS = {"fallback", "unknown", "unresolved", None, ""}

    speaker_labels = []
    for sp in speakers:
        method = resolution_map.get(sp)
        if method and method.split("(")[0] not in _NOISE_METHODS:
            speaker_labels.append(f"{sp} ({method})")
        else:
            speaker_labels.append(sp)
    if speaker_labels:
        lines.append(f"**Speakers**: {', '.join(speaker_labels)}")

    lines += ["", "---", ""]

    prev_speaker = None
    for seg in segments:
        name = seg.speaker_name or seg.speaker_label or "Unknown"
        if name != prev_speaker:
            if prev_speaker is not None:
                lines.append("")
            prev_speaker = name
        ts = format_timestamp(seg.start)
        lines.append(f"[{ts}] **{name}**: {seg.text.strip()}")

    return "\n".join(lines)


def correct_transcript_terminology(
    segments: list[TranscriptSegment],
    ocr_vocabulary: list[str],
) -> list[TranscriptSegment]:
    """
    Post-correction pass: use OCR-extracted vocabulary to fix Whisper
    mangling of domain-specific terms and acronyms.

    Only corrects tokens where a close OCR match exists (cutoff=0.82).
    Case-preserving: uses the exact casing from OCR.
    """
    if not ocr_vocabulary:
        return segments

    vocab_lower = [t.lower() for t in ocr_vocabulary]
    # Build O(1) lookup: lowercase form → original casing
    vocab_lookup = {v_low: orig for v_low, orig in zip(vocab_lower, ocr_vocabulary)}

    corrected_count = 0
    for seg in segments:
        words = seg.text.split()
        new_words = []
        for word in words:
            # Strip punctuation for matching, reattach after
            stripped = word.rstrip(".,;:!?\"'")
            punct = word[len(stripped):]
            matches = difflib.get_close_matches(
                stripped.lower(), vocab_lower, n=1, cutoff=0.82
            )
            if matches:
                original_term = vocab_lookup[matches[0]]
                new_words.append(original_term + punct)
                if original_term != stripped:
                    corrected_count += 1
            else:
                new_words.append(word)
        seg.text = " ".join(new_words)

    if corrected_count:
        log.debug("Terminology correction: %d substitutions applied", corrected_count)
    return segments


def clean_ocr_name(raw: str) -> str | None:
    """
    Heuristic filter for OCR text: returns cleaned name string if it looks
    like a person's name, or None if it should be discarded.

    Rejects:
    - URLs / web strings (://, .com, ?, =, &, @)
    - Very long strings (likely slide text, not names)
    - All-digits or all-punctuation
    - Short all-caps UI labels
    - Strings with excessive special characters
    """
    text = raw.strip()
    if len(text) < 2 or len(text) > 40:
        return None
    # Reject URL/web patterns
    url_indicators = ("://", ".com", ".itg.", "?", "=", "&_", "http", "zoom.us", "@")
    if any(ind in text for ind in url_indicators):
        return None
    # Reject all-digits or all-punctuation
    if re.fullmatch(r'[\d\W]+', text):
        return None
    # Reject very short all-caps strings that are UI labels (OK, HD etc.)
    if len(text) <= 3 and text.isupper():
        return None
    # Reject strings with too many special chars (> 20% non-alphanum/space/hyphen/paren)
    special = sum(1 for c in text if not (c.isalnum() or c in ' ()-_.'))
    if special > len(text) * 0.20:
        return None
    # Must contain at least one alphabetic word of length >= 2
    if not any(len(w) >= 2 and any(c.isalpha() for c in w) for w in text.split()):
        return None
    return text

# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Transcript and diarization data models."""
from __future__ import annotations
from pydantic import BaseModel, Field


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    score: float = 1.0   # confidence 0-1


class TranscriptSegment(BaseModel):
    segment_id: int
    start: float            # seconds
    end: float
    text: str
    words: list[WordTimestamp] = Field(default_factory=list)
    speaker_label: str | None = None    # raw diarization label e.g. "SPEAKER_00"
    speaker_name: str | None = None     # resolved display name
    resolution_method: str | None = None  # "ocr"|"voice"|"face"|"fallback"


class DiarizedTranscript(BaseModel):
    segments: list[TranscriptSegment]
    language: str = "en"
    duration_seconds: float = 0.0
    speakers_detected: int = 0

    def unique_speakers(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for seg in self.segments:
            key = seg.speaker_name or seg.speaker_label or "Unknown"
            if key not in seen:
                seen.add(key)
                result.append(key)
        return result

    def text_for_speaker(self, label: str) -> str:
        """Return concatenated text for a given speaker label or name."""
        return " ".join(
            s.text for s in self.segments
            if s.speaker_label == label or s.speaker_name == label
        )

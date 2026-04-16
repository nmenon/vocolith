# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Meeting session and video frame data models."""
from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid


class VideoFrame(BaseModel):
    timestamp_s: float
    frame_index: int
    source_region: str    # "top_strip" | "bottom_strip" | "full"
    # Raw numpy array NOT stored in model — kept in pipeline context only


class FaceDetection(BaseModel):
    frame_timestamp_s: float
    face_location: tuple[int, int, int, int]   # top, right, bottom, left (pixels)
    face_encoding: list[float]                  # 128-dim from face_recognition
    matched_speaker_id: str | None = None


class MeetingSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    video_path: str
    video_filename: str
    processed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))  # noqa: F821
    duration_seconds: float = 0.0
    language: str = "en"
    speaker_ids: list[str] = Field(default_factory=list)
    output_dir: str = ""

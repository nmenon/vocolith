# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Speaker profile and embedding data models."""
from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SpeakerProfile(BaseModel):
    speaker_id: str = Field(default_factory=_new_id)
    display_name: str
    aliases: list[str] = Field(default_factory=list)
    voice_embedding_ids: list[str] = Field(default_factory=list)
    face_embedding_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    last_seen_at: datetime = Field(default_factory=_now)
    meeting_count: int = 0

    def touch(self) -> None:
        self.last_seen_at = _now()
        self.meeting_count += 1


class VoiceEmbedding(BaseModel):
    embedding_id: str = Field(default_factory=_new_id)
    speaker_id: str
    vector: list[float]        # 256-dim d-vector from resemblyzer
    source_meeting_id: str
    created_at: datetime = Field(default_factory=_now)


class FaceEmbedding(BaseModel):
    embedding_id: str = Field(default_factory=_new_id)
    speaker_id: str
    vector: list[float]        # 128-dim encoding from face_recognition
    source_meeting_id: str
    created_at: datetime = Field(default_factory=_now)

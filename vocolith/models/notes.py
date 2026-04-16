# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Meeting notes data models."""
from __future__ import annotations
from datetime import date as DateType
from pydantic import BaseModel, Field


class ActionItem(BaseModel):
    description: str
    assignee: str | None = None      # only set if explicitly assigned in transcript
    due_date: str | None = None      # ISO date string — only if a date was stated
    priority: str | None = None      # "high"|"medium"|"low" — only if stated; null otherwise
    source_quote: str | None = None  # verbatim excerpt from transcript supporting this item
    timestamp: str | None = None     # approximate timestamp [MM:SS] where this was said


class Decision(BaseModel):
    description: str
    decided_by: str | None = None
    context: str | None = None
    source_quote: str | None = None  # verbatim excerpt from transcript supporting this decision
    timestamp: str | None = None     # approximate timestamp [MM:SS] where this was decided


class MeetingNotes(BaseModel):
    session_id: str = ""
    title: str = "Meeting Notes"
    meeting_date: DateType = Field(default_factory=DateType.today)
    duration_minutes: int = 0
    attendees: list[str] = Field(default_factory=list)
    summary: str = ""
    agenda_items: list[str] = Field(default_factory=list)
    key_topics: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    # Per-template structured data (for standup, 1:1 etc.)
    extra: dict = Field(default_factory=dict)

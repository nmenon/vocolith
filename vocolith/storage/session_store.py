# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""CRUD operations for MeetingSession records."""
from __future__ import annotations
import logging
import sqlite3
from datetime import datetime

from vocolith.models.meeting import MeetingSession

log = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, session: MeetingSession) -> None:
        # INSERT OR IGNORE + UPDATE avoids the INSERT OR REPLACE delete-then-insert
        # pattern which would violate the FK constraint on session_speakers when
        # PRAGMA foreign_keys=ON is active and the session already has linked rows.
        self._conn.execute(
            """INSERT OR IGNORE INTO meeting_sessions
               (session_id, video_filename, video_path, processed_at,
                duration_secs, language, output_dir)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session.session_id,
                session.video_filename,
                session.video_path,
                session.processed_at.isoformat(),
                session.duration_seconds,
                session.language,
                session.output_dir,
            ),
        )
        self._conn.execute(
            """UPDATE meeting_sessions
               SET video_filename=?, video_path=?, processed_at=?,
                   duration_secs=?, language=?, output_dir=?
               WHERE session_id=?""",
            (
                session.video_filename,
                session.video_path,
                session.processed_at.isoformat(),
                session.duration_seconds,
                session.language,
                session.output_dir,
                session.session_id,
            ),
        )
        self._conn.commit()

    def link_speaker(self, session_id: str, speaker_id: str,
                      label_used: str, resolution_method: str = "fallback") -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO session_speakers
               (session_id, speaker_id, label_used, resolution_method)
               VALUES (?, ?, ?, ?)""",
            (session_id, speaker_id, label_used, resolution_method),
        )
        self._conn.commit()

    def get(self, session_id: str) -> MeetingSession | None:
        row = self._conn.execute(
            "SELECT * FROM meeting_sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not row:
            return None
        try:
            processed_at = datetime.fromisoformat(row["processed_at"])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Malformed datetime in meeting_sessions row {row['session_id']!r}: {exc}"
            ) from exc
        return MeetingSession(
            session_id=row["session_id"],
            video_filename=row["video_filename"],
            video_path=row["video_path"],
            processed_at=processed_at,
            duration_seconds=row["duration_secs"],
            language=row["language"],
            output_dir=row["output_dir"],
        )

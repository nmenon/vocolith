# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""CRUD operations for SpeakerProfile and aliases in SQLite."""
from __future__ import annotations
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from vocolith.models.speaker import SpeakerProfile

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SpeakerStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def get(self, speaker_id: str) -> SpeakerProfile | None:
        row = self._conn.execute(
            "SELECT * FROM speakers WHERE speaker_id=?", (speaker_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_profile(row)

    def find_by_name(self, display_name: str) -> SpeakerProfile | None:
        row = self._conn.execute(
            "SELECT * FROM speakers WHERE LOWER(display_name)=LOWER(?)",
            (display_name,)
        ).fetchone()
        return self._row_to_profile(row) if row else None

    def find_by_alias(self, alias: str) -> SpeakerProfile | None:
        """Look up speaker by OCR-extracted name or previous alias."""
        row = self._conn.execute(
            """SELECT s.* FROM speakers s
               JOIN speaker_aliases a ON a.speaker_id = s.speaker_id
               WHERE LOWER(a.alias) = LOWER(?)""",
            (alias,)
        ).fetchone()
        return self._row_to_profile(row) if row else None

    def list_all(self) -> list[SpeakerProfile]:
        rows = self._conn.execute(
            "SELECT * FROM speakers ORDER BY last_seen_at DESC"
        ).fetchall()
        return [self._row_to_profile(r) for r in rows]

    def save(self, profile: SpeakerProfile) -> None:
        # INSERT OR IGNORE + UPDATE avoids the INSERT OR REPLACE cascade-delete
        # behaviour: INSERT OR REPLACE deletes the existing row first, triggering
        # ON DELETE CASCADE and wiping all aliases, embeddings, and samples.
        self._conn.execute(
            """INSERT OR IGNORE INTO speakers
               (speaker_id, display_name, created_at, last_seen_at, meeting_count)
               VALUES (?, ?, ?, ?, ?)""",
            (
                profile.speaker_id,
                profile.display_name,
                profile.created_at.isoformat(),
                profile.last_seen_at.isoformat(),
                profile.meeting_count,
            ),
        )
        self._conn.execute(
            """UPDATE speakers
               SET display_name=?, last_seen_at=?, meeting_count=?
               WHERE speaker_id=?""",
            (
                profile.display_name,
                profile.last_seen_at.isoformat(),
                profile.meeting_count,
                profile.speaker_id,
            ),
        )
        # Add aliases without touching existing ones
        for alias in profile.aliases:
            self._conn.execute(
                """INSERT OR IGNORE INTO speaker_aliases
                   (alias_id, speaker_id, alias, source) VALUES (?, ?, ?, 'profile')""",
                (str(uuid.uuid4()), profile.speaker_id, alias),
            )
        self._conn.commit()

    def add_alias(self, speaker_id: str, alias: str, source: str = "ocr") -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO speaker_aliases
               (alias_id, speaker_id, alias, source) VALUES (?, ?, ?, ?)""",
            (str(uuid.uuid4()), speaker_id, alias, source),
        )
        self._conn.commit()

    def touch(self, speaker_id: str) -> None:
        """Update last_seen_at and increment meeting_count."""
        self._conn.execute(
            """UPDATE speakers
               SET last_seen_at=?, meeting_count=meeting_count+1
               WHERE speaker_id=?""",
            (_now_iso(), speaker_id),
        )
        self._conn.commit()

    def save_sample(self, speaker_id: str, session_id: str,
                     audio_bytes: bytes, duration_s: float) -> str:
        """Store a WAV audio sample for a speaker. Returns the sample_id."""
        sample_id = str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO speaker_samples
               (sample_id, speaker_id, session_id, audio_data, duration_s, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (sample_id, speaker_id, session_id,
             audio_bytes, duration_s, _now_iso()),
        )
        self._conn.commit()
        return sample_id

    def get_sample(self, speaker_id: str) -> bytes | None:
        """Return the most recent WAV audio sample for a speaker, or None."""
        row = self._conn.execute(
            """SELECT audio_data FROM speaker_samples
               WHERE speaker_id=? ORDER BY created_at DESC LIMIT 1""",
            (speaker_id,),
        ).fetchone()
        return row["audio_data"] if row else None

    def delete(self, speaker_id: str) -> None:
        self._conn.execute(
            "DELETE FROM speakers WHERE speaker_id=?", (speaker_id,)
        )
        self._conn.commit()

    _EMBEDDING_TABLES = frozenset({"voice_embeddings", "face_embeddings"})

    def record_embedding(self, table: str, embedding_id: str,
                          speaker_id: str, meeting_id: str) -> None:
        """Record a voice or face embedding in the metadata table.
        Uses a fixed allowlist to prevent SQL injection via the table name.
        """
        if table not in self._EMBEDDING_TABLES:
            raise ValueError(f"Invalid embedding table: {table!r}")
        # Safe: table name validated against a fixed allowlist before interpolation
        sql = {
            "voice_embeddings": (
                "INSERT OR IGNORE INTO voice_embeddings "
                "(embedding_id, speaker_id, source_meeting, created_at) "
                "VALUES (?, ?, ?, ?)"
            ),
            "face_embeddings": (
                "INSERT OR IGNORE INTO face_embeddings "
                "(embedding_id, speaker_id, source_meeting, created_at) "
                "VALUES (?, ?, ?, ?)"
            ),
        }[table]
        self._conn.execute(sql, (embedding_id, speaker_id, meeting_id, _now_iso()))
        self._conn.commit()

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> SpeakerProfile:
        try:
            created_at = datetime.fromisoformat(row["created_at"])
            last_seen_at = datetime.fromisoformat(row["last_seen_at"])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Malformed datetime in speakers row {row['speaker_id']!r}: {exc}"
            ) from exc
        return SpeakerProfile(
            speaker_id=row["speaker_id"],
            display_name=row["display_name"],
            created_at=created_at,
            last_seen_at=last_seen_at,
            meeting_count=row["meeting_count"],
        )

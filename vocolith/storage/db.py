# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""SQLite schema, migrations, and connection management.

Stores speaker metadata only — embedding vectors live in ChromaDB.
"""
from __future__ import annotations
import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA_VERSION = 2

_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS speakers (
    speaker_id   TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    meeting_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS speaker_aliases (
    alias_id    TEXT PRIMARY KEY,
    speaker_id  TEXT NOT NULL REFERENCES speakers(speaker_id) ON DELETE CASCADE,
    alias       TEXT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'user',
    UNIQUE(speaker_id, alias)
);

CREATE TABLE IF NOT EXISTS voice_embeddings (
    embedding_id   TEXT PRIMARY KEY,
    speaker_id     TEXT NOT NULL REFERENCES speakers(speaker_id) ON DELETE CASCADE,
    source_meeting TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id   TEXT PRIMARY KEY,
    speaker_id     TEXT NOT NULL REFERENCES speakers(speaker_id) ON DELETE CASCADE,
    source_meeting TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meeting_sessions (
    session_id      TEXT PRIMARY KEY,
    video_filename  TEXT NOT NULL,
    video_path      TEXT NOT NULL,
    processed_at    TEXT NOT NULL,
    duration_secs   REAL NOT NULL DEFAULT 0,
    language        TEXT NOT NULL DEFAULT 'en',
    output_dir      TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS session_speakers (
    session_id        TEXT NOT NULL REFERENCES meeting_sessions(session_id),
    speaker_id        TEXT NOT NULL REFERENCES speakers(speaker_id),
    label_used        TEXT NOT NULL,
    resolution_method TEXT NOT NULL DEFAULT 'fallback',
    PRIMARY KEY (session_id, speaker_id)
);

-- Audio samples stored directly in the database so profiles are fully
-- self-contained.  No external files required for playback or verification.
-- Cascade-deletes when the speaker profile is removed.
CREATE TABLE IF NOT EXISTS speaker_samples (
    sample_id    TEXT PRIMARY KEY,
    speaker_id   TEXT NOT NULL REFERENCES speakers(speaker_id) ON DELETE CASCADE,
    session_id   TEXT NOT NULL,
    audio_data   BLOB NOT NULL,   -- raw WAV bytes (16kHz mono PCM, ≤10 s)
    duration_s   REAL NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alias_text       ON speaker_aliases(alias);
CREATE INDEX IF NOT EXISTS idx_voice_speaker    ON voice_embeddings(speaker_id);
CREATE INDEX IF NOT EXISTS idx_face_speaker     ON face_embeddings(speaker_id);
CREATE INDEX IF NOT EXISTS idx_session_speakers ON session_speakers(session_id);
CREATE INDEX IF NOT EXISTS idx_sample_speaker   ON speaker_samples(speaker_id);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open a connection with WAL mode and foreign key enforcement."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> sqlite3.Connection:
    """Create tables if they don't exist and apply migrations."""
    conn = get_connection(db_path)
    # Execute each DDL statement individually to avoid executescript()'s
    # implicit COMMIT, which would prematurely commit any open transaction and
    # could leave schema_version un-updated if a crash occurs mid-init.
    for stmt in _DDL.split(";"):
        stmt = stmt.strip()
        if stmt:
            conn.execute(stmt)
    _apply_migrations(conn)
    conn.commit()
    log.debug("Database initialised: %s", db_path)
    return conn


def _apply_migrations(conn: sqlite3.Connection) -> None:
    current = conn.execute(
        "SELECT MAX(version) FROM schema_version"
    ).fetchone()[0] or 0

    if current < SCHEMA_VERSION:
        conn.execute(
            "INSERT OR REPLACE INTO schema_version(version, applied_at) VALUES (?, datetime('now'))",
            (SCHEMA_VERSION,)
        )
        log.debug("Applied schema version %d", SCHEMA_VERSION)

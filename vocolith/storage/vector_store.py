# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""ChromaDB vector store for speaker voice and face embeddings."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import chromadb

log = logging.getLogger(__name__)

# Minimum cosine similarity to consider a match (ChromaDB returns distance 0=identical, 2=opposite)
# cosine_distance = 1 - cosine_similarity, so threshold 0.85 similarity = distance 0.15
_VOICE_DISTANCE_THRESHOLD = 0.15    # similarity >= 0.85
_FACE_DISTANCE_THRESHOLD = 0.40     # similarity >= 0.60 (face_recognition standard)


class VectorStore:
    """
    ChromaDB-backed storage for speaker voice and face embedding vectors.

    Collections:
        voice_embeddings  - 256-dim d-vectors from resemblyzer
        face_embeddings   - 128-dim encodings from face_recognition
    """

    def __init__(self, profiles_dir: Path) -> None:
        import chromadb as _chromadb

        chroma_path = Path(profiles_dir) / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)

        self._client = _chromadb.PersistentClient(path=str(chroma_path))
        self._voices = self._client.get_or_create_collection(
            "voice_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
        self._faces = self._client.get_or_create_collection(
            "face_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
        log.debug(
            "VectorStore: %d voice, %d face embeddings loaded",
            self._voices.count(), self._faces.count(),
        )

    # ── Voice embeddings ─────────────────────────────────────────────────────

    def add_voice(self, embedding_id: str, speaker_id: str,
                   vector: np.ndarray, meeting_id: str) -> None:
        """Store a voice embedding vector."""
        self._voices.upsert(
            ids=[embedding_id],
            embeddings=[vector.tolist()],
            metadatas=[{"speaker_id": speaker_id, "meeting_id": meeting_id}],
        )
        log.debug("Voice embedding stored: %s -> speaker %s", embedding_id, speaker_id)

    def find_voice(self, vector: np.ndarray,
                   threshold: float = _VOICE_DISTANCE_THRESHOLD,
                   exclude_speaker_id: str | None = None,
                   n_candidates: int = 3) -> tuple[str | None, float]:
        """
        Find the closest matching speaker for a voice embedding.

        Returns:
            (speaker_id, similarity_score) or (None, 0.0) if no match above threshold.
        """
        if self._voices.count() == 0:
            return None, 0.0

        results = self._voices.query(
            query_embeddings=[vector.tolist()],
            n_results=min(n_candidates, self._voices.count()),
            include=["metadatas", "distances"],
        )

        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]

        for dist, meta in zip(distances, metadatas):
            sid = str(meta.get("speaker_id", ""))
            if not sid:
                continue
            if exclude_speaker_id and sid == exclude_speaker_id:
                continue
            similarity = 1.0 - float(dist)
            if float(dist) <= threshold:
                log.debug("Voice match: speaker=%s similarity=%.3f", sid, similarity)
                return sid, similarity

        return None, 0.0

    def reassign_all_voice(self, old_speaker_id: str, new_speaker_id: str) -> int:
        """
        Move ALL voice embeddings for old_speaker_id to new_speaker_id,
        across every session.

        Used when correcting an unconfirmed placeholder profile ("Speaker_N"):
        stale embeddings from prior sessions would otherwise remain under the
        old profile and cause the speaker to be re-misidentified in future runs.
        """
        results = self._voices.get(
            where={"speaker_id": {"$eq": old_speaker_id}},
            include=["metadatas"],
        )
        ids = results["ids"]
        metadatas = results.get("metadatas") or [None] * len(ids)
        if not ids:
            return 0
        for emb_id, meta in zip(ids, metadatas):
            new_meta = {
                "speaker_id": new_speaker_id,
                "meeting_id": (meta or {}).get("meeting_id", ""),
            }
            self._voices.update(ids=[emb_id], metadatas=[new_meta])
        log.debug(
            "Reassigned %d voice embedding(s): %s → %s (all sessions)",
            len(ids), old_speaker_id, new_speaker_id,
        )
        return len(ids)

    def delete_voice(self, speaker_id: str) -> int:
        """Remove all voice embeddings for a speaker. Returns count deleted."""
        results = self._voices.get(where={"speaker_id": {"$eq": speaker_id}})
        ids = results["ids"]
        if ids:
            self._voices.delete(ids=ids)
        return len(ids)

    def delete_voice_for_session(self, speaker_id: str, session_id: str) -> int:
        """Remove voice embeddings for a speaker scoped to one session. Returns count deleted."""
        if self._voices.count() == 0:
            return 0
        results = self._voices.get(
            where={"$and": [
                {"speaker_id": {"$eq": speaker_id}},
                {"meeting_id": {"$eq": session_id}},
            ]},
            include=[],
        )
        ids = results["ids"]
        if ids:
            self._voices.delete(ids=ids)
        return len(ids)

    def voice_embedding_count(self, speaker_id: str) -> int:
        """Count voice embeddings stored for a speaker across all meetings."""
        results = self._voices.get(where={"speaker_id": {"$eq": speaker_id}})
        return len(results["ids"])

    def voice_embedding_exists_for_session(self, speaker_id: str,
                                            session_id: str) -> bool:
        """Return True if any voice embedding exists for this speaker+session."""
        if self._voices.count() == 0:
            return False
        results = self._voices.get(
            where={"$and": [
                {"speaker_id": {"$eq": speaker_id}},
                {"meeting_id": {"$eq": session_id}},
            ]},
            include=[],
        )
        return len(results["ids"]) > 0

    def reassign_voice_for_session(self, old_speaker_id: str,
                                    new_speaker_id: str, meeting_id: str) -> int:
        """
        Move voice embeddings from old_speaker_id to new_speaker_id for one
        specific meeting session.  Used when the user corrects a speaker name
        so that only the current run's data migrates to the correct profile,
        leaving the old profile's embeddings from other meetings untouched.

        Returns the number of embeddings moved.
        """
        results = self._voices.get(
            where={"$and": [
                {"speaker_id": {"$eq": old_speaker_id}},
                {"meeting_id": {"$eq": meeting_id}},
            ]}
        )
        ids = results["ids"]
        if not ids:
            return 0
        for emb_id in ids:
            self._voices.update(
                ids=[emb_id],
                metadatas=[{"speaker_id": new_speaker_id, "meeting_id": meeting_id}],
            )
        log.debug(
            "Reassigned %d embedding(s): %s → %s (meeting %s)",
            len(ids), old_speaker_id, new_speaker_id, meeting_id,
        )
        return len(ids)

    # ── Face embeddings ───────────────────────────────────────────────────────

    def add_face(self, embedding_id: str, speaker_id: str,
                  vector: np.ndarray, meeting_id: str) -> None:
        """Store a face embedding vector."""
        self._faces.upsert(
            ids=[embedding_id],
            embeddings=[vector.tolist()],
            metadatas=[{"speaker_id": speaker_id, "meeting_id": meeting_id}],
        )

    def find_face(self, vector: np.ndarray,
                   threshold: float = _FACE_DISTANCE_THRESHOLD) -> tuple[str | None, float]:
        """
        Find the closest matching speaker for a face encoding.

        Returns:
            (speaker_id, similarity_score) or (None, 0.0) if no match above threshold.
        """
        if self._faces.count() == 0:
            return None, 0.0

        results = self._faces.query(
            query_embeddings=[vector.tolist()],
            n_results=1,
            include=["metadatas", "distances"],
        )

        distances = (results.get("distances") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        if not distances or not metadatas:
            return None, 0.0
        dist = float(distances[0])
        meta = metadatas[0]
        similarity = 1.0 - dist
        if dist <= threshold:
            return str(meta.get("speaker_id", "")), similarity
        return None, 0.0

    def delete_face(self, speaker_id: str) -> int:
        """Remove all face embeddings for a speaker. Returns count deleted."""
        results = self._faces.get(where={"speaker_id": {"$eq": speaker_id}})
        ids = results["ids"]
        if ids:
            self._faces.delete(ids=ids)
        return len(ids)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def counts(self) -> dict:
        return {
            "voice_embeddings": self._voices.count(),
            "face_embeddings": self._faces.count(),
        }

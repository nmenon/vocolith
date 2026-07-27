# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""
Stage 8: Speaker resolution — maps diarization labels to named SpeakerProfiles.

Resolution priority:
  1. Voice HIGH  (>=0.92) — biometric, auto-accepted, no confirmation
  2. OCR name match
  3. Addressee inference ("Alice, can you...?")
  4. Voice STANDARD (>=0.85) — accepted if 2+3 silent
  5. Multi-signal boost: voice(mid) + OCR/addressee agree on same person
  6. Face recognition (optional)
  7. Fallback → Speaker_N

Strategies 2-6 are "auto-identified" and can require user confirmation
(see speaker_resolution.confirm_auto_identified in config.yaml, or --confirm).
"""
from __future__ import annotations
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vocolith.pipeline import PipelineContext

log = logging.getLogger(__name__)


@dataclass
class PendingSpeaker:
    """A speaker that has been auto-identified but not yet confirmed by the user."""
    label: str            # diarization label e.g. "SPEAKER_01"
    display_name: str     # resolved name e.g. "Alice Smith"
    method: str           # how it was resolved e.g. "ocr", "voice(mid,0.87)"
    speaker_id: str       # profile UUID in SQLite
    auto_confirmed: bool  # True = voice HIGH, no confirmation needed
    # Set when this label was detected as the shared conference-room mic —
    # routes straight into the segment-by-segment split wizard instead of
    # the normal single-name confirmation panel.
    needs_room_split: bool = False
    room_candidates: list[str] = field(default_factory=list)
    # Best-effort per-segment name guesses (segment_id -> name) used to
    # pre-fill the split wizard.  Segments with no guess are simply absent.
    segment_hints: dict[int, str] = field(default_factory=dict)


def _extract_first_name(full_name: str) -> str:
    """Extract the first name from 'First Last(ORG)' or 'First Last' format."""
    import re as _re
    # Strip org suffix: "Alice Smith(Acme)" -> "Alice Smith"
    name = _re.sub(r'\([^)]*\)', '', full_name).strip()
    return name.split()[0] if name.split() else ""


def _resolve_to_attendee(name: str, attendees: list[str]) -> str | None:
    """
    Return the canonical attendee name that best matches `name`, or None.

    Matching priority (case-insensitive):
      1. Exact match against full attendee string
      2. Exact match against attendee first name only
         (e.g. OCR "alice" matches attendee "Alice Smith(Org)")
      3. Attendee string starts with `name`
      4. `name` appears anywhere in attendee string (last resort)
    """
    if not attendees or not name:
        return None
    name_lower = name.lower().strip()
    # Pass 1: exact full match
    for attendee in attendees:
        if not attendee:
            continue
        if attendee.lower().strip() == name_lower:
            return attendee
    # Pass 2: exact first-name match
    for attendee in attendees:
        if not attendee:
            continue
        if _extract_first_name(attendee).lower() == name_lower:
            return attendee
    # Pass 3: attendee string starts with name
    for attendee in attendees:
        if not attendee:
            continue
        if attendee.lower().startswith(name_lower):
            return attendee
    # Pass 4: name is a substring of an attendee (last resort)
    for attendee in attendees:
        if not attendee:
            continue
        if name_lower in attendee.lower():
            return attendee
    return None


def _mine_names_from_transcript(ctx: "PipelineContext") -> list[str]:
    """
    Extract likely person names from transcript text without any external input.

    Uses two complementary strategies:

    1. Addressee patterns: "Hi Alice", "Thanks Bob", "Alice, can you..."
       Strong signal — these are almost always direct name uses.

    2. Repeated proper nouns: capitalised single words that appear 3+ times
       across speaker turns and don't look like acronyms or common words.

    Returns a deduplicated list of candidate names, ordered by frequency.
    This runs before addressee inference so newly discovered names become seeds.
    """
    if not ctx.transcript:
        return []

    import re as _re

    # Patterns that strongly indicate a person is being addressed by name
    _ADDRESSEE_PATTERNS = [
        r'\b([A-Z][a-z]{1,20}),\s+(?:can|could|would|do|did|is|are|have|will|should)',
        r'\b([A-Z][a-z]{1,20}),?\s+(?:what|how|when|where|why|who)\b',
        r'(?:Hi|Hey|Thanks|Thank you|Sorry|Okay|Right),?\s+([A-Z][a-z]{1,20})\b',
        r'\bask\s+([A-Z][a-z]{1,20})\b',
        r'\b([A-Z][a-z]{1,20})\s+(?:mentioned|said|noted|pointed|confirmed|agreed)\b',
    ]

    # Common English words that match capitalised-word patterns but aren't names
    _STOPWORDS = frozenset({
        "The", "This", "That", "These", "Those", "There", "Their", "They",
        "What", "When", "Where", "Which", "While", "With", "From", "Into",
        "About", "After", "Before", "During", "Between", "Through",
        "Because", "Although", "However", "Therefore", "Also", "Just",
        "Very", "Really", "Actually", "Maybe", "Probably", "Already",
        "Always", "Never", "Still", "Well", "Okay", "Right", "Yeah",
        "Yes", "No", "But", "And", "For", "Not", "Can", "Will", "May",
        "Should", "Could", "Would", "Have", "Has", "Had", "Does", "Did",
        "Are", "Was", "Were", "Been", "Being", "Let", "Get", "Got",
        "Make", "Made", "Take", "Took", "Give", "Given", "Come", "Came",
        "Think", "Know", "See", "Look", "Want", "Need", "Going", "Like",
        "Good", "New", "First", "Last", "Next", "Same", "Different",
    })

    all_text = " ".join(s.text for s in ctx.transcript.segments)

    # Strategy 1: addressee patterns
    addressee_hits: Counter = Counter()
    for pattern in _ADDRESSEE_PATTERNS:
        for m in _re.finditer(pattern, all_text):
            name = m.group(1)
            if name not in _STOPWORDS and 2 <= len(name) <= 25:
                addressee_hits[name] += 1

    # Strategy 2: repeated proper nouns (capitalised, not sentence-start noise)
    # Split into individual speaker turns to reduce sentence-start capitals
    proper_counter: Counter = Counter()
    for seg in ctx.transcript.segments:
        words = seg.text.split()
        # Skip the very first word of each turn (likely sentence-start capital)
        for word in words[1:]:
            clean = _re.sub(r"[^A-Za-z]", "", word)   # strip ALL punctuation incl apostrophes
            if (clean and clean[0].isupper() and clean[1:].islower()
                    and 3 <= len(clean) <= 20
                    and clean not in _STOPWORDS):
                proper_counter[clean] += 1

    # Require 3+ occurrences for proper noun mining (reduces false positives)
    frequent_proper = {w for w, c in proper_counter.items() if c >= 3}

    # Merge: addressee hits are high-quality (any count); proper nouns need 3+
    candidates: dict[str, int] = {}
    for name, count in addressee_hits.items():
        candidates[name] = candidates.get(name, 0) + count * 3  # weight addressee hits
    for name in frequent_proper:
        candidates[name] = candidates.get(name, 0) + proper_counter[name]

    # Sort by combined score, return top candidates
    return [name for name, _ in sorted(candidates.items(), key=lambda x: -x[1])[:20]]


def _apply_names_to_transcript(ctx: "PipelineContext",
                                label_to_name: dict[str, str],
                                label_to_method: dict[str, str]) -> None:
    """Write resolved names into transcript segments."""
    if not ctx.transcript:
        return
    for seg in ctx.transcript.segments:
        if seg.speaker_label and seg.speaker_label in label_to_name:
            seg.speaker_name = label_to_name[seg.speaker_label]
            seg.resolution_method = label_to_method[seg.speaker_label]


def resolve_speakers(ctx: "PipelineContext") -> "PipelineContext":
    """
    Assign human-readable names to diarization speaker labels.

    Modifies ctx.transcript segments in-place (sets speaker_name,
    resolution_method) and persists new/updated speaker profiles.

    Returns the modified context.
    """
    from vocolith.config import AppConfig
    from vocolith.models.speaker import SpeakerProfile
    from vocolith.storage.db import init_db
    from vocolith.storage.speaker_store import SpeakerStore
    from vocolith.storage.session_store import SessionStore
    from vocolith.storage.vector_store import VectorStore
    from vocolith.stages.voice_embedder import compute_speaker_embeddings
    from vocolith.models.meeting import MeetingSession

    cfg = ctx.config
    profiles_dir = Path(cfg.storage.profiles_dir)
    db_path = profiles_dir / cfg.storage.db_filename

    # Initialise storage
    conn = init_db(db_path)
    speaker_store = SpeakerStore(conn)
    session_store = SessionStore(conn)
    vector_store = VectorStore(profiles_dir)

    # Compute voice embeddings for all diarized speakers
    voice_embeddings: dict[str, np.ndarray] = {}
    if ctx.effective_audio and ctx.transcript and ctx.transcript.speakers_detected > 0:
        voice_embeddings = compute_speaker_embeddings(
            ctx.effective_audio,
            ctx.transcript,
        )

    # Build unique set of speaker labels present in transcript
    labels = sorted({
        s.speaker_label for s in ctx.transcript.segments
        if s.speaker_label
    }) if ctx.transcript else []

    if not labels:
        log.info("No speaker labels in transcript — all segments remain 'Unknown'.")
        return ctx

    log.info("Resolving %d speaker label(s): %s", len(labels), labels)

    label_to_name: dict[str, str] = {}
    label_to_method: dict[str, str] = {}
    label_to_id: dict[str, str] = {}
    pending_speakers: list[PendingSpeaker] = []  # for confirmation wizard

    # If attendees were provided, filter OCR names to only those matching an attendee.
    # Browser chrome (tab titles, bookmarks) won't match real participant names.
    ocr_names_filtered = ctx.ocr_names
    if ctx.attendees:
        attendee_first_names = {
            _extract_first_name(a).lower() for a in ctx.attendees if _extract_first_name(a)
        }
        attendee_full_lower = {a.lower() for a in ctx.attendees}
        ocr_names_filtered = [
            n for n in ctx.ocr_names
            if n.lower() in attendee_full_lower
            or any(n.lower() in a.lower() or a.lower().startswith(n.lower())
                   for a in ctx.attendees)
            or n.lower() in attendee_first_names
        ]
        if len(ocr_names_filtered) < len(ctx.ocr_names):
            log.info(
                "OCR names filtered to attendee list: %d → %d  (removed: %s)",
                len(ctx.ocr_names), len(ocr_names_filtered),
                [n for n in ctx.ocr_names if n not in ocr_names_filtered][:5],
            )

    # Build candidate name pool: OCR names (filtered) + --attendees + transcript-mined names
    all_candidate_names = list(ocr_names_filtered)
    for name in ctx.attendees:
        if name.lower() not in [n.lower() for n in all_candidate_names]:
            all_candidate_names.append(name)

    # Mine additional candidate names from transcript text — enables addressee
    # inference on fresh meetings with no OCR names and no --attendees
    mined = _mine_names_from_transcript(ctx)
    for name in mined:
        if name.lower() not in [n.lower() for n in all_candidate_names]:
            all_candidate_names.append(name)
    if mined:
        log.info("Mined %d candidate name(s) from transcript: %s", len(mined), mined)

    ocr_names_lower = [n.lower() for n in all_candidate_names]

    if ctx.attendees:
        log.info("Attendee hints: %s", ctx.attendees)

    sr = cfg.speaker_resolution

    for i, label in enumerate(labels):
        speaker_id: str | None = None
        method: str = "fallback"
        display_name: str = f"Speaker_{i + 1}"
        _is_self_match: bool = False   # True when sim≥0.999 (same audio re-run)

        # ── Strategy 1: Voice d-vector — HIGH confidence (immediate accept) ──
        # Biometric signal. If similarity ≥ threshold_high we trust it fully
        # and skip every other strategy — no point checking OCR or names.
        voice_mid_id: str | None = None    # medium confidence voice match for later
        voice_mid_sim: float = 0.0
        if label in voice_embeddings:
            vec = voice_embeddings[label]
            matched_id, sim = vector_store.find_voice(
                vec, threshold=1.0 - sr.voice_similarity_threshold_high
            )
            if matched_id and sim >= 0.999:
                # Cosine sim ≈ 1.0 means the stored embedding is numerically
                # identical to the query — almost always the same audio file
                # being re-processed (self-match from a previous run).
                _is_self_match = True   # track as bool — no string parsing later
                _self_profile = speaker_store.get(matched_id)
                _is_confirmed = (
                    _self_profile is not None
                    and not _self_profile.display_name.startswith("Speaker_")
                )
                if _is_confirmed:
                    log.debug(
                        "  %s: sim=%.4f self-match with confirmed profile '%s' "
                        "— keeping high confidence (re-run of same audio)",
                        label, sim, _self_profile.display_name,
                    )
                    # matched_id kept → strategy 1 path proceeds as HIGH
                else:
                    log.debug(
                        "  %s: sim=%.4f self-match with unconfirmed profile "
                        "— demoting to medium confidence",
                        label, sim,
                    )
                    voice_mid_id = matched_id
                    voice_mid_sim = sim
                    matched_id = None
            if matched_id:
                profile = speaker_store.get(matched_id)
                if profile:
                    speaker_id = profile.speaker_id
                    display_name = profile.display_name
                    method = f"voice(high,{sim:.2f})"
                    log.info("  %s -> '%s' via voice HIGH (sim=%.3f) — accepted immediately",
                             label, display_name, sim)
            elif not voice_mid_id:
                # No high match and no self-match — try at standard threshold
                matched_id2, sim2 = vector_store.find_voice(
                    vec, threshold=1.0 - sr.voice_similarity_threshold
                )
                if matched_id2:
                    voice_mid_id = matched_id2
                    voice_mid_sim = sim2

        if speaker_id:
            # High-confidence voice hit — skip all other strategies
            pass

        else:
            # ── Reference transcript (e.g. Teams .vtt) cross-reference ───────
            # Computed once here so both the room-mixed check and Strategy 2b
            # below can use it.
            ref_match: str | None = None
            ref_coverage: float = 0.0
            if ctx.reference_entries:
                ref_match, ref_coverage = _match_reference_transcript(label, ctx)

            # ── Room-mixed detection: shared conference-room mic ──────────────
            # A label is "room-mixed" (multiple people on one physical mic)
            # when either the user told us which reference-transcript name is
            # the room device (--room-label), or no single reference speaker
            # dominates this label's overlap (--room-attendees given, no
            # --room-label).  Routes straight to the split wizard instead of
            # picking a single (wrong) name.
            is_room_mixed = False
            if ctx.room_attendees:
                if ctx.room_label and ref_match and ref_match.lower() == ctx.room_label.lower():
                    is_room_mixed = True
                elif not ctx.room_label and ctx.reference_entries and ref_coverage < sr.reference_match_threshold:
                    is_room_mixed = True

            if is_room_mixed:
                display_name = f"Speaker_{i + 1} (room)"
                profile = SpeakerProfile(display_name=display_name)
                speaker_store.save(profile)
                speaker_id = profile.speaker_id
                ctx.pending_new_profiles.add(label)
                method = "room(needs_split)"
                hints = _room_split_hints(label, ctx, ctx.room_attendees)
                log.info("  %s -> room-mixed, needs split among %s (%d segment hint(s))",
                         label, ctx.room_attendees, len(hints))

                label_to_name[label] = display_name
                label_to_method[label] = method
                label_to_id[label] = speaker_id
                pending_speakers.append(PendingSpeaker(
                    label=label,
                    display_name=display_name,
                    method=method,
                    speaker_id=speaker_id,
                    auto_confirmed=False,
                    needs_room_split=True,
                    room_candidates=list(ctx.room_attendees),
                    segment_hints=hints,
                ))
                continue  # skip strategies 2-6 and the common tail below

            # ── Strategy 2: Voice d-vector — MEDIUM confidence ───────────────
            # Voice is tried before OCR/addressee: a biometric match is always
            # more reliable than screen text.  OCR and addressee inference only
            # run when voice finds nothing at all.
            if voice_mid_id:
                profile = speaker_store.get(voice_mid_id)
                if profile:
                    speaker_id = profile.speaker_id
                    display_name = profile.display_name
                    method = f"voice(mid,{voice_mid_sim:.2f})"
                    log.info("  %s -> '%s' via voice MED (sim=%.3f)",
                             label, display_name, voice_mid_sim)

            # ── Strategy 2b: Reference transcript — clean single-speaker match ──
            # A Teams-style reference transcript's per-device attribution is a
            # stronger signal than the OCR/addressee text heuristics below —
            # trusted here, ahead of them, but below both voice tiers so
            # cross-meeting biometric identity continuity still wins.
            if not speaker_id and ref_match and ref_coverage >= sr.reference_match_threshold:
                canonical = _resolve_to_attendee(ref_match, ctx.attendees) or ref_match
                existing = (speaker_store.find_by_name(canonical)
                            or speaker_store.find_by_alias(canonical)
                            or (speaker_store.find_by_alias(ref_match)
                                if canonical != ref_match else None))
                if existing:
                    speaker_id = existing.speaker_id
                    display_name = existing.display_name
                else:
                    profile = SpeakerProfile(display_name=canonical)
                    speaker_store.save(profile)
                    speaker_store.add_alias(profile.speaker_id, canonical, "reference_transcript")
                    if canonical != ref_match:
                        speaker_store.add_alias(profile.speaker_id, ref_match, "reference_transcript")
                    speaker_id = profile.speaker_id
                    display_name = canonical
                    ctx.pending_new_profiles.add(label)
                method = f"reference_transcript({ref_coverage:.2f})"
                log.info("  %s -> '%s' via reference transcript (coverage=%.2f)",
                         label, display_name, ref_coverage)

            # ── Strategy 3: Addressee inference ──────────────────────────────
            # Speech-based: "Hey Alice, can you..." — more reliable than OCR.
            if not speaker_id:
                inferred = _try_addressee_inference(
                    label, labels, ctx, ocr_names_lower,
                    sr.addressee_min_votes
                )
                if inferred:
                    # Resolve to canonical attendee name if available
                    canonical = _resolve_to_attendee(inferred, ctx.attendees) or inferred
                    existing = (speaker_store.find_by_name(canonical)
                                or speaker_store.find_by_alias(canonical)
                                or (speaker_store.find_by_alias(inferred)
                                    if canonical != inferred else None))
                    if existing:
                        speaker_id = existing.speaker_id
                        display_name = existing.display_name
                    else:
                        profile = SpeakerProfile(display_name=canonical)
                        speaker_store.save(profile)
                        speaker_store.add_alias(profile.speaker_id, canonical, "addressee")
                        if canonical != inferred:
                            # Also alias the raw inferred name for future matching
                            speaker_store.add_alias(profile.speaker_id, inferred, "addressee")
                        speaker_id = profile.speaker_id
                        display_name = canonical
                        ctx.pending_new_profiles.add(label)
                    method = "addressee"
                    log.info("  %s -> '%s' via addressee inference", label, display_name)

            # ── Strategy 4: OCR name match ────────────────────────────────────
            # Last text-based resort: unreliable in screen-recording scenarios
            # (participant sidebar keeps all names visible regardless of who is
            # speaking), so only used when voice and speech both found nothing.
            if not speaker_id:
                ocr_name = _try_ocr_match(label, ctx, ocr_names_lower, cfg)
                if ocr_name:
                    # Resolve raw OCR text to canonical attendee name if available
                    canonical = _resolve_to_attendee(ocr_name, ctx.attendees) or ocr_name
                    existing = (speaker_store.find_by_name(canonical)
                                or speaker_store.find_by_alias(canonical)
                                or (speaker_store.find_by_alias(ocr_name)
                                    if canonical != ocr_name else None))
                    if existing:
                        speaker_id = existing.speaker_id
                        display_name = existing.display_name
                    else:
                        profile = SpeakerProfile(display_name=canonical)
                        speaker_store.save(profile)
                        speaker_store.add_alias(profile.speaker_id, canonical, "ocr")
                        if canonical != ocr_name:
                            # Also alias the raw OCR text for future matching
                            speaker_store.add_alias(profile.speaker_id, ocr_name, "ocr")
                        speaker_id = profile.speaker_id
                        display_name = canonical
                        ctx.pending_new_profiles.add(label)
                    method = "ocr"
                    log.info("  %s -> '%s' via OCR", label, display_name)

            # ── Strategy 5: Face recognition ──────────────────────────────────
            if not speaker_id and ctx.face_detections:
                face_match = _try_face_match(label, ctx, vector_store)
                if face_match:
                    profile = speaker_store.get(face_match)
                    if profile:
                        speaker_id = profile.speaker_id
                        display_name = profile.display_name
                        method = "face"
                        log.info("  %s -> '%s' via face", label, display_name)

            # ── Strategy 6: Fallback — new unknown speaker ────────────────────
            if not speaker_id:
                profile = SpeakerProfile(display_name=display_name)
                speaker_store.save(profile)
                speaker_id = profile.speaker_id
                ctx.pending_new_profiles.add(label)
                log.info("  %s -> '%s' (new, fallback)", label, display_name)

        # Voice embeddings: defer to wizard so they are stored only after the
        # user has verified the speaker name.  This ensures embeddings always
        # match the human-confirmed identity, never an auto-assigned guess.
        #
        # When confirmation is disabled (--no-confirm), store here directly
        # with guards: skip self-matches (same audio re-run) and skip if this
        # session already has an embedding for this profile.
        if label in voice_embeddings and speaker_id:
            confirm_enabled = cfg.speaker_resolution.confirm_auto_identified
            if confirm_enabled:
                # Queue for wizard — stored in _store_confirmed_embeddings()
                ctx.pending_voice_embeddings[label] = voice_embeddings[label]
            elif not _is_self_match and not vector_store.voice_embedding_exists_for_session(
                    speaker_id, ctx.session_id):
                import uuid
                emb_id = str(uuid.uuid4())
                vector_store.add_voice(
                    emb_id, speaker_id, voice_embeddings[label], ctx.session_id
                )
                speaker_store.record_embedding(
                    "voice_embeddings", emb_id, speaker_id, ctx.session_id
                )
                log.debug("  %s: voice embedding stored for '%s' (no-confirm mode)",
                          label, display_name)

        # Update speaker last seen once per profile per session to avoid
        # inflating meeting_count when multiple labels resolve to same speaker.
        # Skip pending new profiles — their touch() is deferred to the wizard
        # so meeting_count is incremented once, under the confirmed name.
        if speaker_id not in label_to_id.values() and label not in ctx.pending_new_profiles:
            speaker_store.touch(speaker_id)

        label_to_name[label] = display_name
        label_to_method[label] = method
        label_to_id[label] = speaker_id

        # All speakers require user confirmation — voice matches are shown
        # with a "voice pattern match" note but still need an Enter press.
        is_auto_confirmed = False
        assert speaker_id is not None  # fallback always creates one
        pending_speakers.append(PendingSpeaker(
            label=label,
            display_name=display_name,
            method=method,
            speaker_id=speaker_id,
            auto_confirmed=is_auto_confirmed,
        ))

    # Store pending speakers on context so pipeline can run confirmation wizard
    ctx.pending_speakers = pending_speakers

    # Apply resolved names to transcript
    # (may be overridden by confirmation wizard before transcript.md is written)
    _apply_names_to_transcript(ctx, label_to_name, label_to_method)
    ctx.speaker_map = label_to_name

    # Persist session record
    from datetime import datetime, timezone
    session = MeetingSession(
        session_id=ctx.session_id,
        video_path=str(ctx.video_path),
        video_filename=ctx.video_path.name,
        processed_at=datetime.now(timezone.utc),
        duration_seconds=ctx.transcript.duration_seconds if ctx.transcript else 0.0,
        language=ctx.transcript.language if ctx.transcript else "en",
        speaker_ids=list(label_to_id.values()),
        output_dir=str(ctx.output_dir),
    )
    session_store.save(session)
    for label, sid in label_to_id.items():
        session_store.link_speaker(
            ctx.session_id, sid, label, label_to_method.get(label, "fallback")
        )

    return ctx


# ── Internal helpers ──────────────────────────────────────────────────────────

def _try_addressee_inference(
    label: str,
    all_labels: list[str],
    ctx: "PipelineContext",
    ocr_names_lower: list[str],
    min_votes: int = 3,
) -> str | None:
    """
    Infer speaker identity from conversational name references.

    Pattern: if speaker A says "Alice, what do you think?" and Alice has
    been identified as a known name (via OCR or already resolved), then the
    next speaker after A is probably Alice.

    Uses a scoring approach:
    - For each segment from this speaker, look at the immediately following segment
    - If that following segment belongs to label X, and the current segment's
      text ends with/contains a known name N, then +1 vote for "label X = N"
    - The label with the highest consistent vote score gets assigned if >= threshold

    Only fires when:
    - The name being referenced appears in ocr_names or already-resolved speakers
    - The pattern occurs >= 2 times (reduces false positives)
    - No other resolution method has already identified the label

    Args:
        label:          The diarization label we're trying to resolve.
        all_labels:     All speaker labels in this meeting.
        ctx:            Pipeline context with transcript.
        ocr_names_lower: OCR-extracted names (lowercase).

    Returns:
        Inferred display name, or None if pattern not strong enough.
    """
    import re

    if not ctx.transcript or not ocr_names_lower:
        return None

    segments = ctx.transcript.segments
    if len(segments) < 2:
        return None

    # Build index: position -> segment
    # For label, find segments where the speaker calls someone by name,
    # then check what label responds next.
    candidate_votes = Counter()

    for i, seg in enumerate(segments[:-1]):
        if seg.speaker_label != label:
            continue

        next_seg = segments[i + 1]
        next_label = next_seg.speaker_label
        if not next_label or next_label == label:
            continue

        # Check if current segment text addresses a known name
        text_lower = seg.text.lower().strip()
        for name_lower in ocr_names_lower:
            if len(name_lower) < 2:
                continue
            # Name appears at start (direct address) or as "Hey X," or "X, can you"
            # Use word-boundary regex to avoid partial matches
            pattern = r'\b' + re.escape(name_lower) + r'\b'
            if re.search(pattern, text_lower):
                # The next speaker is probably the addressed person
                # Vote: next_label -> name
                candidate_votes[(next_label, name_lower)] += 1

    if not candidate_votes:
        return None

    # Find the best-voted (next_label == our label) entry
    best_count = 0
    best_name = None
    for (next_label, name), count in candidate_votes.items():
        if next_label == label and count >= min_votes:
            if count > best_count:
                best_count = count
                best_name = name

    if best_name:
        # Return with original casing from ocr_names
        for ocr_name in ctx.ocr_names:
            if ocr_name.lower() == best_name:
                return ocr_name
        return best_name.title()

    return None


def _try_ocr_match(
    label: str,
    ctx: "PipelineContext",
    ocr_names_lower: list[str],
    cfg,
) -> str | None:
    """
    Check if an OCR name is temporally correlated with this speaker's turns.
    Returns the matched name string or None.
    """
    if not ctx.ocr_names or not ctx.frames:
        return None

    # Get the time windows where this speaker is talking
    speaker_windows: list[tuple[float, float]] = []
    if ctx.transcript:
        for seg in ctx.transcript.segments:
            if seg.speaker_label == label:
                speaker_windows.append((seg.start, seg.end))

    if not speaker_windows:
        return None

    # Temporal correlation: find which OCR names were visible while this speaker
    # was talking, using the per-frame name map (timestamp → names in that frame).
    frame_name_map: dict[float, list[str]] = getattr(ctx, "frame_name_map", {})
    if not frame_name_map:
        return None

    # Vote: name → overlap score (sum of frame durations while speaker is talking)
    # A frame is "during" a speaker turn if its timestamp falls in any turn window
    # Build a set of allowed names for O(1) lookup.
    # Only names in the attendee-filtered candidate pool can win a vote —
    # raw OCR noise like browser tab titles ("ChatLLM Teams") is excluded.
    allowed = set(ocr_names_lower)

    name_votes: Counter = Counter()
    for ts, names in frame_name_map.items():
        for start, end in speaker_windows:
            if start <= ts <= end:
                for name in names:
                    if name.lower() in allowed:
                        name_votes[name] += 1
                break  # don't double-count this frame

    if not name_votes:
        return None

    # Pick the most-seen name; require it appears in >50% of this speaker's frames
    best_name, best_count = name_votes.most_common(1)[0]
    total_frames_for_speaker = sum(
        1 for ts in frame_name_map
        if any(s <= ts <= e for s, e in speaker_windows)
    )
    if total_frames_for_speaker == 0:
        return None

    coverage = best_count / total_frames_for_speaker
    if coverage >= 0.5:
        log.debug("OCR temporal match: '%s' in %.0f%% of frames for %s",
                  best_name, coverage * 100, label)
        return best_name

    return None


def _match_reference_transcript(
    label: str,
    ctx: "PipelineContext",
) -> tuple[str | None, float]:
    """
    Time-overlap vote of ctx.reference_entries against this label's segment
    windows — same overlap-voting shape as _try_ocr_match, but using real
    durations from a reference transcript (e.g. Teams .vtt) rather than
    frame counts.

    Returns (best_speaker_name, coverage) where coverage is the share of all
    reference speech overlapping this label's turns that is attributed to
    best_speaker_name.  A high coverage means one reference speaker's speech
    lines up almost exclusively with this label's turns — a clean 1:1 match.
    A low coverage means this label's turns overlap diffusely with several
    different reference speakers.  Returns (None, 0.0) when there is nothing
    to compare.
    """
    if not ctx.reference_entries or not ctx.transcript:
        return None, 0.0

    windows = [
        (seg.start, seg.end)
        for seg in ctx.transcript.segments
        if seg.speaker_label == label
    ]
    if not windows:
        return None, 0.0

    votes: Counter = Counter()
    total = 0.0
    for entry in ctx.reference_entries:
        for start, end in windows:
            overlap = min(entry.end, end) - max(entry.start, start)
            if overlap > 0:
                votes[entry.speaker] += overlap
                total += overlap
                break  # count each reference entry once per label
    if not votes or total <= 0:
        return None, 0.0

    best_speaker, best_overlap = votes.most_common(1)[0]
    return best_speaker, best_overlap / total


def _room_split_hints(
    label: str,
    ctx: "PipelineContext",
    room_candidates: list[str],
) -> dict[int, str]:
    """
    Best-effort per-segment name guesses for a room-mixed diarization label,
    restricted to `room_candidates`.  Used to pre-fill the segment-by-segment
    split wizard so most turns just need an Enter to confirm.

    Two signal sources, tried per segment:
      1. A single reference-transcript entry overlapping this segment whose
         speaker resolves to a room candidate.
      2. Addressee inference on the immediately preceding transcript segment
         ("Hey Bob, ...") restricted to room candidate names.

    Segments with no hint are simply absent from the returned dict.
    """
    if not ctx.transcript or not room_candidates:
        return {}

    label_segments = [s for s in ctx.transcript.segments if s.speaker_label == label]
    if not label_segments:
        return {}

    hints: dict[int, str] = {}

    # Signal 1: reference-transcript overlap, restricted to a single
    # unambiguous overlapping entry per segment.
    if ctx.reference_entries:
        for seg in label_segments:
            overlapping = [
                e for e in ctx.reference_entries
                if min(e.end, seg.end) - max(e.start, seg.start) > 0
            ]
            if len(overlapping) == 1:
                match = _resolve_to_attendee(overlapping[0].speaker, room_candidates)
                if match:
                    hints[seg.segment_id] = match

    # Signal 2: addressee inference on the preceding segment's text,
    # restricted to room candidate names only.
    import re as _re
    all_segments = ctx.transcript.segments
    for i, seg in enumerate(all_segments):
        if seg.speaker_label != label or seg.segment_id in hints or i == 0:
            continue
        prev_text = all_segments[i - 1].text.lower().strip()
        for candidate in room_candidates:
            name_lower = _extract_first_name(candidate).lower() or candidate.lower()
            if len(name_lower) < 2:
                continue
            pattern = r'\b' + _re.escape(name_lower) + r'\b'
            if _re.search(pattern, prev_text):
                hints[seg.segment_id] = candidate
                break

    return hints


def _try_face_match(
    label: str,
    ctx: "PipelineContext",
    vector_store,
) -> str | None:
    """
    Find a face detection temporally overlapping with speaker turns and
    match against stored face embeddings.
    Returns matched speaker_id or None.
    """
    if not ctx.face_detections or not ctx.transcript:
        return None

    # Get speaker turn windows
    windows: list[tuple[float, float]] = [
        (seg.start, seg.end)
        for seg in ctx.transcript.segments
        if seg.speaker_label == label
    ]

    for det in ctx.face_detections:
        ts = det.frame_timestamp_s
        for start, end in windows:
            if start <= ts <= end:
                vec = np.array(det.face_encoding, dtype=np.float32)
                matched_id, _sim = vector_store.find_face(vec)
                if matched_id:
                    return matched_id
    return None

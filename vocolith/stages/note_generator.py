# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""
Stage 9: AI-powered meeting notes generation via LLM.

Takes the diarized transcript and generates structured MeetingNotes
(summary, action items, decisions, attendees) using the configured LLM.
Handles long transcripts by chunking and merging.
"""
from __future__ import annotations
import json
import logging
from datetime import date as DateType
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vocolith.pipeline import PipelineContext

log = logging.getLogger(__name__)

# System prompt for meeting note generation
_SYSTEM_PROMPT_TEMPLATE = """You are a precise meeting note-taker. Your only job is to EXTRACT information
that was explicitly stated in the transcript — never infer, guess, or complete missing information.

STRICT RULES — follow these without exception:
1. ONLY include facts explicitly stated in the transcript. If something was not said, omit it.
2. NEVER invent names, dates, numbers, decisions, or action items not present in the transcript.
3. If you are uncertain whether something was decided or assigned, put it in follow_up_questions instead.
4. Due dates: only set if a specific date or deadline was explicitly mentioned. Use null otherwise.
5. Assignee: only set if a specific person was explicitly asked to do something. Use null otherwise.
6. Priority: only set if urgency was explicitly stated ("urgent", "ASAP", "before Friday"). Use null otherwise.
7. For each decision and action item, include a short verbatim source_quote from the transcript.
8. If a speaker's name is unknown, use their label (e.g. "Speaker_1") — do not guess names.
9. Mark anything you are even slightly uncertain about with [?] at the end of the field value.
10. An empty list is always correct when nothing was found. Do not pad with guesses.

Return a single valid JSON object with EXACTLY these fields:
{{
  "title": "Brief factual title derived from the transcript content",
  "attendees": ["names of participants who actually spoke or were identified"],
  "summary": "Factual summary using only information from the transcript",
  "agenda_items": ["topics that were actually discussed"],
  "key_topics": ["technical terms, product names, or themes explicitly mentioned"],
  "decisions": [
    {{
      "description": "exact decision made — use words close to what was said",
      "decided_by": "speaker name or null if not stated",
      "context": "brief rationale if stated, else null",
      "source_quote": "short verbatim excerpt supporting this decision",
      "timestamp": "approximate [MM:SS] from transcript"
    }}
  ],
  "action_items": [
    {{
      "description": "what was explicitly asked to be done",
      "assignee": "person explicitly assigned, or null",
      "due_date": "YYYY-MM-DD only if a date was stated, else null",
      "priority": "high|medium|low only if urgency was stated, else null",
      "source_quote": "short verbatim excerpt supporting this action item",
      "timestamp": "approximate [MM:SS] from transcript"
    }}
  ],
  "follow_up_questions": ["things left unresolved, unclear, or that need follow-up"],
  "extra": {{}}
}}

{template_guidance}

Known technical terms from this meeting (use exact spelling): {terminology}
"""

# Template-specific guidance injected into the system prompt
_TEMPLATE_GUIDANCE = {
    "standard": "",
    "standup": (
        'Also populate extra.standup as a dict mapping each attendee to '
        '{"yesterday": "...", "today": "...", "blockers": "..."}. '
        'Extract standup updates from each person\'s speaking turns.'
    ),
    "design_review": (
        'Also populate extra.problem_statement with the problem being reviewed, '
        'and extra.proposals as a list of {title, description, pros, cons} objects. '
        'Capture technical rationale for decisions.'
    ),
    "one_on_one": (
        'Populate extra.feedback with any explicit feedback exchanged, '
        'and extra.goals_progress as a list of goal progress updates.'
    ),
    "brainstorm": (
        'Populate extra.ideas_generated as a full list of ideas mentioned, '
        'and extra.ideas_shortlisted as the subset chosen for follow-up. '
        'Capture the problem statement in extra.problem_statement.'
    ),
    "detailed_technical_discussion_notes": (
        'Populate extra.discussion_sections as a list of objects, one per '
        'major topic discussed. Each object must have: '
        '"title" (short topic label, same as its agenda_items entry) and '
        '"points" (list of concise bullet-point strings covering what was '
        'said, raised, or resolved under that topic — 3 to 8 bullets per '
        'topic, using only facts from the transcript). '
        'Example: {"title": "LP3 IO retention handoff", "points": ['
        '"PHY controls MEMRESET until DFI init complete", '
        '"Controller takes over after DFI init complete signal", '
        '"Exact handoff signal still needs Woody confirmation"]}. '
        'Every topic in agenda_items must have a matching entry in '
        'extra.discussion_sections.'
    ),
}


def _load_guidance_sidecar(template_key: str, cfg) -> str:
    """
    Look for a <template_key>.guidance.txt file alongside the template.
    This lets custom templates ship LLM extraction instructions without
    touching note_generator.py.

    Search order mirrors template resolution:
      1. User templates dir
      2. Built-in templates dir
    """
    from vocolith.utils.template_renderer import get_template_dirs
    for directory in get_template_dirs(cfg.templates.user_templates_dir):
        sidecar = directory / f"{template_key}.guidance.txt"
        if sidecar.exists():
            log.debug("Loaded guidance sidecar: %s", sidecar)
            return sidecar.read_text(encoding="utf-8").strip()
    return ""


def generate_notes(
    ctx: "PipelineContext",
    template: str | None = None,
) -> "PipelineContext":
    """
    Generate structured meeting notes for each template in cfg.templates.run.

    If ``template`` is given (from --template CLI flag) it is used AS WELL AS
    (not instead of) the configured run list, unless it is already in that list.

    The LLM extraction is run once per template since each guidance file may
    request different fields in ``extra``.  The transcript is formatted once
    and reused.

    Output files are named <template_key>.md in ctx.output_dir.
    ctx.meeting_notes is set to the notes from the first template rendered.
    """
    from vocolith.models.notes import MeetingNotes, ActionItem, Decision  # noqa: F401
    from vocolith.llm.client import build_client_from_config
    from vocolith.utils.template_renderer import render_notes

    if not ctx.transcript:
        log.warning("No transcript available — skipping note generation.")
        return ctx

    cfg = ctx.config
    client = build_client_from_config(cfg.llm)
    transcript_text = _format_transcript(ctx)

    # Terminology hint from OCR — sanitize to prevent prompt injection
    safe_terms = [
        re.sub(r"[^A-Za-z0-9_./:@#\-]", "", term)[:30]
        for term in (ctx.ocr_vocabulary or [])[:40]
        if term and len(term) >= 2
    ]
    terminology = ", ".join(safe_terms) if safe_terms else "none detected"

    # Build the ordered list of templates to run
    run_list: list[str] = list(cfg.templates.run) if cfg.templates.run else []
    if template:
        tkey = template.replace(".md.j2", "").replace(".j2", "").lower()
        if tkey not in run_list:
            run_list.insert(0, tkey)
    if not run_list:
        run_list = [cfg.templates.default]

    log.info("Generating notes for %d template(s): %s", len(run_list), run_list)

    from vocolith.utils.progress import add_task, advance_task, update_task
    notes_task = add_task(f"Notes (0/{len(run_list)} templates)", total=len(run_list))

    written: list[Path] = []
    for i, tkey in enumerate(run_list):
        tkey = tkey.replace(".md.j2", "").replace(".j2", "").lower()
        template_guidance = _TEMPLATE_GUIDANCE.get(tkey, "") \
                            or _load_guidance_sidecar(tkey, cfg)

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            template_guidance=template_guidance,
            terminology=terminology,
        )

        max_chars = cfg.llm.max_transcript_chars
        log.info("[%d/%d] %s (%.0f chars)...", i + 1, len(run_list), tkey, len(transcript_text))

        if len(transcript_text) <= max_chars:
            notes_dict = _generate_single(client, system_prompt, transcript_text)
        else:
            notes_dict = _generate_chunked(
                client, system_prompt, transcript_text,
                max_chars, cfg.llm.chunk_overlap_chars,
            )

        if cfg.llm.verify_notes and notes_dict:
            notes_dict = _verify_notes(client, notes_dict, transcript_text)

        notes = _dict_to_notes(notes_dict, ctx)

        rendered = render_notes(notes, tkey, cfg.templates.user_templates_dir)
        notes_path = ctx.output_dir / f"{tkey}.md"
        notes_path.write_text(rendered, encoding="utf-8")
        written.append(notes_path)
        update_task(notes_task,
                    description=f"Notes ({i+1}/{len(run_list)} templates — last: {tkey})")
        advance_task(notes_task)
        log.info("  -> %s", notes_path.name)

        if i == 0:
            ctx.meeting_notes = notes  # expose first template on context

    from vocolith.utils.progress import complete_task
    complete_task(notes_task)
    return ctx


# ── Internal helpers ──────────────────────────────────────────────────────────

_VERIFY_PROMPT = """You are a fact-checker for meeting notes. You will be given:
1. A transcript of a meeting
2. A set of extracted meeting notes (JSON)

Your job: verify every decision and action item against the transcript.

For each item, check:
- Is there evidence in the transcript that this was actually said or decided?
- Is the assignee correct (was this person explicitly asked to do it)?
- Is the due date correct (was this date actually mentioned)?

Rules:
- Remove any item not supported by the transcript.
- Append [?] to descriptions you are uncertain about.
- Remove invented due dates (keep only if a date was explicitly stated).
- Remove invented assignees (keep only if someone was explicitly asked).
- Do NOT add new items — only verify and clean the existing ones.
- Return the same JSON structure with corrections applied.

Transcript:
{transcript}

Notes to verify:
{notes}
"""


def _verify_notes(client, notes_dict: dict, transcript_text: str) -> dict:
    """
    Second LLM pass: remove or flag items not supported by the transcript.
    Returns the cleaned notes dict.
    """
    import json as _json

    log.info("Verifying notes against transcript...")
    try:
        verified = client.call_json(
            "You are a strict fact-checker. Return only valid JSON.",
            _VERIFY_PROMPT.format(
                transcript=transcript_text[:32000],  # enough for ~1.5h meeting; first N chars
                notes=_json.dumps(notes_dict, indent=2),
            ),
        )
        removed_actions = len(notes_dict.get("action_items", [])) - len(verified.get("action_items", []))
        removed_decisions = len(notes_dict.get("decisions", [])) - len(verified.get("decisions", []))
        if removed_actions or removed_decisions:
            log.info(
                "Verification removed %d action item(s) and %d decision(s) not found in transcript.",
                removed_actions, removed_decisions,
            )
        # Verification only checks decisions and action_items.
        # The LLM often returns extra:{} because the verify prompt does not
        # mention template-specific extra fields.  Restore the original extra
        # so custom fields (discussion_sections, qa_highlights, etc.) survive.
        if not verified.get("extra") and notes_dict.get("extra"):
            verified["extra"] = notes_dict["extra"]
            log.debug("Restored extra fields from pre-verification notes.")
        return verified
    except Exception as exc:
        log.warning("Verification pass failed (%s) — using unverified notes.", exc)
        return notes_dict


def _format_transcript(ctx: "PipelineContext") -> str:
    """Format the diarized transcript as readable text for the LLM prompt."""
    lines = []
    if not ctx.transcript:
        return ""
    for seg in ctx.transcript.segments:
        name = seg.speaker_name or seg.speaker_label or "Unknown"
        from vocolith.utils.text import format_timestamp
        ts = format_timestamp(seg.start)
        lines.append(f"[{ts}] {name}: {seg.text.strip()}")
    return "\n".join(lines)


def _generate_single(client, system_prompt: str, transcript: str) -> dict:
    """Single LLM call for short transcripts."""
    user_prompt = f"Please generate meeting notes for this transcript:\n\n{transcript}"
    try:
        return client.call_json(system_prompt, user_prompt)
    except Exception as exc:
        log.error("Note generation failed: %s", exc)
        return {}


def _generate_chunked(
    client,
    system_prompt: str,
    transcript: str,
    max_chars: int,
    overlap_chars: int,
) -> dict:
    """
    Split long transcript into overlapping chunks, generate partial notes for
    each, then merge with a final consolidation pass.
    """
    # Split at speaker boundaries (newlines) to preserve context
    chunks = _split_at_boundaries(transcript, max_chars, overlap_chars)
    log.info("Processing %d transcript chunks...", len(chunks))

    partial_notes: list[dict] = []
    for i, chunk in enumerate(chunks):
        user_prompt = (
            f"This is part {i+1}/{len(chunks)} of a longer transcript. "
            f"Extract partial meeting notes:\n\n{chunk}"
        )
        try:
            partial = client.call_json(system_prompt, user_prompt)
            partial_notes.append(partial)
        except Exception as exc:
            log.warning("Chunk %d/%d failed: %s", i+1, len(chunks), exc)

    if not partial_notes:
        return {}

    if len(partial_notes) == 1:
        return partial_notes[0]

    # Merge partial notes
    log.info("Merging %d partial note sets...", len(partial_notes))
    merge_prompt = (
        "Merge these partial meeting note JSON objects into a single coherent set. "
        "Deduplicate action items and decisions. Combine summaries into a coherent whole."
    )
    merged_text = json.dumps(partial_notes, indent=2)
    try:
        return client.call_json(merge_prompt, merged_text)
    except Exception as exc:
        log.warning("Merge failed (%s). Using first partial.", exc)
        return partial_notes[0]


def _split_at_boundaries(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into chunks at line boundaries."""
    lines = text.split("\n")
    chunks = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1
        if current_len + line_len > max_chars and current:
            chunks.append("\n".join(current))
            # Keep overlap lines for context
            overlap_lines = []
            overlap_len = 0
            for l in reversed(current):
                if overlap_len + len(l) > overlap:
                    break
                overlap_lines.insert(0, l)
                overlap_len += len(l) + 1
            current = overlap_lines
            current_len = overlap_len

        current.append(line)
        current_len += line_len

    if current:
        chunks.append("\n".join(current))

    return chunks if chunks else [text]


def _dict_to_notes(data: dict, ctx: "PipelineContext") -> "Any":  # returns MeetingNotes
    """Convert raw LLM JSON dict to MeetingNotes model."""
    from vocolith.models.notes import MeetingNotes, ActionItem, Decision

    duration_min = 0
    if ctx.transcript:
        duration_min = int(ctx.transcript.duration_seconds / 60)

    # Use LLM-extracted attendees as the primary list.
    # OCR names are only used as hints in the prompt (for diarization correlation),
    # NOT auto-added here — they contain too much noise (URLs, slide text, browser labels).
    # The LLM already sees the OCR names in its context via speaker_map / transcript.
    attendees = data.get("attendees") or []

    decisions = [
        Decision(
            description=d.get("description", ""),
            decided_by=d.get("decided_by"),
            context=d.get("context"),
        )
        for d in (data.get("decisions") or [])
        if isinstance(d, dict) and d.get("description")
    ]

    action_items = [
        ActionItem(
            description=a.get("description", ""),
            assignee=a.get("assignee"),
            due_date=a.get("due_date"),
            priority=a.get("priority") or None,
        )
        for a in (data.get("action_items") or [])
        if isinstance(a, dict) and a.get("description")
    ]

    return MeetingNotes(
        session_id=ctx.session_id,
        title=data.get("title", "Meeting Notes"),
        meeting_date=DateType.today(),
        duration_minutes=duration_min,
        attendees=attendees,
        summary=data.get("summary", ""),
        agenda_items=data.get("agenda_items") or [],
        key_topics=data.get("key_topics") or [],
        decisions=decisions,
        action_items=action_items,
        follow_up_questions=data.get("follow_up_questions") or [],
        extra=data.get("extra") or {},
    )

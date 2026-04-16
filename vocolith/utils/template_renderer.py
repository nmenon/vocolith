# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Jinja2 template rendering for meeting notes output."""
from __future__ import annotations
import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

from vocolith.models.notes import MeetingNotes

log = logging.getLogger(__name__)

# Built-in templates directory
_BUILTIN_TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"


def _is_safe_path(path: Path, parent: Path) -> bool:
    """Return True if path is within parent (prevents directory traversal)."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def get_template_dirs(user_templates_dir: str | None = None) -> list[Path]:
    """Return template search paths (user first, then built-in)."""
    dirs = []
    if user_templates_dir:
        p = Path(user_templates_dir)
        if p.is_dir():
            dirs.append(p)
    if _BUILTIN_TEMPLATES_DIR.is_dir():
        dirs.append(_BUILTIN_TEMPLATES_DIR)
    return dirs


def render_notes(
    notes: MeetingNotes,
    template_name: str = "standard",
    user_templates_dir: str | None = None,
) -> str:
    """
    Render MeetingNotes to a Markdown string using a Jinja2 template.

    Template resolution order:
      1. User templates directory (if configured)
      2. Built-in templates directory
      3. If a path is given (ends with .j2 or .md.j2), load directly

    Args:
        notes:              MeetingNotes model instance.
        template_name:      Template name ("standard", "standup", etc.) or file path.
        user_templates_dir: Optional additional directory to search.

    Returns:
        Rendered Markdown string.
    """
    # If template_name is an explicit file path — validate against allowed directories
    template_path = Path(template_name)
    if template_path.exists() and template_path.suffix == ".j2":
        resolved = template_path.resolve()
        search_dirs = get_template_dirs(user_templates_dir)
        allowed = [d.resolve() for d in search_dirs]
        # Allow any path that is within a known templates directory (no traversal)
        if not any(_is_safe_path(resolved, allowed_dir) for allowed_dir in allowed):
            # Path is outside all known template directories — block unconditionally.
            # A ".." check is insufficient: an absolute path to /tmp/evil.j2 has no
            # ".." but is still a traversal outside the allowed tree.
            log.warning("Template path outside allowed directories — blocked: %s", template_name)
            return _plain_fallback(notes)
        env = Environment(
            loader=FileSystemLoader(str(resolved.parent)),
            autoescape=select_autoescape([]),
        )
        tmpl = env.get_template(resolved.name)
        return tmpl.render(**_template_context(notes))

    # Search built-in + user dirs
    search_dirs = get_template_dirs(user_templates_dir)
    if not search_dirs:
        log.warning("No template directories found; using plain text fallback.")
        return _plain_fallback(notes)

    env = Environment(
        loader=FileSystemLoader([str(d) for d in search_dirs]),
        autoescape=select_autoescape([]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Try "standard.md.j2" then "standard.j2"
    for suffix in [".md.j2", ".j2"]:
        fname = template_name + suffix
        try:
            tmpl = env.get_template(fname)
            rendered = tmpl.render(**_template_context(notes))
            log.debug("Rendered notes using template: %s", fname)
            return rendered
        except TemplateNotFound:
            continue

    log.warning("Template '%s' not found; using plain text fallback.", template_name)
    return _plain_fallback(notes)


def _template_context(notes: MeetingNotes) -> dict:
    """Build the Jinja2 context dict from a MeetingNotes instance."""
    return {
        "title": notes.title,
        "date": notes.meeting_date.isoformat() if notes.meeting_date else "",
        "duration_minutes": notes.duration_minutes,
        "attendees": notes.attendees,
        "summary": notes.summary,
        "agenda_items": notes.agenda_items,
        "key_topics": notes.key_topics,
        "decisions": notes.decisions,
        "action_items": notes.action_items,
        "follow_up_questions": notes.follow_up_questions,
        # Template-specific structured data (standup per-person entries etc.)
        "extra": notes.extra,
        "standup": notes.extra.get("standup", {}),
        # Helpers
        "notes": notes,
    }


def _plain_fallback(notes: MeetingNotes) -> str:
    """Minimal Markdown output when no template is available."""
    lines = [
        f"# {notes.title}",
        f"**Date**: {notes.meeting_date}",
        f"**Duration**: {notes.duration_minutes} min",
        f"**Attendees**: {', '.join(notes.attendees) or 'Unknown'}",
        "",
        "## Summary",
        notes.summary or "_No summary available._",
        "",
    ]
    if notes.decisions:
        lines += ["## Decisions", ""]
        for d in notes.decisions:
            lines.append(f"- {d.description}")
        lines.append("")
    if notes.action_items:
        lines += ["## Action Items", ""]
        lines.append("| # | Description | Owner | Due |")
        lines.append("|---|---|---|---|")
        for i, item in enumerate(notes.action_items, 1):
            lines.append(
                f"| {i} | {item.description} | "
                f"{item.assignee or 'Unassigned'} | {item.due_date or ''} |"
            )
    return "\n".join(lines)

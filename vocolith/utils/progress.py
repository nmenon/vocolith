# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""
Shared Rich progress manager for the vocolith pipeline.

Stages call get_progress() to obtain the active Progress instance and
add their own tasks.  If no progress manager is active (e.g. in tests),
all calls are no-ops.

Usage in pipeline.py:
    with pipeline_progress() as progress:
        ctx = _run_stages(ctx, progress)

Usage in a stage:
    from vocolith.utils.progress import get_progress, add_task, advance_task

    task = add_task("OCR frames", total=len(frames))
    for frame in frames:
        process(frame)
        advance_task(task)
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Any

from rich.console import Console

_status_console = Console()   # standalone fallback when no progress bar active

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

# Module-level singleton — set by pipeline_progress() context manager
_active: Progress | None = None


def get_progress() -> Progress | None:
    """Return the currently active Progress instance, or None."""
    return _active


def add_task(description: str, total: float | None = None,
             visible: bool = True) -> TaskID | None:
    """Add a task to the active progress bar.  Returns None if no bar active."""
    if _active is None:
        return None
    return _active.add_task(description, total=total, visible=visible)


def advance_task(task_id: TaskID | None, advance: float = 1) -> None:
    """Advance a progress task by the given amount."""
    if _active is None or task_id is None:
        return
    _active.advance(task_id, advance)


def update_task(task_id: TaskID | None, **kwargs: Any) -> None:
    """Update arbitrary task fields (description, completed, total, …)."""
    if _active is None or task_id is None:
        return
    _active.update(task_id, **kwargs)


def status(message: str, style: str = "dim") -> None:
    """
    Print a user-facing status line that is always visible regardless of
    log level.  Rendered by the active progress console so it doesn't
    break the live progress bar display.

    Use this for key facts the user always wants to see:
      GPU / model selection, OOM retries, stage completion summaries.
    Not for debug detail — use log.info() / log.debug() for that.
    """
    if _active is not None:
        _active.console.print(f"  [dim]•[/dim] {message}", style=style)
    else:
        _status_console.print(f"  [dim]•[/dim] {message}", style=style)


def complete_task(task_id: TaskID | None, description: str | None = None) -> None:
    """Mark a task as complete (visible=False) with an optional final label."""
    if _active is None or task_id is None:
        return
    upd: dict[str, Any] = {"visible": False}
    if description:
        upd["description"] = description
    _active.update(task_id, **upd)


@contextmanager
def pipeline_progress():
    """
    Context manager that creates a Rich Progress bar for the full pipeline.

    The overall pipeline bar shows stage N/9 + elapsed only — no ETA,
    because stages have wildly different durations (extract=4s, transcribe=15min)
    making rate-based ETA extrapolation meaningless and misleading.

    Per-stage sub-task bars (diarization %, OCR frames, notes templates) DO show
    ETA because their increments have uniform cost.

    Columns shown:
      spinner | description | bar | N/M | elapsed
      (sub-tasks also show ETA when they have uniform increments)
    """
    global _active
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}", justify="left"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        # TimeRemainingColumn intentionally omitted from the overall bar:
        # stages have wildly different durations so rate-based ETA is wrong.
        # Sub-task bars (diarize, OCR, notes) use their own add_task() calls
        # which inherit these columns — they are uniform-cost so ETA is valid there.
        refresh_per_second=4,
        expand=False,
    )
    with progress:
        _active = progress
        try:
            yield progress
        finally:
            _active = None


@contextmanager
def pause_progress():
    """
    Temporarily stop the live progress display for interactive prompts.

    The spinner/bars are hidden for the duration so they don't overlay
    Rich panels and prompts.  The display resumes automatically on exit.
    No-op when no progress bar is active (tests, standalone commands).
    """
    prog = _active
    if prog is not None:
        prog.live.stop()
    try:
        yield
    finally:
        if prog is not None:
            prog.live.start(refresh=True)

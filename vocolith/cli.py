# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""CLI entry point for meeting-decoder."""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import List, Optional

import signal
import sys
import threading

import typer
from rich.console import Console
from vocolith.utils.progress import _console as console  # shared with progress bar and logging
from rich.table import Table

from vocolith.utils.logging_setup import setup_logging


# ---------------------------------------------------------------------------
# Ctrl+C handling
# ---------------------------------------------------------------------------

_interrupt_event = threading.Event()


def _sigint_handler(signum: int, frame: object) -> None:
    """Replace default SIGINT so KeyboardInterrupt is NOT raised mid-stage.

    First Ctrl+C: sets flag, prints "finishing stage" notice.
    Second Ctrl+C (while flag already set): exits immediately with code 130.
    """
    if _interrupt_event.is_set():
        # Restore default handler and re-deliver the signal — the OS then
        # raises KeyboardInterrupt through normal Python machinery, which
        # properly unwinds the stack and runs finally blocks.
        sys.stderr.write("\n\033[31m[Second Ctrl+C — exiting]\033[0m\n")
        sys.stderr.flush()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)
        return
    _interrupt_event.set()
    sys.stderr.write(
        "\n\033[33m[Ctrl+C — finishing current stage, then pausing]\033[0m\n"
    )
    sys.stderr.flush()


def _prompt_quit() -> bool:
    """Ask the user to confirm quit after Ctrl+C. Returns True if confirmed."""
    try:
        return typer.confirm("\nReally quit?", default=True)
    except (KeyboardInterrupt, EOFError):
        return True


def _check_interrupt(console: Console) -> bool:
    """Called at pipeline stage boundaries.

    Returns True  -> abort (caller should stop and return).
    Returns False -> user chose to continue; pipeline keeps running.
    """
    if not _interrupt_event.is_set():
        return False
    _interrupt_event.clear()
    if _prompt_quit():
        console.print("[red]Aborted.[/red]")
        return True
    console.print("[dim]Resuming...[/dim]")
    return False

app = typer.Typer(
    name="meeting-decoder",
    help="Transcribe meeting recordings and generate AI-powered meeting notes.",
    no_args_is_help=True,
)
profiles_app     = typer.Typer(help="Manage speaker profiles.")
templates_app    = typer.Typer(help="Manage note templates.")
mtypes_app       = typer.Typer(help="Manage meeting type aliases.")
app.add_typer(profiles_app,  name="profiles")
app.add_typer(templates_app, name="templates")
app.add_typer(mtypes_app,    name="meeting-types")

log = logging.getLogger(__name__)

# Film-strip border  |  lips  |  voice device + wave  |  ≡ notes
# ░░ = sprocket holes   ▐▌ = device bezel   ≡ = transcript lines
# Column widths (display):  frame=4  inner=14  frame=4  → total=22
# 👄 is a 2-column wide emoji; each line with one 👄 has one fewer
# space relative to its Python char count to keep display width = 22.
_BANNER_FRAME = [
    "╔══╦══════════════╦══╗",
    "║░░║              ║░░║",
    "╠══╣ 👄 [green]▐/\\/\\▌[/green]    ╠══╣",
    "║░░║    [green]▐    ▌[/green]  ≡ ║░░║",
    "╠══╣ 👄 [green]▐\\/\\/▌[/green]    ╠══╣",
    "║░░║              ║░░║",
    "╚══╩══════════════╩══╝",
]


def _print_banner() -> None:
    for line in _BANNER_FRAME:
        console.print(line, highlight=False)
    console.print("       [bold green]V O C O L I T H[/bold green]")
    console.print("   [dim]video  \u00b7  voice  \u00b7  notes[/dim]")
    console.print()


@app.command()
def process(
    video: Path = typer.Argument(..., help="Path to the video file (MP4, WebM, etc.)"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Where to write transcript.md and meeting_notes.md "
             "(default: <video_dir>/<stem>_<timestamp>/)"),
    debug_dir: Optional[Path] = typer.Option(
        None, "--debug-dir",
        help="Where to write intermediate files: WAVs, sampled frames, diarization.json "
             "(default: <output_dir>/debug/)"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c",
                                                help="Path to config.yaml"),
    model_size: Optional[str] = typer.Option(None, "--model-size",
                                              help="Whisper model: tiny|base|small|medium|large-v2|auto"),
    language: Optional[str] = typer.Option(None, "--language",
                                            help="Force language code e.g. 'en' (default: auto-detect)"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model",
                                             help="LLM model name for summarization"),
    local: bool = typer.Option(False, "--local",
                                help="Use local Ollama endpoint instead of cloud LLM"),
    local_model: Optional[str] = typer.Option(None, "--local-model",
                                               help="Local model name (default: from config, e.g. mistral:7b)"),
    local_url: Optional[str] = typer.Option(None, "--local-url",
                                             help="Local LLM base URL (default: http://localhost:11434/v1)"),
    attendees: Optional[str] = typer.Option(
        None, "--attendees", "-a",
        help="Comma-separated list of expected attendee names "
             "(e.g. 'Alice Smith, Bob Jones') — used as hints for speaker identification"),
    reference_transcript: Optional[Path] = typer.Option(
        None, "--reference-transcript",
        help="Path to a Teams-generated .vtt transcript to cross-reference for speaker accuracy "
             "(hybrid meetings: names remote attendees confidently, flags the shared room mic)"),
    room_attendees: Optional[str] = typer.Option(
        None, "--room-attendees",
        help="Comma-separated subset of --attendees sharing one conference-room mic — "
             "their diarization label is auto-routed into the split wizard"),
    room_label: Optional[str] = typer.Option(
        None, "--room-label",
        help="Exact speaker name as it appears in --reference-transcript for the shared "
             "room device, if known (skips the ambiguous-coverage heuristic)"),
    template: Optional[str] = typer.Option(None, "--template", "-t",
                                            help="Add a single extra template to this run"),
    meeting_type: Optional[str] = typer.Option(None, "--meeting-type", "-m",
                                                help="Meeting type alias from config (overrides templates.run)"),
    confirm: Optional[bool] = typer.Option(
        None, "--confirm/--no-confirm",
        help="Show identified speakers for review before writing the transcript "
             "(default: on). Voice HIGH (≥0.92) shown with green badge and fast-confirm. "
             "Use --no-confirm for batch/automated runs."),
    identify: bool = typer.Option(False, "--identify",
                                   help="After processing, run interactive wizard to name any unresolved Speaker_N speakers"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                  help="Transcribe only; skip LLM note generation"),
    no_faces: bool = typer.Option(False, "--no-faces",
                                   help="Skip face recognition (useful for audio-only)"),
    no_ocr: bool = typer.Option(False, "--no-ocr",
                                 help="Skip OCR name extraction"),
    verbose: bool = typer.Option(False, "--verbose", "-v",
                                  help="Show INFO logs (stage details)"),
    debug: bool = typer.Option(False, "--debug",
                                help="Show DEBUG logs (everything, for diagnosing failures)"),
) -> None:
    """Process a meeting recording: transcribe, identify speakers, generate notes."""
    setup_logging(verbose=verbose, debug=debug)

    if not video.exists():
        console.print(f"[red]Error:[/red] File not found: {video}")
        raise typer.Exit(1)

    from vocolith.config import load_config
    cfg = load_config(config_file)

    # CLI overrides
    if model_size:
        cfg.transcription.model_size = model_size
    if language:
        cfg.transcription.language = language
    if llm_model:
        cfg.llm.model = llm_model
    if local:
        cfg.llm.use_local = True
    if local_model:
        cfg.llm.local_model = local_model
    if local_url:
        cfg.llm.local_base_url = local_url

    # Determine output directory:
    #   1. --output-dir flag
    #   2. pipeline.output_dir in config (if set)
    #   3. Default: folder next to the input video
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        stem = video.stem.replace(" ", "_")[:40]
        if cfg.pipeline.output_dir:
            output_dir = Path(cfg.pipeline.output_dir) / f"{stem}_{ts}"
        else:
            output_dir = video.resolve().parent / f"{stem}_{ts}"

    _print_banner()
    console.print(f"Input    : {video}")
    console.print(f"Output   : {output_dir}")
    if debug_dir:
        console.print(f"Debug    : {debug_dir}")
    console.print(f"Template : {template or cfg.templates.default}")
    if dry_run:
        console.print("[yellow]Dry-run mode: LLM summarization disabled[/yellow]")
    console.print()

    # Resolve meeting type alias -> override templates.run
    if meeting_type:
        mt = cfg.meeting_types.get(meeting_type)
        if mt:
            cfg.templates.run = mt.templates
            console.print(f"Meeting type: [bold]{meeting_type}[/bold] — {mt.description}")
            console.print(f"  Templates : {', '.join(mt.templates)}")
        else:
            known = ", ".join(sorted(cfg.meeting_types.keys()))
            console.print(
                f"[yellow]Warning:[/yellow] unknown meeting type '{meeting_type}'. "
                f"Known types: {known}"
            )

    # debug_dir: --debug-dir flag > config > default (<output_dir>/debug/)
    resolved_debug_dir = debug_dir
    if resolved_debug_dir is None and cfg.pipeline.debug_dir:
        resolved_debug_dir = Path(cfg.pipeline.debug_dir)

    # --confirm / --no-confirm overrides config
    if confirm is not None:
        cfg.speaker_resolution.confirm_auto_identified = confirm

    # Parse attendees list
    attendees_list = [a.strip() for a in (attendees or "").split(",") if a.strip()]
    if attendees_list:
        console.print(f"Attendees  : {', '.join(attendees_list)}")

    # Reference transcript / room-mixed cross-referencing
    if reference_transcript and not reference_transcript.exists():
        console.print(f"[red]Error:[/red] Reference transcript not found: {reference_transcript}")
        raise typer.Exit(1)
    room_attendees_list = [a.strip() for a in (room_attendees or "").split(",") if a.strip()]
    if reference_transcript:
        console.print(f"Reference  : {reference_transcript}")
    if room_attendees_list:
        console.print(f"Room mic   : {', '.join(room_attendees_list)}")

    from vocolith.pipeline import run_pipeline
    _interrupt_event.clear()
    old_handler = signal.signal(signal.SIGINT, _sigint_handler)
    try:
        ctx = run_pipeline(
            video_path=video,
            output_dir=output_dir,
            config=cfg,
            debug_dir=resolved_debug_dir,
            attendees=attendees_list,
            dry_run=dry_run,
            template=template,
            no_faces=no_faces,
            no_ocr=no_ocr,
            interrupt_check=lambda: _check_interrupt(console),
            reference_transcript=reference_transcript,
            room_attendees=room_attendees_list,
            room_label=room_label,
        )
    finally:
        signal.signal(signal.SIGINT, old_handler)

    # Summary
    console.print("\n[bold]Results[/bold]")
    if ctx.transcript:
        console.print(f"  Segments   : {len(ctx.transcript.segments)}")
        console.print(f"  Speakers   : {ctx.transcript.speakers_detected}")
        console.print(f"  Language   : {ctx.transcript.language}")
        console.print(f"  Duration   : {ctx.transcript.duration_seconds:.0f}s")

    transcript_path = output_dir / "transcript.md"
    if transcript_path.exists():
        console.print(f"\n  [green]Transcript[/green] -> {transcript_path}")

    # List all generated note files (<template_key>.md)
    note_files = sorted(
        f for f in output_dir.glob("*.md")
        if f.name != "transcript.md"
    )
    for nf in note_files:
        console.print(f"  [green]Notes     [/green] -> {nf}")

    if ctx.errors:
        console.print(f"\n[yellow]Warnings ({len(ctx.errors)}):[/yellow]")
        for err in ctx.errors:
            prefix = "[red]ERROR[/red]" if err.fatal else "[yellow]WARN[/yellow]"
            console.print(f"  {prefix} [{err.stage}] {err.message}")

    if any(e.fatal for e in ctx.errors):
        raise typer.Exit(1)

    # Auto-run identification wizard if --identify was set
    if identify:
        from vocolith.stages.identifier_wizard import run_wizard
        run_wizard(output_dir, config=cfg)


# ─── Notes command ────────────────────────────────────────────────────────────

@app.command()
def notes(
    transcript: Path = typer.Argument(
        ..., help="Transcript file to generate notes from. "
                  "Accepts vocolith transcript.md (auto-detected) "
                  "or any plain-text transcript."),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Directory to write note files "
             "(default: timestamped subdirectory next to transcript)"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c",
                                                help="Path to config.yaml"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model",
                                             help="LLM model name for summarization"),
    local: bool = typer.Option(False, "--local",
                                help="Use local Ollama endpoint instead of cloud LLM"),
    local_model: Optional[str] = typer.Option(None, "--local-model",
                                               help="Local model name (default: from config, e.g. mistral:7b)"),
    local_url: Optional[str] = typer.Option(None, "--local-url",
                                             help="Local LLM base URL (default: http://localhost:11434/v1)"),
    meeting_type: Optional[str] = typer.Option(
        None, "--meeting-type", "-m",
        help="Meeting type alias from config (sets template list)"),
    template: Optional[List[str]] = typer.Option(
        None, "--template", "-t",
        help="Template(s) to generate — repeat for multiple: -t executive_summary -t detailed_technical_discussion_notes"),
    attendees: Optional[str] = typer.Option(
        None, "--attendees", "-a",
        help="Comma-separated attendee names — used as hints in the LLM prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v",
                                  help="Show INFO logs"),
    debug: bool = typer.Option(False, "--debug",
                                help="Show DEBUG logs"),
) -> None:
    """
    Generate meeting notes from an existing transcript file.

    Skips audio/video processing (stages 1-8) and runs only the LLM note
    generation stage.  Accepts the transcript.md produced by
    ``vocolith process`` as well as any plain-text transcript.

    Example:
        vocolith notes transcript.md -m woody -a 'Alice(Acme), Bob(Woody)'
    """
    setup_logging(verbose=verbose, debug=debug)

    if not transcript.exists():
        console.print(f"[red]Error:[/red] File not found: {transcript}")
        raise typer.Exit(1)

    from vocolith.config import load_config
    cfg = load_config(config_file)

    if llm_model:
        cfg.llm.model = llm_model
    if local:
        cfg.llm.use_local = True
    if local_model:
        cfg.llm.local_model = local_model
    if local_url:
        cfg.llm.local_base_url = local_url

    # Apply meeting type overrides (same logic as process command)
    if meeting_type:
        mtype = cfg.meeting_types.get(meeting_type)
        if mtype:
            cfg.templates.run = mtype.templates
            console.print(
                f"Meeting type: [bold]{meeting_type}[/bold] — {mtype.description}\n"
                f"  Templates : {', '.join(mtype.templates)}"
            )
        else:
            console.print(f"[yellow]Warning:[/yellow] Meeting type '{meeting_type}' not found in config.")
    elif template:
        # --template with no --meeting-type: run ONLY the requested template(s).
        cfg.templates.run = [
            t.replace(".md.j2", "").replace(".j2", "").lower() for t in template
        ]

    # Resolve output directory
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_output = output_dir or (transcript.parent / f"{transcript.stem}_notes_{ts}")
    resolved_output.mkdir(parents=True, exist_ok=True)

    attendees_list = [a.strip() for a in (attendees or "").split(",") if a.strip()]

    console.print(f"\n[bold]Vocolith Notes[/bold]")
    console.print(f"Input      : {transcript}")
    console.print(f"Output     : {resolved_output}")
    if attendees_list:
        console.print(f"Attendees  : {', '.join(attendees_list)}")

    # Parse transcript into a DiarizedTranscript
    from vocolith.utils.transcript_reader import load_transcript_file
    diarized = load_transcript_file(transcript)
    console.print(
        f"Transcript : {len(diarized.segments)} segment(s), "
        f"{diarized.speakers_detected} speaker(s) detected"
        + (f", {diarized.duration_seconds:.0f}s" if diarized.duration_seconds else "")
    )

    # Build a minimal PipelineContext for note generation
    from vocolith.pipeline import PipelineContext
    ctx = PipelineContext(
        video_path=transcript,   # used only for metadata / session label
        output_dir=resolved_output,
        debug_dir=resolved_output / "debug",
        config=cfg,
        transcript=diarized,
        attendees=attendees_list,
    )

    # Run note generation (Stage 9 only)
    from vocolith.stages.note_generator import generate_notes
    from vocolith.utils.progress import pipeline_progress, pause_progress
    with pipeline_progress():
        while True:
            try:
                ctx = generate_notes(ctx, template=None)
                break
            except typer.Exit:
                raise
            except Exception as exc:
                console.print(f"[red]Note generation failed:[/red] {exc}")
                raise typer.Exit(1)
            except KeyboardInterrupt:
                with pause_progress():
                    console.print("\n[yellow]Note generation interrupted.[/yellow]")
                    if _prompt_quit():
                        console.print("[red]Aborted.[/red]")
                        raise typer.Exit(130)
                console.print("[dim]Retrying...[/dim]")

    # Summary
    note_files = sorted(
        f for f in resolved_output.glob("*.md")
        if f.name != "transcript.md"
    )
    if note_files:
        console.print("\n[bold]Results[/bold]")
        for nf in note_files:
            console.print(f"  [green]Notes[/green] -> {nf}")
    else:
        console.print("[yellow]No notes generated.[/yellow]")


# ─── Identify command ─────────────────────────────────────────────────────────

@app.command()
def identify(
    output_dir: Path = typer.Argument(
        ..., help="Output directory from a previous `vocolith process` run"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """
    Interactively identify unknown Speaker_N speakers from a previous run.

    Shows transcript snippets, plays audio clips, and optionally shows video
    frames to help you name each unidentified speaker. Names are saved to
    the speaker profile database and the transcript.md is updated in-place.
    """
    setup_logging()
    if not output_dir.exists():
        console.print(f"[red]Directory not found:[/red] {output_dir}")
        raise typer.Exit(1)

    from vocolith.config import load_config
    from vocolith.stages.identifier_wizard import run_wizard
    cfg = load_config(config_file)
    while True:
        try:
            run_wizard(output_dir, config=cfg)
            break
        except typer.Exit:
            raise
        except Exception as exc:
            console.print(f"[red]Wizard error:[/red] {exc}")
            raise typer.Exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Wizard interrupted.[/yellow]")
            if _prompt_quit():
                console.print("[red]Aborted.[/red]")
                raise typer.Exit(130)
            console.print("[dim]Restarting wizard...[/dim]")


# ─── Profiles subcommands ────────────────────────────────────────────────────

@profiles_app.command("list")
def profiles_list(
    config_file: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """List all known speaker profiles."""
    setup_logging()
    from vocolith.config import load_config
    cfg = load_config(config_file)
    db_path = Path(cfg.storage.profiles_dir) / cfg.storage.db_filename

    if not db_path.exists():
        console.print("[dim]No speaker profiles found.[/dim]")
        return

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT display_name, meeting_count, last_seen_at FROM speakers ORDER BY meeting_count DESC, display_name ASC"
    ).fetchall()
    conn.close()

    if not rows:
        console.print("[dim]No speaker profiles found.[/dim]")
        return

    table = Table("Name", "Meetings", "Last Seen")
    for name, count, last_seen in rows:
        table.add_row(name, str(count), (last_seen or "")[:10])
    console.print(table)


@profiles_app.command("play")
def profiles_play(
    name: str = typer.Argument(..., help="Speaker display name to play"),
    config_file: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Play a stored voice sample for a speaker profile."""
    setup_logging()
    from vocolith.config import load_config
    cfg = load_config(config_file)
    db_path = Path(cfg.storage.profiles_dir) / cfg.storage.db_filename

    if not db_path.exists():
        console.print("[red]No speaker profiles found.[/red]")
        raise typer.Exit(1)

    from vocolith.storage.db import init_db
    from vocolith.storage.speaker_store import SpeakerStore

    conn = init_db(db_path)
    store = SpeakerStore(conn)

    profile = store.find_by_name(name) or store.find_by_alias(name)
    if not profile:
        console.print(f"[red]Speaker '{name}' not found.[/red]")
        raise typer.Exit(1)

    wav_bytes = store.get_sample(profile.speaker_id)
    if not wav_bytes:
        console.print(
            f"[yellow]No audio sample stored for '{name}'.[/yellow]\n"
            "[dim]Samples are recorded when you confirm a speaker in the wizard. "
            "Re-process a meeting to capture one.[/dim]"
        )
        raise typer.Exit(1)

    import tempfile, subprocess
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, prefix="vocolith_") as f:
        f.write(wav_bytes)
        tmp_path = f.name

    console.print(f"Playing voice sample for [bold]{name}[/bold] …")
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
            check=False,
        )
    except FileNotFoundError:
        console.print(
            f"[yellow]ffplay not found.[/yellow] Sample saved to: [dim]{tmp_path}[/dim]\n"
            "[dim]Install ffmpeg to enable playback.[/dim]"
        )
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


@profiles_app.command("delete")
def profiles_delete(
    name: str = typer.Argument(..., help="Speaker display name to delete"),
    config_file: Optional[Path] = typer.Option(None, "--config"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Delete a speaker profile and all their embeddings."""
    setup_logging()
    if not yes:
        confirmed = typer.confirm(f"Delete speaker '{name}' and all their data?")
        if not confirmed:
            raise typer.Abort()

    from vocolith.config import load_config
    cfg = load_config(config_file)
    _delete_speaker(cfg, name)
    console.print(f"[green]Deleted speaker: {name}[/green]")


@profiles_app.command("rename")
def profiles_rename(
    old_name: str = typer.Argument(...),
    new_name: str = typer.Argument(...),
    config_file: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Rename a speaker (e.g. 'Speaker_00' -> 'Alice Smith')."""
    setup_logging()
    from vocolith.config import load_config
    import sqlite3

    cfg = load_config(config_file)
    db_path = Path(cfg.storage.profiles_dir) / cfg.storage.db_filename

    if not db_path.exists():
        console.print("[red]No profiles database found.[/red]")
        raise typer.Exit(1)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "UPDATE speakers SET display_name=? WHERE display_name=?",
            (new_name, old_name)
        )
        conn.commit()
    finally:
        conn.close()

    if rows.rowcount == 0:
        console.print(f"[red]Speaker '{old_name}' not found.[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Renamed '{old_name}' -> '{new_name}'[/green]")


@profiles_app.command("check")
def profiles_check(
    config_file: Optional[Path] = typer.Option(None, "--config"),
    warn_threshold: float = typer.Option(0.89, "--warn", help="Similarity above this is a warning (STD threshold)"),
    fail_threshold: float = typer.Option(0.92, "--fail", help="Similarity above this is a false-positive risk (HIGH threshold)"),
) -> None:
    """Check voice embedding consistency — pairwise similarities and orphan profiles."""
    setup_logging()
    from vocolith.config import load_config
    import sqlite3
    cfg = load_config(config_file)
    db_path = Path(cfg.storage.profiles_dir) / cfg.storage.db_filename
    chroma_path = Path(cfg.storage.profiles_dir) / cfg.storage.chroma_dir

    if not db_path.exists():
        console.print("[dim]No speaker profiles found.[/dim]")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    speakers = conn.execute(
        "SELECT speaker_id, display_name, meeting_count FROM speakers ORDER BY display_name"
    ).fetchall()
    conn.close()

    if not speakers:
        console.print("[dim]No speaker profiles found.[/dim]")
        return

    # ── Profile health table ──────────────────────────────────────────────────
    console.print("\n[bold]Profile health[/bold]")
    health_table = Table("Name", "Meetings", "Status")
    issues: list[str] = []
    warn_issues_set: set[str] = set()   # issues that are warnings not errors
    for s in speakers:
        if s["meeting_count"] == 0:
            health_table.add_row(
                s["display_name"], str(s["meeting_count"]),
                "[yellow]⚠ no confirmed meetings[/yellow]",
            )
            issues.append(f"'{s['display_name']}' has meeting_count=0 — never confirmed")
        else:
            health_table.add_row(
                s["display_name"], str(s["meeting_count"]), "[green]OK[/green]",
            )
    console.print(health_table)

    # ── ChromaDB embedding check ──────────────────────────────────────────────
    try:
        import chromadb as _chroma
        import numpy as np
        from collections import defaultdict

        client = _chroma.PersistentClient(path=str(chroma_path))
        try:
            col = client.get_collection("voice_embeddings")
        except Exception:
            console.print("[dim]No voice embeddings stored yet.[/dim]")
            return
        all_stored = col.get(include=["embeddings", "metadatas"])
    except Exception as exc:
        console.print(f"[red]Could not load ChromaDB: {exc}[/red]")
        raise typer.Exit(1)

    metadatas  = all_stored.get("metadatas") or []
    raw_embs   = all_stored.get("embeddings")
    embeddings = raw_embs if raw_embs is not None else []
    if not metadatas or (hasattr(embeddings, "__len__") and len(embeddings) == 0):
        console.print("[dim]No voice embeddings stored yet.[/dim]")
        return

    id_to_name = {s["speaker_id"]: s["display_name"] for s in speakers}

    # Average embeddings per speaker across sessions
    embs_by_name: dict[str, list] = defaultdict(list)
    if len(metadatas) != len(embeddings):
        console.print(
            f"[yellow]⚠ ChromaDB length mismatch: {len(metadatas)} metadata vs "
            f"{len(embeddings)} embeddings — DB may be corrupt[/yellow]"
        )
    for meta, emb in zip(metadatas, embeddings):
        name = id_to_name.get(meta["speaker_id"], "?")
        embs_by_name[name].append(np.array(emb))

    # Build normalised mean per speaker
    avg_embs: dict[str, np.ndarray] = {}
    for name, vecs in embs_by_name.items():
        mean = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            avg_embs[name] = mean / norm

    if not avg_embs:
        console.print("[dim]No voice embeddings found.[/dim]")
        return

    # ── Embedding density check ───────────────────────────────────────────────
    # Infer embedding dimension from the data (don't hardcode 256)
    sample_dim = len(next(iter(avg_embs.values())))
    sparse_threshold = int(sample_dim * 0.55)   # < 55% nonzero = likely degenerate

    console.print(f"\n[bold]Voice embeddings[/bold]  ({len(metadatas)} stored, "
                  f"{len(avg_embs)} unique speakers, dim={sample_dim})")
    density_table = Table("Name", "Sessions", f"Nonzero/{sample_dim}", "Status")
    for name in sorted(avg_embs):
        n_sessions = len(embs_by_name[name])
        nz = int(np.count_nonzero(avg_embs[name]))
        status = "[green]OK[/green]" if nz >= sparse_threshold else "[red]sparse — may be degenerate[/red]"
        if nz < sparse_threshold:
            issues.append(
                f"'{name}' sparse embedding ({nz}/{sample_dim}) — "
                f"remedy: vocolith profiles delete '{name}' then re-enroll"
            )
        density_table.add_row(name, str(n_sessions), str(nz), status)
    console.print(density_table)

    # ── Pairwise similarity matrix ────────────────────────────────────────────
    names = sorted(avg_embs)
    fp_high: list[tuple] = []
    fp_warn: list[tuple] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = avg_embs[names[i]], avg_embs[names[j]]
            sim = float(np.dot(a, b))
            if sim >= fail_threshold:
                fp_high.append((sim, names[i], names[j]))
            elif sim >= warn_threshold:
                fp_warn.append((sim, names[i], names[j]))

    if fp_high or fp_warn:
        console.print(f"\n[bold]Pairwise similarity warnings[/bold]")
        sim_table = Table("Speaker A", "Speaker B", "Similarity", "Risk")
        for sim, a, b in sorted(fp_high, reverse=True):
            sim_table.add_row(a, b, f"{sim:.4f}",
                              f"[bold red]✗ FP risk (>={fail_threshold})[/bold red]")
            issues.append(
                f"'{a}' vs '{b}' sim={sim:.4f} — above HIGH threshold; "
                f"remedy: delete the profile with fewer confirmed meetings and re-enroll"
            )
        for sim, a, b in sorted(fp_warn, reverse=True):
            margin = fail_threshold - sim
            sim_table.add_row(a, b, f"{sim:.4f}",
                              f"[yellow]⚠ warn (margin {margin:.4f})[/yellow]")
            # Warn pairs are listed separately from errors — tag them
            msg = (f"'{a}' vs '{b}' sim={sim:.4f} — above STD threshold; "
                   f"watch for STD-confidence false matches (no auto-accept risk)")
            issues.append(msg)
            warn_issues_set.add(msg)
        console.print(sim_table)
    else:
        console.print(
            f"\n[green]✓ All {len(names)*(len(names)-1)//2} pairwise similarities "
            f"below warn threshold ({warn_threshold})[/green]"
        )

    # ── Remediation recommendations ───────────────────────────────────────────
    remediation: list[str] = []
    # Which profiles to delete for each FP pair (fewest confirmed meetings = weaker evidence)
    conn2 = sqlite3.connect(str(db_path))
    conn2.row_factory = sqlite3.Row
    count_map = {r["display_name"]: r["meeting_count"]
                 for r in conn2.execute("SELECT display_name, meeting_count FROM speakers").fetchall()}
    conn2.close()

    for sim, a, b in fp_high:
        ca, cb = count_map.get(a, 0), count_map.get(b, 0)
        if ca == cb:
            # Tie: flag both; user must decide which is the contaminated one
            remediation.append(
                f"  [red]FP {sim:.4f}[/red]  '{a}' ↔ '{b}'\n"
                f"    → Both have {ca} confirmed meeting(s) — inspect which profile\n"
                f"      has contaminated voice data, then:\n"
                f"      [bold]vocolith profiles delete \"<contaminated>\"[/bold]\n"
                f"    → Re-enroll the deleted profile in the next meeting run"
            )
        else:
            weaker   = a if ca < cb else b
            stronger = b if weaker == a else a
            remediation.append(
                f"  [red]FP {sim:.4f}[/red]  '{a}' ↔ '{b}'\n"
                f"    → Delete weaker profile: [bold]vocolith profiles delete \"{weaker}\"[/bold]\n"
                f"      ('{weaker}' has {min(ca,cb)} confirmed meeting(s) vs "
                f"{max(ca,cb)} for '{stronger}')\n"
                f"    → Re-enroll '{weaker}' in next meeting run"
            )
    for sim, a, b in sorted(fp_warn, reverse=True)[:3]:   # top 3 warns only
        remediation.append(
            f"  [yellow]WARN {sim:.4f}[/yellow]  '{a}' ↔ '{b}'\n"
            f"    → No immediate action needed; correct in wizard if offered as wrong match\n"
            f"    → More confirmed sessions will improve embedding quality and separation"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    has_errors = bool(fp_high or any("sparse" in i for i in issues)
                      or any("meeting_count=0" in i for i in issues))
    has_warnings = bool(fp_warn)

    if has_errors or has_warnings:
        error_issues = [i for i in issues if i not in warn_issues_set]
        warn_issues  = [i for i in issues if i in warn_issues_set]

        if error_issues:
            console.print(f"[bold red]ERRORS ({len(error_issues)}):[/bold red]")
            for issue in error_issues:
                console.print(f"  [red]•[/red] {issue}")
        if warn_issues:
            console.print(f"\n[bold yellow]WARNINGS ({len(warn_issues)}):[/bold yellow]")
            for issue in warn_issues:
                console.print(f"  [yellow]•[/yellow] {issue}")
        if remediation:
            console.print(f"\n[bold]Recommended remediation:[/bold]")
            for r in remediation:
                console.print(r)
        if has_errors:
            raise typer.Exit(1)
    else:
        console.print("[bold green]✓ All checks passed — database is consistent.[/bold green]")


@profiles_app.command("clear")
def profiles_clear(
    config_file: Optional[Path] = typer.Option(None, "--config"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Delete ALL speaker profiles and embeddings."""
    setup_logging()
    if not yes:
        confirmed = typer.confirm("Delete ALL speaker profiles? This cannot be undone.")
        if not confirmed:
            raise typer.Abort()

    from vocolith.config import load_config
    import shutil

    cfg = load_config(config_file)
    profiles_dir = Path(cfg.storage.profiles_dir)

    if profiles_dir.exists():
        shutil.rmtree(profiles_dir)
        profiles_dir.mkdir(parents=True, exist_ok=True)
        console.print("[green]All profiles cleared.[/green]")
    else:
        console.print("[dim]Nothing to clear.[/dim]")


# ─── Templates subcommands ───────────────────────────────────────────────────

@templates_app.command("list")
def templates_list(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """List available note templates and their locations."""
    setup_logging()
    from vocolith.config import load_config

    cfg = load_config(config_file)
    builtin_dir = Path(__file__).parent.parent / "templates"
    user_dir = Path(cfg.templates.user_templates_dir) \
               if cfg.templates.user_templates_dir else None

    # Collect templates: user dir takes precedence over built-in
    seen: dict[str, tuple[Path, str]] = {}  # stem -> (path, source_label)
    for directory, label in [
        (builtin_dir, "built-in"),
        (user_dir,    "user"),
    ]:
        if directory and directory.exists():
            for f in sorted(directory.glob("*.md.j2")):
                stem = f.stem.replace(".md", "")
                seen[stem] = (f, label)

    if not seen:
        console.print("[dim]No templates found.[/dim]")
        return

    table = Table("Template", "Source", "File")
    for stem, (path, label) in sorted(seen.items()):
        table.add_row(stem, label, str(path))
    console.print(table)

    console.print()
    console.print(f"[dim]Built-in dir : {builtin_dir}[/dim]")
    console.print(f"[dim]User dir     : {user_dir or '(not set)'}[/dim]")
    console.print(f"[dim]  └─ configured via templates.user_templates_dir in config.yaml[/dim]")
    console.print(f"[dim]Active default: [bold]{cfg.templates.default}[/bold][/dim]")


# ─── Meeting types subcommands ───────────────────────────────────────────────

@mtypes_app.command("list")
def mtypes_list(
    config_file: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """List configured meeting type aliases and the templates they generate."""
    setup_logging()
    from vocolith.config import load_config
    cfg = load_config(config_file)

    if not cfg.meeting_types:
        console.print("[dim]No meeting types configured.[/dim]")
        return

    table = Table("Alias", "Templates", "Description")
    for name, mt in sorted(cfg.meeting_types.items()):
        table.add_row(name, "\n".join(mt.templates), mt.description)
    console.print(table)
    console.print()
    console.print("[dim]Use with: vocolith process meeting.mp4 --meeting-type <alias>[/dim]")
    console.print("[dim]Define your own in the [meeting_types] section of config.yaml[/dim]")


# ─── Capture helpers ──────────────────────────────────────────────────────────

def _resolve_screen_device(spec: str, devices: list):  # -> CaptureDevice | None
    """Resolve a user-supplied screen specifier to a CaptureDevice.

    Matching order (first match wins):
    1. Exact device ID (e.g. ``:0.0+3440,0``)
    2. Monitor index (``0``, ``1``, …) — maps to the Nth screen device
    3. Case-insensitive name substring (e.g. ``DP-0``, ``dp-2``, ``full``)

    Returns None when no device matches.
    """
    screens = [d for d in devices if d.kind == "screen"]

    # 1. Exact ID match
    for d in screens:
        if d.id == spec:
            return d

    # 2. Integer index into screen list
    if spec.isdigit():
        idx = int(spec)
        if 0 <= idx < len(screens):
            return screens[idx]

    # 3. Case-insensitive substring of device name
    spec_lower = spec.lower()
    for d in screens:
        if spec_lower in d.name.lower():
            return d

    return None


# ─── Capture command ──────────────────────────────────────────────────────────

@app.command()
def capture(
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o",
        help="Directory for the recording and all generated notes "
             "(default: timestamped folder next to where you run vocolith)"),
    audio: Optional[str] = typer.Option(
        None, "--audio",
        help="Audio device ID to record from (default: auto-select system loopback). "
             "Run --list-devices to see available IDs."),
    screen: bool = typer.Option(
        False, "--screen/--no-screen",
        help="Also capture the screen (enables OCR name detection in post-processing). "
             "Default: off (audio-only)."),
    screen_device: Optional[str] = typer.Option(
        None, "--screen-device",
        help="Screen device ID override (default: auto-detect from $DISPLAY / $WAYLAND_DISPLAY)"),
    list_devices: bool = typer.Option(
        False, "--list-devices", "-l",
        help="List available audio and screen capture devices, then exit."),
    duration: Optional[str] = typer.Option(
        None, "--duration", "-d",
        help="Maximum recording duration, e.g. '90m', '1h30m', '3600s'. "
             "Stops automatically when reached. Ctrl+C always works too."),
    no_process: bool = typer.Option(
        False, "--no-process",
        help="Record only; skip transcript and note generation."),
    # ── post-processing pass-through (same as vocolith process) ───────────────
    config_file: Optional[Path] = typer.Option(None, "--config", "-c",
                                                help="Path to config.yaml"),
    model_size: Optional[str] = typer.Option(None, "--model-size",
                                              help="Whisper model: tiny|base|small|medium|large-v2|auto"),
    language: Optional[str] = typer.Option(None, "--language",
                                            help="Force language code e.g. 'en'"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model",
                                             help="LLM model for summarization"),
    local: bool = typer.Option(False, "--local", help="Use local Ollama instead of cloud LLM"),
    local_model: Optional[str] = typer.Option(None, "--local-model"),
    local_url: Optional[str] = typer.Option(None, "--local-url"),
    attendees: Optional[str] = typer.Option(
        None, "--attendees", "-a",
        help="Comma-separated attendee names used as speaker ID hints"),
    reference_transcript: Optional[Path] = typer.Option(
        None, "--reference-transcript",
        help="Path to a Teams-generated .vtt transcript to cross-reference for speaker accuracy"),
    room_attendees: Optional[str] = typer.Option(
        None, "--room-attendees",
        help="Comma-separated subset of --attendees sharing one conference-room mic"),
    room_label: Optional[str] = typer.Option(
        None, "--room-label",
        help="Exact speaker name as it appears in --reference-transcript for the shared room device"),
    template: Optional[str] = typer.Option(None, "--template", "-t",
                                            help="Extra note template to generate"),
    meeting_type: Optional[str] = typer.Option(None, "--meeting-type", "-m",
                                                help="Meeting type alias from config"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Transcribe only; skip LLM notes"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show INFO logs"),
    debug_flag: bool = typer.Option(False, "--debug", help="Show DEBUG logs"),
) -> None:
    """Record a meeting in real-time, then automatically transcribe and generate notes.

    Uses ffmpeg to capture audio (and optionally screen) from the current machine.
    When you stop recording (Ctrl+C or --duration reached), vocolith automatically
    runs the full pipeline on the captured file — the same as running
    'vocolith process' on a pre-recorded video.

    \b
    Quick start:
        vocolith capture                           # record system audio, auto-process
        vocolith capture --screen                  # + screen for OCR/face detection
        vocolith capture --list-devices            # see what's available
        vocolith capture --duration 90m            # auto-stop after 90 minutes
        vocolith capture --audio default.monitor   # explicit device

    \b
    OS support:
        Linux   PulseAudio/PipeWire  (system loopback auto-detected)
                X11/XWayland screen capture via x11grab
                Pure-Wayland screen capture via PipeWire (portal dialog required)
        macOS   AVFoundation — system audio needs BlackHole or Loopback installed
        Windows WASAPI loopback (built-in) + GDI screen grab
    """
    setup_logging(verbose=verbose, debug=debug_flag)

    from vocolith.capture.devices import discover_devices, get_default_audio, get_default_screen

    # ── --list-devices: show table and exit ────────────────────────────────────
    if list_devices:
        try:
            devs = discover_devices()
        except RuntimeError as exc:
            console.print(f"[red]Device discovery error:[/red] {exc}")
            raise typer.Exit(1)

        from rich.table import Table as RTable
        tbl = RTable("Kind", "Format", "Device ID", "Name")
        for d in devs:
            loopback = " [dim](loopback)[/dim]" if d.is_loopback else ""
            tbl.add_row(d.kind, d.fmt, d.id, f"{d.name}{loopback}")
        console.print(tbl)
        console.print()
        console.print("[dim]Use --audio <Device ID> or --screen-device <Device ID> to override.[/dim]")
        raise typer.Exit(0)

    # ── Resolve devices ────────────────────────────────────────────────────────
    try:
        all_devices = discover_devices()
    except RuntimeError as exc:
        console.print(f"[red]Device discovery error:[/red] {exc}")
        raise typer.Exit(1)

    # Audio
    if audio:
        from vocolith.capture.devices import CaptureDevice as _CD
        audio_dev = next((d for d in all_devices if d.id == audio), None)
        if audio_dev is None:
            # User specified a raw device ID not in the discovered list — trust them.
            # Infer format from platform.
            import platform as _pl
            _os = _pl.system()
            _fmt = {"Linux": "pulse", "Darwin": "avfoundation", "Windows": "wasapi"}.get(_os, "pulse")
            audio_dev = _CD(id=audio, name=audio, kind="audio_mic", fmt=_fmt)
    else:
        audio_dev = get_default_audio(all_devices)

    if audio_dev is None:
        console.print(
            "[red]No audio capture device found.[/red]\n"
            "[dim]Run 'vocolith capture --list-devices' to see what's available.\n"
            "On macOS, system audio requires BlackHole (https://existential.audio/blackhole/)[/dim]"
        )
        raise typer.Exit(1)

    # Screen (optional)
    screen_dev = None
    if screen:
        if screen_device:
            # Accept: exact device ID  OR  monitor name substring (e.g. "DP-0", "1").
            screen_dev = _resolve_screen_device(screen_device, all_devices)
            if screen_dev is None:
                console.print(
                    f"[red]Unknown screen device:[/red] {screen_device!r}\n"
                    "[dim]Run 'vocolith capture --list-devices' to see valid IDs and names.[/dim]"
                )
                raise typer.Exit(1)
        else:
            screen_dev = get_default_screen(all_devices)
        if screen_dev is None:
            console.print("[yellow]Warning:[/yellow] no screen device found — recording audio only.")
        elif "pipewire" in screen_dev.fmt.lower():
            console.print(
                "[yellow]Note:[/yellow] Wayland screen capture requires the PipeWire portal.\n"
                "[dim]A system dialog will appear asking you to select a window/screen to share.[/dim]"
            )

    # ── Resolve output directory ───────────────────────────────────────────────
    from datetime import datetime
    from vocolith.config import load_config
    cfg = load_config(config_file)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        if cfg.pipeline.output_dir:
            output_dir = Path(cfg.pipeline.output_dir) / f"capture_{ts}"
        else:
            output_dir = Path.cwd() / f"capture_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ext = ".mkv" if screen_dev else ".wav"
    recording_path = output_dir / f"recording{ext}"

    # ── Parse duration ─────────────────────────────────────────────────────────
    max_secs: Optional[float] = None
    if duration:
        from vocolith.capture.recorder import parse_duration
        try:
            max_secs = parse_duration(duration)
        except ValueError as exc:
            console.print(f"[red]Invalid --duration:[/red] {exc}")
            raise typer.Exit(1)

    # ── Banner + recording notice ──────────────────────────────────────────────
    _print_banner()
    console.print(f"[bold green]Recording[/bold green]  →  {recording_path}")
    console.print(f"  Audio  : [cyan]{audio_dev.name}[/cyan]")
    if screen_dev:
        console.print(f"  Screen : [cyan]{screen_dev.name}[/cyan]")
    else:
        console.print(f"  Screen : [dim]off (audio-only)[/dim]")
    if max_secs:
        h = int(max_secs // 3600)
        m = int((max_secs % 3600) // 60)
        s = int(max_secs % 60)
        limit_str = f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"
        console.print(f"  Limit  : [yellow]{limit_str}[/yellow]")
    _is_tty = sys.stdin.isatty()
    _stop_hint = "  [q/Enter] stop  [Ctrl+C] abort" if _is_tty else "  Ctrl+C to stop"
    console.print()
    console.print(f"[bold]Recording…[/bold]  [dim]{_stop_hint}[/dim]")
    console.print()

    # ── Start recording ────────────────────────────────────────────────────────
    from vocolith.capture.recorder import Recorder
    recorder = Recorder()
    recorder.start(recording_path, audio_dev, screen_dev, max_secs)

    # Intercept Ctrl+C to signal a clean stop (not a process kill).
    import signal as _signal
    _stop_requested = threading.Event()

    def _capture_sigint(_sig: int, _frm: object) -> None:
        if _stop_requested.is_set():
            sys.stderr.write("\n\033[31m[Force-stopping]\033[0m\n")
            sys.stderr.flush()
            raise SystemExit(130)
        _stop_requested.set()
        # Write directly so the \r status line is not mangled by Rich.
        sys.stderr.write("\n")
        sys.stderr.flush()

    old_handler = _signal.signal(_signal.SIGINT, _capture_sigint)

    # ── Interactive recording loop ─────────────────────────────────────────────
    # On a real TTY we put stdin in cbreak mode so single keypresses are
    # delivered immediately without pressing Enter, and without echoing them.
    # On a non-TTY (pipe, script) we fall back to Ctrl+C only.
    import time as _time

    _STOP_KEYS = frozenset(("q", "Q", "s", "S", "\r", "\n", " "))

    def _fmt_elapsed(secs: float) -> str:
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    # Import platform-specific modules up front so they're always bound.
    _termios = None
    _select = None
    _orig_tty = None

    try:
        if _is_tty:
            import tty as _tty_mod
            import termios as _termios
            import select as _select
            _orig_tty = _termios.tcgetattr(sys.stdin)
            _tty_mod.setcbreak(sys.stdin.fileno())  # single-char, no echo, Ctrl+C still works

        while recorder.is_running() and not _stop_requested.is_set():
            label = _fmt_elapsed(recorder.elapsed_seconds())
            # Overwrite the same line with \r — bypass Rich to keep it predictable.
            # Use ANSI green ● only on a real TTY (where colours render).
            dot = "\033[32m●\033[0m" if _is_tty else "●"
            sys.stdout.write(f"\r  {dot} REC {label}   ")
            sys.stdout.flush()

            if _is_tty and _select is not None:
                # Poll stdin for up to 1 s, then redraw the counter.
                rlist, _, _ = _select.select([sys.stdin], [], [], 1.0)
                if rlist:
                    key = sys.stdin.read(1)
                    if key in _STOP_KEYS:
                        _stop_requested.set()
                        break
            else:
                _time.sleep(1.0)

    finally:
        # Always restore terminal and SIGINT handler.
        if _orig_tty is not None and _termios is not None:
            _termios.tcsetattr(sys.stdin, _termios.TCSADRAIN, _orig_tty)
        _signal.signal(_signal.SIGINT, old_handler)

    sys.stdout.write("\n")  # move past the \r status line
    sys.stdout.flush()

    result = recorder.stop()

    if not result.output_path.exists() or result.output_path.stat().st_size < 4096:
        console.print(
            f"[red]Recording failed or produced an empty file:[/red] {result.output_path}\n"
            "[dim]Check that the audio device is active and not muted.[/dim]"
        )
        raise typer.Exit(1)

    dur_m = int(result.duration_seconds) // 60
    dur_s = int(result.duration_seconds) % 60
    console.print(f"[green]Recorded {dur_m}m{dur_s:02d}s[/green]  →  {result.output_path}")
    console.print(f"  Size: {result.output_path.stat().st_size / 1024 / 1024:.1f} MB")

    if no_process:
        console.print("[dim]--no-process: skipping transcription.[/dim]")
        raise typer.Exit(0)

    # ── Auto-process: hand the recording to the existing pipeline ──────────────
    console.print()
    console.print("[bold]Processing recording...[/bold]")
    console.print()

    # Apply CLI overrides to config
    if model_size:
        cfg.transcription.model_size = model_size
    if language:
        cfg.transcription.language = language
    if llm_model:
        cfg.llm.model = llm_model
    if local:
        cfg.llm.use_local = True
    if local_model:
        cfg.llm.local_model = local_model
    if local_url:
        cfg.llm.local_base_url = local_url
    if meeting_type:
        mt = cfg.meeting_types.get(meeting_type)
        if mt:
            cfg.templates.run = mt.templates

    resolved_debug_dir = Path(cfg.pipeline.debug_dir) if cfg.pipeline.debug_dir else None
    attendees_list = [a.strip() for a in (attendees or "").split(",") if a.strip()]

    if reference_transcript and not reference_transcript.exists():
        console.print(f"[red]Error:[/red] Reference transcript not found: {reference_transcript}")
        raise typer.Exit(1)
    room_attendees_list = [a.strip() for a in (room_attendees or "").split(",") if a.strip()]

    # For audio-only recording, skip the OCR and face stages (no video frames).
    skip_ocr = not result.has_video
    skip_faces = not result.has_video

    _interrupt_event.clear()
    old_proc_handler = _signal.signal(_signal.SIGINT, _sigint_handler)
    try:
        from vocolith.pipeline import run_pipeline
        ctx = run_pipeline(
            video_path=result.output_path,
            output_dir=output_dir,
            config=cfg,
            debug_dir=resolved_debug_dir,
            attendees=attendees_list,
            dry_run=dry_run,
            template=template,
            no_faces=skip_faces,
            no_ocr=skip_ocr,
            interrupt_check=lambda: _check_interrupt(console),
            reference_transcript=reference_transcript,
            room_attendees=room_attendees_list,
            room_label=room_label,
        )
    finally:
        _signal.signal(_signal.SIGINT, old_proc_handler)

    # ── Summary ────────────────────────────────────────────────────────────────
    console.print("\n[bold]Results[/bold]")
    if ctx.transcript:
        console.print(f"  Segments : {len(ctx.transcript.segments)}")
        console.print(f"  Speakers : {ctx.transcript.speakers_detected}")
        console.print(f"  Language : {ctx.transcript.language}")

    transcript_path = output_dir / "transcript.md"
    if transcript_path.exists():
        console.print(f"\n  [green]Transcript[/green] -> {transcript_path}")

    for nf in sorted(f for f in output_dir.glob("*.md") if f.name != "transcript.md"):
        console.print(f"  [green]Notes     [/green] -> {nf}")

    if ctx.errors:
        console.print(f"\n[yellow]Warnings ({len(ctx.errors)}):[/yellow]")
        for err in ctx.errors:
            prefix = "[red]ERROR[/red]" if err.fatal else "[yellow]WARN[/yellow]"
            console.print(f"  {prefix} [{err.stage}] {err.message}")

    if any(e.fatal for e in ctx.errors):
        raise typer.Exit(1)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _delete_speaker(cfg, name: str) -> None:
    import sqlite3

    db_path = Path(cfg.storage.profiles_dir) / cfg.storage.db_filename
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT speaker_id FROM speakers WHERE display_name=?", (name,)
        ).fetchone()
        if row:
            sid = row[0]
            conn.execute("DELETE FROM speakers WHERE speaker_id=?", (sid,))
            conn.execute("DELETE FROM speaker_aliases WHERE speaker_id=?", (sid,))
            conn.execute("DELETE FROM voice_embeddings WHERE speaker_id=?", (sid,))
            conn.execute("DELETE FROM face_embeddings WHERE speaker_id=?", (sid,))
            conn.commit()

            # Also remove from ChromaDB
            try:
                import chromadb
                chroma_path = Path(cfg.storage.profiles_dir) / cfg.storage.chroma_dir
                client = chromadb.PersistentClient(path=str(chroma_path))
                for coll_name in ("voice_embeddings", "face_embeddings"):
                    try:
                        coll = client.get_collection(coll_name)
                        ids = coll.get(where={"speaker_id": {"$eq": sid}})["ids"]
                        if ids:
                            coll.delete(ids=ids)
                    except Exception:
                        pass
            except Exception:
                pass
    finally:
        conn.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()

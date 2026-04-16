# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""
Interactive wizard for identifying unknown speakers after a vocolith run.

For each unresolved Speaker_N label:
  - Shows 2-3 representative transcript snippets
  - Offers to play the audio clip (via ffplay)
  - Offers to show a video frame from that timestamp (via cv2 + xdg-open)
  - Prompts the user to name the speaker
  - Renames the profile in SQLite + ChromaDB
  - Updates transcript.md in the output directory

Usage:
  vocolith identify <output_dir>          # standalone after a run
  vocolith process meeting.mp4 --identify # auto-wizard after processing
"""
from __future__ import annotations
import json
import logging
import subprocess
import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

log = logging.getLogger(__name__)
console = Console()

# Number of representative samples to show per speaker
_SAMPLES_PER_SPEAKER = 3
# Minimum segment length (words) to be worth showing
_MIN_WORDS = 5
# Audio clip duration around each segment (seconds padding either side)
_AUDIO_PAD_S = 1.0


def run_confirmation_wizard(ctx: "PipelineContext", config=None) -> None:
    """
    Run the inline confirmation wizard after speaker_resolver has run.

    Called from pipeline.py when confirm_auto_identified is True.

    Shows each auto-identified speaker (strategies 2-5) and asks the user
    to confirm, correct, or skip.  Voice HIGH (strategy 1) speakers are
    shown as already confirmed — no interaction needed unless the user
    passes --confirm which shows them for review too.

    Modifies ctx.transcript.segments in-place with confirmed names.
    """
    from vocolith.stages.speaker_resolver import PendingSpeaker, _apply_names_to_transcript
    from vocolith.utils.progress import pause_progress

    pending: list[PendingSpeaker] = getattr(ctx, "pending_speakers", [])
    if not pending:
        return

    n_total   = len(pending)
    n_high    = sum(1 for p in pending if p.auto_confirmed)
    n_others  = n_total - n_high

    with pause_progress():
        _run_confirmation_wizard_inner(ctx, config, pending, n_total, n_high, n_others)


def _run_confirmation_wizard_inner(
    ctx: "PipelineContext",
    config,
    pending,
    n_total: int,
    n_high: int,
    n_others: int,
) -> None:
    """Inner body of run_confirmation_wizard, called with progress paused."""
    from vocolith.stages.speaker_resolver import _apply_names_to_transcript

    console.print(
        f"\n[bold yellow]Speaker Confirmation[/bold yellow] — "
        f"{n_total} speaker(s)  "
        f"[dim]({n_high} high-confidence, {n_others} need review)[/dim]\n"
    )

    # Build segment lookup
    segments_by_label: dict[str, list[dict]] = {}
    if ctx.transcript:
        for seg in ctx.transcript.segments:
            lbl = seg.speaker_label or ""
            if lbl:
                segments_by_label.setdefault(lbl, []).append(seg.model_dump())

    # Override map — starts as identity, updated per confirmation
    label_to_name: dict[str, str]   = {p.label: p.display_name for p in pending}
    label_to_method: dict[str, str] = {p.label: p.method       for p in pending}

    # ── Voice HIGH: show as a summary table, accept all at once ─────────────
    # Rather than pressing Enter N times, show all high-confidence matches in
    # one table. User can type a label to correct one, or Enter to accept all.
    high_conf = [p for p in pending if p.auto_confirmed]
    if high_conf:
        from rich.table import Table
        import re as _re
        tbl = Table("Label", "Name", "Confidence", box=None, show_header=True,
                    header_style="bold green")
        for p in high_conf:
            m = _re.search(r'[\d.]+\)', p.method)
            sim = m.group(0).rstrip(')') if m else "high"
            tbl.add_row(p.label, p.display_name, sim)
        console.print(Panel(tbl, title="[bold green]✓ High-Confidence Voice Matches[/bold green]",
                             border_style="green", padding=(0, 1)))
        console.print(
            "[dim]"
            "[green]Enter[/green]=Confirm all?  "
            "[yellow]a[/yellow]=Review each one  "
            "or type a label (e.g. SPEAKER_03) to correct:[/dim]"
        )
        correction_input = Prompt.ask("").strip()
        if correction_input.lower() == "a":
            # Walk through every high-confidence speaker one-by-one
            for p in high_conf:
                label_segs = segments_by_label.get(p.label, [])
                corrected = _confirm_one(p, label_segs, ctx.effective_audio, ctx.video_path)
                if isinstance(corrected, dict):
                    # Split — mixed voices, do NOT mark as confirmed single person
                    _apply_segment_split(ctx, p, corrected,
                                         label_to_name, label_to_method, config)
                elif isinstance(corrected, str) and corrected != p.display_name:
                    label_to_name[p.label] = corrected
                    label_to_method[p.label] = f"user_corrected({p.method})"
                    _correct_speaker(p.speaker_id, corrected, ctx.session_id, config,
                                     full_merge=True)
                    ctx.confirmed_single_labels.add(p.label)
                elif corrected is not None:
                    label_to_method[p.label] = f"user_confirmed({p.method})"
                    ctx.confirmed_single_labels.add(p.label)
                # skipped (corrected is None) → not added
        elif correction_input:
            match = next((p for p in high_conf
                          if p.label.lower() == correction_input.lower()), None)
            if match:
                label_segs = segments_by_label.get(match.label, [])
                corrected = _confirm_one(match, label_segs,
                                         ctx.effective_audio, ctx.video_path)
                if isinstance(corrected, dict):
                    _apply_segment_split(ctx, match, corrected,
                                         label_to_name, label_to_method, config)
                elif isinstance(corrected, str):
                    label_to_name[match.label] = corrected
                    if corrected.lower() == match.display_name.lower():
                        # User confirmed as-is (typed back the same name)
                        label_to_method[match.label] = f"user_confirmed({match.method})"
                    else:
                        label_to_method[match.label] = f"user_corrected({match.method})"
                        _correct_speaker(match.speaker_id, corrected, ctx.session_id, config,
                                         full_merge=True)
                    ctx.confirmed_single_labels.add(match.label)
            else:
                console.print(f"[yellow]Label '{correction_input}' not found in high-confidence list.[/yellow]")
        else:
            # Bulk Enter — ask for explicit confirmation before committing
            n_hc = len(high_conf)
            confirm_bulk = Prompt.ask(
                f"  Accept all [bold]{n_hc}[/bold] high-confidence match(es)?"
                f" ([green]y[/green]/[yellow]n[/yellow])"
            ).strip().lower()
            if confirm_bulk == "y":
                for p in high_conf:
                    ctx.confirmed_single_labels.add(p.label)
                    label_to_method[p.label] = f"user_confirmed({p.method})"
            else:
                console.print("[dim]Bulk accept cancelled — use [yellow]a[/yellow] to review individually.[/dim]")
        console.print()

    # ── Other strategies: individual interactive panels ───────────────────────
    for p in [p for p in pending if not p.auto_confirmed]:
        label_segs = segments_by_label.get(p.label, [])
        confirmed_name = _confirm_one(
            pending=p,
            segments=label_segs,
            audio_path=ctx.effective_audio,
            video_path=ctx.video_path,
        )

        if confirmed_name is None:
            # User skipped — do NOT store embedding (identity unverified)
            console.print(f"  [dim]Kept: {p.label} → {p.display_name}[/dim]")

        elif isinstance(confirmed_name, dict):
            # Segment-by-segment split — embedding is a mixture, do NOT store
            _apply_segment_split(ctx, p, confirmed_name, label_to_name,
                                  label_to_method, config)

        elif confirmed_name != p.display_name:
            # Single correction — rename the whole label, mark confirmed
            label_to_name[p.label]   = confirmed_name
            label_to_method[p.label] = f"user_corrected({p.method})"
            _correct_speaker(p.speaker_id, confirmed_name, ctx.session_id, config,
                             full_merge=True)
            ctx.confirmed_single_labels.add(p.label)
            console.print(
                f"  [green]✓[/green] {p.label} → [bold]{confirmed_name}[/bold] "
                f"[dim](corrected from '{p.display_name}')[/dim]"
            )
        else:
            # Confirmed as-is — mark confirmed
            label_to_method[p.label] = f"user_confirmed({p.method})"
            ctx.confirmed_single_labels.add(p.label)
            console.print(
                f"  [green]✓[/green] {p.label} → [bold]{p.display_name}[/bold] "
                f"[dim](confirmed)[/dim]"
            )

    # Re-apply names to transcript with any label-level corrections
    _apply_names_to_transcript(ctx, label_to_name, label_to_method)
    ctx.speaker_map = label_to_name

    # Store voice embeddings under the user-confirmed names.
    # pending_voice_embeddings was populated by resolve_speakers and deferred
    # to here so the embedding always reflects what the user verified.
    _store_confirmed_embeddings(ctx, label_to_name, config)


def _extract_audio_sample(ctx: "PipelineContext", label: str,
                           max_duration_s: float = 8.0) -> tuple[bytes, float] | None:
    """
    Extract a representative WAV audio sample for a speaker label.

    Picks the longest segment for the label (most likely to be clean speech),
    clamps it to max_duration_s, and returns (wav_bytes, actual_duration_s).
    Returns None if audio is unavailable or extraction fails.
    """
    if not ctx.effective_audio or not ctx.transcript:
        return None

    segs = [s for s in ctx.transcript.segments if s.speaker_label == label]
    if not segs:
        return None

    # Longest segment is most likely to give a clean, representative sample
    segs.sort(key=lambda s: s.end - s.start, reverse=True)
    seg = segs[0]
    start_s = seg.start
    end_s = min(seg.end, seg.start + max_duration_s)
    duration_s = end_s - start_s

    if duration_s < 0.5:
        return None

    try:
        import io
        import soundfile as _sf

        data, sr = _sf.read(str(ctx.effective_audio))
        start_i = int(start_s * sr)
        end_i = int(end_s * sr)
        clip = data[start_i:end_i]

        # Ensure mono
        if clip.ndim > 1:
            clip = clip.mean(axis=1)

        buf = io.BytesIO()
        # PCM_16 is the most universally compatible WAV subtype.
        # soundfile scales float→int16 automatically; no manual normalisation needed.
        _sf.write(buf, clip, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue(), duration_s

    except Exception as exc:
        log.debug("Audio sample extraction failed for %s: %s", label, exc)
        return None


def _store_confirmed_embeddings(ctx: "PipelineContext", label_to_name: dict,
                                config=None) -> None:
    """
    Persist voice embeddings AND audio samples under user-confirmed speaker names.

    Called at the end of the confirmation wizard after all speakers have been
    reviewed.  label_to_name maps diarization label → final confirmed name,
    so data is filed under what the user actually said, not the resolver's guess.

    Audio samples are stored as WAV BLOBs in SQLite so the profile database is
    fully self-contained — playback works regardless of whether debug files exist.

    Skips duplicate session entries (one embedding + sample per speaker per session).
    """
    pve = getattr(ctx, "pending_voice_embeddings", {})
    if not pve:
        return

    if config is None:
        from vocolith.config import load_config
        config = load_config()

    try:
        from vocolith.storage.db import init_db
        from vocolith.storage.speaker_store import SpeakerStore
        from vocolith.storage.vector_store import VectorStore
        from vocolith.models.speaker import SpeakerProfile
        from pathlib import Path as _Path
        import uuid

        db_path = _Path(config.storage.profiles_dir) / config.storage.db_filename
        conn = init_db(db_path)
        store = SpeakerStore(conn)
        vstore = VectorStore(_Path(config.storage.profiles_dir))

        # Build label → resolver-assigned speaker_id lookup so we can merge
        # the auto-created OCR/fallback profile into the confirmed name rather
        # than leaving a duplicate short-name profile behind.
        pending_sid: dict[str, str] = {
            p.label: p.speaker_id
            for p in getattr(ctx, "pending_speakers", [])
        }
        pending_name: dict[str, str] = {
            p.label: p.display_name
            for p in getattr(ctx, "pending_speakers", [])
        }

        # Only store embeddings for labels explicitly confirmed as a single
        # person by the user.  Splits (mixed voices) and skips (unverified)
        # are excluded — their embeddings would corrupt future voice matching.
        confirmed_set = getattr(ctx, "confirmed_single_labels", set())
        # Profiles we've already incremented meeting_count for in this wizard
        # run, so the same profile isn't touched twice if it appears under
        # multiple diarizer labels.
        wizard_touched: set[str] = set()

        for label, vec in pve.items():
            if label not in confirmed_set:
                log.debug("Skipping embedding for %s — not confirmed single-person", label)
                continue

            confirmed_name = label_to_name.get(label)
            if not confirmed_name:
                continue

            old_sid = pending_sid.get(label)
            old_name = pending_name.get(label, "")
            is_rename = bool(old_sid and old_name.lower() != confirmed_name.lower())

            if is_rename:
                # User renamed — merge old resolver profile into confirmed name.
                # full_merge=True only when the old profile is unconfirmed (new
                # this session, meeting_count==0): moves ALL embeddings so the
                # OCR stub "mallesh" doesn't persist after confirming "Mallesh".
                # full_merge=False for returning confirmed speakers (count>0):
                # a wrong voice-match at STD confidence should NOT destroy the
                # confirmed speaker's historical embeddings from prior meetings.
                old_profile_for_merge = store.get(old_sid)  # type: ignore[arg-type]
                full_merge_flag = (
                    old_profile_for_merge is None
                    or old_profile_for_merge.meeting_count == 0
                )
                sid = _correct_speaker(old_sid, confirmed_name, ctx.session_id, config,  # type: ignore[arg-type]
                                       full_merge=full_merge_flag)
                log.debug("Merged profile '%s' → '%s' for %s", old_name, confirmed_name, label)
                # reassign_all_voice (called inside _correct_speaker) may have moved
                # a same-session embedding from another label that shared old_sid
                # (e.g. two labels both OCR-matched to "Lou Gallo") into this profile.
                # That would cause voice_embedding_exists_for_session to return True
                # and skip storing the actual audio for this label.  Delete any such
                # wrongly-placed session embedding so we always store the real vec.
                vstore.delete_voice_for_session(sid, ctx.session_id)
            else:
                # User accepted resolver's suggestion — reuse existing profile.
                profile = store.find_by_name(confirmed_name) or store.find_by_alias(confirmed_name)
                if not profile:
                    profile = SpeakerProfile(display_name=confirmed_name)
                    store.save(profile)
                sid = profile.speaker_id
                # Accept path: skip if session already has an embedding (genuine
                # same-session duplicate, e.g. same speaker confirmed in two labels).
                if vstore.voice_embedding_exists_for_session(sid, ctx.session_id):
                    continue

            # Store voice embedding under the confirmed, de-duplicated profile
            emb_id = str(uuid.uuid4())
            vstore.add_voice(emb_id, sid, vec, ctx.session_id)
            store.record_embedding("voice_embeddings", emb_id, sid, ctx.session_id)
            # Increment meeting_count exactly once per confirmed profile per run.
            # Use meeting_count == 0 as the signal: it means the resolver deferred
            # the touch (new profile created by OCR/fallback) or the profile was
            # just created as the renamed target.  Returning speakers have
            # meeting_count > 0 because the resolver already touched them.
            if sid not in wizard_touched:
                current = store.get(sid)
                if current is not None and current.meeting_count == 0:
                    store.touch(sid)
                wizard_touched.add(sid)
            log.debug("Stored verified embedding: %s → '%s'", label, confirmed_name)

            # Store audio sample — makes profile self-contained for future playback
            sample = _extract_audio_sample(ctx, label)
            if sample:
                wav_bytes, dur = sample
                store.save_sample(sid, ctx.session_id, wav_bytes, dur)
                log.debug("Stored audio sample (%.1fs) for '%s'", dur, confirmed_name)
            else:
                log.debug("No audio sample available for '%s'", confirmed_name)

        # ── Touch confirmed profiles that have no voice embedding ──
        # Short speakers may have been confirmed but produced no usable audio,
        # so they were skipped by the embedding loop above.  Apply the same
        # meeting_count == 0 rule to give them their first count increment.
        for label in confirmed_set:
            if label in pve:
                continue  # already handled in the embedding loop
            confirmed_name = label_to_name.get(label)
            if not confirmed_name:
                continue
            profile = store.find_by_name(confirmed_name) or store.find_by_alias(confirmed_name)
            if profile and profile.speaker_id not in wizard_touched:
                current = store.get(profile.speaker_id)
                if current is not None and current.meeting_count == 0:
                    store.touch(profile.speaker_id)
                wizard_touched.add(profile.speaker_id)

        # ── Store split-derived embeddings (one per person from segment splits) ──
        # Reuse the same conn/store/vstore opened above — no need for a second
        # connection to the same SQLite file.
        split_embs = getattr(ctx, "confirmed_split_embeddings", [])
        for person_name, vec in split_embs:
            profile_sp = store.find_by_name(person_name) or store.find_by_alias(person_name)
            if not profile_sp:
                profile_sp = SpeakerProfile(display_name=person_name)
                store.save(profile_sp)

            sid_sp = profile_sp.speaker_id
            if vstore.voice_embedding_exists_for_session(sid_sp, ctx.session_id):
                continue

            emb_id_sp = str(uuid.uuid4())
            vstore.add_voice(emb_id_sp, sid_sp, vec, ctx.session_id)
            store.record_embedding("voice_embeddings", emb_id_sp, sid_sp, ctx.session_id)
            if sid_sp not in wizard_touched:
                current_sp = store.get(sid_sp)
                if current_sp is not None and current_sp.meeting_count == 0:
                    store.touch(sid_sp)
                wizard_touched.add(sid_sp)
            log.debug("Stored split embedding for '%s'", person_name)

    except Exception as exc:
        log.warning("Could not store split voice embeddings: %s", exc)


def _confirm_one(
    pending: "PendingSpeaker",
    segments: list[dict],
    audio_path,
    video_path,
) -> "str | dict[int, str] | None":
    """
    Show a single speaker identification and ask the user to confirm/correct.

    Voice HIGH (auto_confirmed=True) is shown with a green HIGH CONFIDENCE
    badge — Enter confirms instantly.  All other strategies show a yellow
    badge.  User can always correct or go segment-by-segment regardless.

    Returns:
        str              — single name for all segments (confirmed or corrected)
        dict[int, str]   — per-segment overrides when the label contains mixed speakers
                           keys are segment_id values from the transcript
        None             — user skipped (keep existing name unchanged)
    """
    samples = _pick_samples(segments)
    if not samples:
        return None  # No displayable samples — treat as skip, same as user pressing 's'
    sample_idx = 0
    method_label = pending.method.split("(")[0]

    # Visual style — color-coded by confidence level:
    #   voice HIGH (>=0.92): green  — reliable, auto-accept path
    #   voice STD  (>=0.85): yellow — needs confirmation, possibly wrong
    #   voice LOW  (<0.85):  red    — weak match, likely wrong
    #   OCR / addressee:     orange — text-based, unreliable for identity
    #   fallback:            red    — no signal, just a guess
    if pending.method.startswith("voice"):
        # Extract similarity score from method string, e.g. "voice(high,0.93)"
        import re as _re
        _m = _re.search(r"[\d.]+\)$", pending.method)
        _sim = float(_m.group().rstrip(")")) if _m else 0.0
        if _sim >= 0.92:
            border = "green"
            badge  = f"[bold green]voice pattern match[/bold green]  [dim]{pending.method}[/dim]"
        elif _sim >= 0.85:
            border = "yellow"
            badge  = f"[bold yellow]voice pattern match[/bold yellow]  [dim]{pending.method}[/dim]"
        else:
            border = "red"
            badge  = f"[bold red]voice pattern match[/bold red]  [dim]{pending.method}[/dim]"
        subtitle = "[dim][Enter] Accept · type to correct ·  segment-by-segment[/dim]"
    elif method_label in ("ocr", "addressee"):
        border   = "dark_orange"
        badge    = f"[bold dark_orange]?[/bold dark_orange]  [dim]via {method_label}[/dim]"
        subtitle = "[dim]Confirm, correct, or split if samples show different people[/dim]"
    else:
        border   = "red"
        badge    = f"[bold red]?[/bold red]  [dim]via {method_label}[/dim]"
        subtitle = "[dim]Confirm, correct, or split if samples show different people[/dim]"

    while True:
        lines = []
        for seg in samples[sample_idx: sample_idx + _SAMPLES_PER_SPEAKER]:
            ts   = _fmt_ts(seg.get("start", 0))
            text = (seg.get("text") or "").strip()
            lines.append(f"[dim][{ts}][/dim]  {text}")

        console.print(Panel(
            "\n\n".join(lines),
            title=(
                f"{pending.label} → [bold]{pending.display_name}[/bold]  {badge}"
            ),
            subtitle=subtitle,
            border_style=border,
            padding=(0, 1),
        ))

        legend = (
            "[bold]Keys:[/bold]  "
            "[green]Enter[/green]=Confirm?  "
            "[cyan]name[/cyan]=Rename all  "
        )
        if audio_path:
            legend += "[yellow]p[/yellow]=Play all clips  "
        if video_path:
            legend += "[yellow]f[/yellow]=Show frame  "
        if len(samples) > _SAMPLES_PER_SPEAKER:
            legend += "[yellow]n[/yellow]=More samples  "
        legend += "[yellow]g[/yellow]=Segment-by-segment  [yellow]s[/yellow]=Skip"
        console.print(legend)

        answer = Prompt.ask("").strip()

        if answer == "":
            # Confirm prompt — protects against Enter-key bounce
            confirmed = Prompt.ask(
                f"  Accept [bold]{pending.display_name!r}[/bold]?"
                f" ([green]y[/green]/[yellow]n[/yellow])"
            ).strip().lower()
            if confirmed == "y":
                return pending.display_name
            # n / empty / anything else → back to panel
        elif answer.lower() == "s":
            return None
        elif answer.lower() == "g":
            result = _confirm_segment_by_segment(
                pending, segments, audio_path, video_path
            )
            if result is None:
                # User pressed q — re-show the speaker panel
                continue
            return result
        elif answer.lower() == "p" and audio_path:
            # Play all currently shown clips in sequence
            visible = samples[sample_idx: sample_idx + _SAMPLES_PER_SPEAKER]
            for i, seg in enumerate(visible):
                console.print(f"[dim]Clip {i+1}/{len(visible)}: [{_fmt_ts(seg.get('start',0))}] {seg.get('text','')[:60]}...[/dim]")
                _play_audio(audio_path, seg.get("start", 0), seg.get("end", 0))
        elif answer.lower() == "f" and video_path:
            seg = samples[sample_idx % len(samples)]
            _show_frame(video_path, (seg.get("start", 0) + seg.get("end", 0)) / 2)
        elif answer.lower() == "n":
            sample_idx = (sample_idx + _SAMPLES_PER_SPEAKER) % max(1, len(samples))
        else:
            return answer


def _confirm_segment_by_segment(
    pending: "PendingSpeaker",
    segments: list[dict],
    audio_path,
    video_path,
) -> "dict[int, str] | None":
    """
    Walk through every segment for this speaker label one at a time.

    Returns a dict of {segment_id: speaker_name}, or None if the user
    pressed [q] to abandon and go back to the speaker panel.
    """
    good_segments = [s for s in segments
                     if len((s.get("text") or "").split()) >= _MIN_WORDS]
    if not good_segments:
        good_segments = segments

    n_segs = len(good_segments)
    console.print(
        f"\n[bold]Segment-by-segment mode[/bold] — "
        f"{n_segs} segment(s) for [yellow]{pending.label}[/yellow]\n"
        f"[dim]Press Enter to keep '[bold]{pending.display_name}[/bold]' for each segment. "
        f"Type a name to reassign.  During the loop, type [yellow]r NAME[/yellow] to "
        f"assign the current and all remaining segments in one go.[/dim]\n"
    )

    # ── Fast-path: bulk assign all segments before entering the loop ─────────
    fast = Prompt.ask(
        f"  [dim]Assign all {n_segs} segment(s) to "
        f"'[bold]{pending.display_name}[/bold]'? "
        f"([green]Enter[/green]=yes  [cyan]name[/cyan]=assign all to that name  "
        f"[yellow]n[/yellow]=review one-by-one)[/dim]",
        default="",
    ).strip()

    if fast.lower() != "n":
        bulk_name = pending.display_name if fast == "" else fast.strip()
        if not bulk_name:
            bulk_name = pending.display_name
        assignment = {
            seg.get("segment_id", i): bulk_name
            for i, seg in enumerate(good_segments)
        }
        console.print(
            f"  [green]✓[/green] All {n_segs} segment(s) assigned to "
            f"'[bold]{bulk_name}[/bold]'."
        )
        # Jump straight to the apply confirmation
        if Prompt.ask("\n[dim]Apply this split? [y/N][/dim]", default="N").strip().lower() == "y":
            return assignment
        # Cancelled — return None so caller re-shows the speaker panel
        console.print("[dim]Split cancelled.[/dim]")
        return None

    assignment: dict[int, str] = {}   # segment_id → name
    current_default = pending.display_name
    bulk_remaining = False   # True = auto-fill all remaining with current_default

    for i, seg in enumerate(good_segments):
        seg_id = seg.get("segment_id", i)

        if bulk_remaining:
            assignment[seg_id] = current_default
            continue

        ts     = _fmt_ts(seg.get("start", 0))
        ts_end = _fmt_ts(seg.get("end", 0))
        text   = (seg.get("text") or "").strip()

        body_lines = [f"[dim][{ts} → {ts_end}][/dim]  {text}", ""]
        legend = "[bold]Keys:[/bold]  [green]Enter[/green]=Keep '{default}'  [cyan]name[/cyan]=Assign  [yellow]r NAME[/yellow]=Assign remaining".format(
            default=current_default)
        if audio_path:
            legend += "  [yellow]p[/yellow]=Play audio"
        if video_path:
            legend += "  [yellow]f[/yellow]=Show frame"
        legend += "  [yellow]s[/yellow]=Skip segment  [yellow]q[/yellow]=Back to speaker"
        body_lines.append(legend)

        console.print(
            Panel(
                "\n".join(body_lines),
                title=(
                    f"[bold cyan]Segment {i+1}/{len(good_segments)}[/bold cyan]"
                    f"  default: [bold]{current_default}[/bold]"
                ),
                border_style="cyan",
                padding=(0, 1),
            )
        )

        while True:
            ans = Prompt.ask("").strip()
            if ans.lower() == "q":
                console.print("[dim]Returning to speaker panel...[/dim]")
                return None
            elif ans.lower() == "p" and audio_path:
                console.print(f"[dim]▶ Playing [{ts} → {ts_end}]...[/dim]")
                _play_audio(audio_path, seg.get("start", 0), seg.get("end", 0))
                console.print(f"[dim]Finished. Press Enter to keep '{current_default}', or type a name:[/dim]")
            elif ans.lower() == "f" and video_path:
                mid = (seg.get("start", 0) + seg.get("end", 0)) / 2
                console.print(f"[dim]📷 Extracting frame at {_fmt_ts(mid)}...[/dim]")
                _show_frame(video_path, mid)
                console.print(f"[dim]Frame opened. Press Enter to keep '{current_default}', or type a name:[/dim]")
            elif ans.lower() == "s":
                assignment[seg_id] = current_default
                break
            elif ans == "":
                assignment[seg_id] = current_default
                break
            elif ans.lower().startswith("r ") and ans[2:].strip():
                # "r NAME" — assign this segment AND all remaining to NAME
                bulk_name = ans[2:].strip()
                assignment[seg_id] = bulk_name
                current_default = bulk_name
                remaining = len(good_segments) - i - 1
                if remaining > 0:
                    console.print(
                        f"  [dim]Auto-assigning {remaining} remaining "
                        f"segment(s) to '[bold]{bulk_name}[/bold]'[/dim]"
                    )
                    bulk_remaining = True
                break
            else:
                assignment[seg_id] = ans
                if ans != current_default:
                    from rich.prompt import Confirm
                    if Confirm.ask(
                        f"  Make [cyan]'{ans}'[/cyan] the default for remaining segments?",
                        default=False,
                    ):
                        current_default = ans
                        remaining = len(good_segments) - i - 1
                        if remaining > 0:
                            console.print(
                                f"  [dim]Auto-assigning {remaining} remaining "
                                f"segment(s) to '{current_default}'[/dim]"
                            )
                            bulk_remaining = True
                break

    # Show summary
    names_used: dict[str, list[str]] = {}
    for sid, name in assignment.items():
        names_used.setdefault(name, []).append(str(sid))

    console.print("\n[bold]Split summary:[/bold]")
    for name, sids in sorted(names_used.items()):
        console.print(f"  [green]{name}[/green]: {len(sids)} segment(s)")

    if Prompt.ask("\n[dim]Apply this split? [y/N][/dim]", default="N").strip().lower() == "y":
        return assignment
    # Cancelled — return None so caller re-shows the speaker panel
    console.print("[dim]Split cancelled.[/dim]")
    return None


def _apply_segment_split(
    ctx: "PipelineContext",
    pending: "PendingSpeaker",
    assignment: dict[int, str],
    label_to_name: dict[str, str],
    label_to_method: dict[str, str],
    config=None,
) -> None:
    """
    Apply a segment-by-segment split to the transcript.

    The original speaker label (e.g. SPEAKER_01) may have been incorrectly
    grouping two different people.  For each unique name in `assignment`,
    find or create a speaker profile and set speaker_name + resolution_method
    directly on each affected TranscriptSegment.

    Segments not in `assignment` keep their existing name.
    """
    from vocolith.storage.db import init_db
    from vocolith.storage.speaker_store import SpeakerStore
    from pathlib import Path as _Path

    if config is None:
        from vocolith.config import load_config
        config = load_config()

    db_path = _Path(config.storage.profiles_dir) / config.storage.db_filename
    conn = init_db(db_path)
    store = SpeakerStore(conn)

    # Build name → speaker_id mapping, creating profiles for new names
    name_to_id: dict[str, str] = {}
    for name in set(assignment.values()):
        existing = store.find_by_name(name) or store.find_by_alias(name)
        if existing:
            name_to_id[name] = existing.speaker_id
        else:
            from vocolith.models.speaker import SpeakerProfile
            profile = SpeakerProfile(display_name=name)
            store.save(profile)
            name_to_id[name] = profile.speaker_id
            console.print(
                f"  [dim]Created new profile for '{name}' "
                f"(voice will enroll on next vocolith process run)[/dim]"
            )

    # Correct the original label's profile to the dominant name
    default_name = pending.display_name
    if not assignment:
        return
    dominant = max(set(assignment.values()), key=list(assignment.values()).count)
    if dominant != default_name:
        session_id = getattr(ctx, "session_id", None)
        _correct_speaker(pending.speaker_id, dominant, session_id, config)
        label_to_name[pending.label] = dominant
        label_to_method[pending.label] = f"user_split({pending.method})"

    # Apply per-segment overrides directly on the transcript
    if ctx.transcript:
        for seg in ctx.transcript.segments:
            seg_id = seg.segment_id
            if seg_id in assignment:
                new_name = assignment[seg_id]
                seg.speaker_name = new_name
                seg.resolution_method = "user_split"

    # ── Compute per-person embeddings from their audio slices ─────────────────
    # Each person identified in this split gets their OWN d-vector computed
    # from ONLY their audio segments — not the mixed full-label audio.
    # Results are queued in ctx.confirmed_split_embeddings for storage after
    # all confirmations complete.
    audio_path = getattr(ctx, "effective_audio", None)
    transcript = getattr(ctx, "transcript", None)
    if audio_path and transcript:
        # Build per-person segment time lists from transcript segments
        person_times: dict[str, list[tuple[float, float]]] = {}
        for seg in transcript.segments:
            if seg.segment_id in assignment:
                name = assignment[seg.segment_id]
                person_times.setdefault(name, []).append((seg.start, seg.end))

        try:
            from vocolith.stages.voice_embedder import embed_segments
            for name, times in person_times.items():
                # min_duration_s=0.0: store even a single short segment —
                # if this is all the audio we have for this person from a
                # split, a noisy embedding is better than no embedding.
                vec = embed_segments(audio_path, times, min_duration_s=0.0)
                if vec is not None:
                    ctx.confirmed_split_embeddings.append((name, vec))
                    log.debug(
                        "Split embedding computed for '%s' (%d segment(s) from %s)",
                        name, len(times), pending.label,
                    )
        except Exception as exc:
            log.warning("Could not compute split embeddings for %s: %s", pending.label, exc)

    # Summary
    names_used = {}
    for name in assignment.values():
        names_used[name] = names_used.get(name, 0) + 1
    for name, count in sorted(names_used.items()):
        marker = "[green]✓[/green]" if name == dominant else "[cyan]↪[/cyan]"
        console.print(f"  {marker} {pending.label} → [bold]{name}[/bold] "
                      f"({count} segment(s))")


def _correct_speaker(old_speaker_id: str, new_name: str,
                      session_id: str | None, config=None,
                      full_merge: bool = False) -> str:
    """
    Correct a speaker identification by finding or creating the right profile
    and reassigning the current session's voice embeddings to it.

    Unlike a simple rename this does NOT touch the old profile's display_name
    or its embeddings from other meetings.  That prevents cross-contamination
    where renaming "Alice Chen" to "Bob Martinez" would cause all of Alice's
    stored voice data to appear as Bob in future runs.

    Returns the new speaker_id.
    """
    if config is None:
        from vocolith.config import load_config
        config = load_config()
    from vocolith.storage.db import init_db
    from vocolith.storage.speaker_store import SpeakerStore
    from vocolith.storage.vector_store import VectorStore
    from vocolith.models.speaker import SpeakerProfile
    from pathlib import Path as _Path

    db_path = _Path(config.storage.profiles_dir) / config.storage.db_filename
    conn = init_db(db_path)
    store = SpeakerStore(conn)
    vector_store = VectorStore(_Path(config.storage.profiles_dir))

    # Find existing profile for the target name or create a fresh one
    target = store.find_by_name(new_name) or store.find_by_alias(new_name)
    if target:
        new_speaker_id = target.speaker_id
    else:
        profile = SpeakerProfile(display_name=new_name)
        store.save(profile)
        new_speaker_id = profile.speaker_id

    if new_speaker_id == old_speaker_id:
        # Nothing to move — just make sure the name is right
        conn.execute(
            "UPDATE speakers SET display_name=? WHERE speaker_id=?",
            (new_name, old_speaker_id),
        )
        conn.commit()
        return new_speaker_id

    # Move voice embeddings to the correct profile.
    #
    # For unconfirmed ("Speaker_N") profiles, move ALL embeddings across every
    # session: the old placeholder may have accumulated embeddings from prior
    # runs of the same audio, and leaving them in place would let future
    # find_voice() calls match back to the stale "Speaker_N" label instead of
    # the corrected name.
    #
    # full_merge=True is passed when the caller knows the old profile is a
    # resolver guess (OCR/addressee/fallback) even if it has a human-looking
    # name.  In that case we do the same full migration as for Speaker_N so
    # that embeddings accumulated under the OCR spelling ("mallesh") follow
    # the user-confirmed canonical name ("Mallesh Gowda") and no duplicate
    # profile is left behind.
    #
    # For all other confirmed profiles, only move the current session to avoid
    # cross-contaminating correctly-identified data from other meetings.
    old_profile = store.get(old_speaker_id)
    old_is_unconfirmed = (
        full_merge
        or old_profile is None
        or old_profile.display_name.startswith("Speaker_")
    )
    if old_is_unconfirmed:
        moved = vector_store.reassign_all_voice(old_speaker_id, new_speaker_id)
    elif session_id:
        moved = vector_store.reassign_voice_for_session(
            old_speaker_id, new_speaker_id, session_id
        )
    else:
        moved = 0
    log.debug("Moved %d embedding(s) from %s to %s",
              moved, old_speaker_id, new_speaker_id)

    # Clean up the old profile if it now has no voice embeddings at all
    # (i.e. it was only ever a placeholder for this mis-identification).
    # Must migrate session_speakers rows first: the table has a FK to speakers
    # without ON DELETE CASCADE, so the DELETE would fail otherwise.
    remaining = vector_store.voice_embedding_count(old_speaker_id)
    if remaining == 0:
        # Migrate any session history to the new profile before deleting old one
        conn.execute(
            """INSERT OR IGNORE INTO session_speakers
                   (session_id, speaker_id, label_used, resolution_method)
               SELECT session_id, ?, label_used, resolution_method
               FROM session_speakers WHERE speaker_id=?""",
            (new_speaker_id, old_speaker_id),
        )
        conn.execute("DELETE FROM session_speakers WHERE speaker_id=?", (old_speaker_id,))
        conn.execute("DELETE FROM speakers WHERE speaker_id=?", (old_speaker_id,))
        conn.commit()
        log.debug("Deleted orphan profile %s (no remaining embeddings)", old_speaker_id)

    return new_speaker_id


# Need TYPE_CHECKING import for PipelineContext type hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from vocolith.pipeline import PipelineContext
    from vocolith.stages.speaker_resolver import PendingSpeaker


def run_wizard(output_dir: Path, config=None) -> int:
    """
    Run the speaker identification wizard for an output directory.

    Uses the same _confirm_one TUI as the inline process wizard, giving
    identical shortcut keys (p=play, f=frame, n=next samples,
    g=segment-by-segment, s=skip, Enter=accept).

    Returns the number of speakers that were successfully identified.
    """
    from vocolith.config import load_config as _load_config
    if config is None:
        config = _load_config()

    output_dir = Path(output_dir)

    # Load the diarization manifest
    manifest_path = _find_manifest(output_dir)
    if not manifest_path:
        console.print(f"[red]No diarization.json found under {output_dir}[/red]")
        console.print("  Run `vocolith process <video>` first.")
        return 0

    manifest = json.loads(manifest_path.read_text())
    meta     = manifest.get("meta", {})
    segments = manifest.get("segments", [])
    session_id: str | None = meta.get("session_id")

    video_path = Path(meta["video_path"]) if meta.get("video_path") else None
    audio_path = Path(meta["audio_path"]) if meta.get("audio_path") else None

    if not segments:
        console.print("[dim]No segments in diarization data.[/dim]")
        return 0

    # Find unresolved Speaker_N labels
    unresolved: dict[str, list[dict]] = {}
    for seg in segments:
        label = seg.get("speaker_label") or ""
        name  = seg.get("speaker_name") or label
        if label and name.startswith("Speaker_"):
            unresolved.setdefault(label, []).append(seg)

    if not unresolved:
        console.print("[green]All speakers are already identified.[/green]")
        return 0

    console.print(
        f"\n[bold]Speaker Identification Wizard[/bold]\n"
        f"Found [yellow]{len(unresolved)}[/yellow] unresolved speaker(s) in: "
        f"[dim]{output_dir.name}[/dim]\n"
    )

    if not audio_path or not audio_path.exists():
        console.print("[yellow]Warning:[/yellow] Audio file not found — playback unavailable.")
        audio_path = None
    if not video_path or not video_path.exists():
        console.print("[yellow]Warning:[/yellow] Video file not found — frame preview unavailable.")
        video_path = None

    # Load DB for profile lookups / creation
    from vocolith.storage.db import init_db
    from vocolith.storage.speaker_store import SpeakerStore
    from vocolith.stages.speaker_resolver import PendingSpeaker
    from vocolith.models.speaker import SpeakerProfile

    db_path = Path(config.storage.profiles_dir) / config.storage.db_filename
    conn = init_db(db_path)
    store = SpeakerStore(conn)

    identified = 0

    for label, label_segs in sorted(unresolved.items()):
        # Current display name (may be "Speaker_N" or a partial name from a previous run)
        current_name = (label_segs[0].get("speaker_name") or label) if label_segs else label

        # Find or create a SQLite profile so _confirm_one / _correct_speaker can work
        profile = store.find_by_name(current_name) or store.find_by_alias(current_name)
        if not profile:
            profile = SpeakerProfile(display_name=current_name)
            store.save(profile)

        pending = PendingSpeaker(
            label=label,
            display_name=current_name,
            method="unresolved",
            speaker_id=profile.speaker_id,
            auto_confirmed=False,
        )

        result = _confirm_one(
            pending=pending,
            segments=label_segs,
            audio_path=audio_path,
            video_path=video_path,
        )

        if result is None:
            console.print(f"[dim]Skipped {label}[/dim]\n")

        elif isinstance(result, dict):
            # Segment-by-segment split — update transcript.md directly
            _apply_split_to_transcript_file(
                label, result, output_dir, config,
                current_display=current_name,
            )
            n_names = len(set(result.values()))
            identified += 1
            console.print(
                f"[green]✓[/green] {label} split into "
                f"[bold]{n_names} speaker(s)[/bold]\n"
            )

        else:
            if result != current_name:
                # Correct the profile: find/create target, reassign embeddings,
                # clean up orphan old profile, then update transcript.md.
                _correct_speaker(pending.speaker_id, result, session_id, config)
                _apply_rename(
                    label, result, output_dir, config,
                    speaker_id=None,          # DB already updated by _correct_speaker
                    current_display=current_name,
                )
                identified += 1
                console.print(f"[green]✓[/green] {label} → [bold]{result}[/bold]\n")
            else:
                console.print(f"  [dim]Kept: {label} → {current_name}[/dim]\n")

    if identified:
        console.print(
            f"[bold green]{identified} speaker(s) identified.[/bold green] "
            f"Run `vocolith process` again on the same video for richer notes, "
            f"or the names are now stored for future meetings."
        )
    return identified


# ── Sample selection ──────────────────────────────────────────────────────────

def _pick_samples(segments: list[dict], n: int = 6) -> list[dict]:
    """
    Pick up to n representative segments for this speaker, spread across
    their speaking time and long enough to be useful.
    """
    good = [
        s for s in segments
        if len((s.get("text") or "").split()) >= _MIN_WORDS
    ]
    if not good:
        good = segments  # fall back to whatever we have

    if len(good) <= n:
        return good

    # Spread evenly across the speaker's total speaking time
    step = len(good) / n
    return [good[int(i * step)] for i in range(n)]


# ── Audio playback ────────────────────────────────────────────────────────────

def _play_audio(audio_path: Path, start: float, end: float) -> None:
    """Play an audio clip using ffplay (non-blocking display, auto-exit)."""
    duration = max(0.5, (end - start)) + _AUDIO_PAD_S * 2
    t_start  = max(0.0, start - _AUDIO_PAD_S)

    console.print(f"[dim]Playing [{_fmt_ts(start)} → {_fmt_ts(end)}] ...[/dim]")
    try:
        subprocess.run(
            [
                "ffplay", "-nodisp", "-autoexit",
                "-ss", str(t_start),
                "-t",  str(duration),
                str(audio_path),
            ],
            check=False,
            capture_output=True,
        )
    except FileNotFoundError:
        console.print("[yellow]ffplay not found — install ffmpeg to enable audio playback.[/yellow]")


# ── Video frame extraction ────────────────────────────────────────────────────

def _show_frame(video_path: Path, timestamp_s: float) -> None:
    """
    Extract a single frame from the video at timestamp_s and open it
    with the system image viewer (xdg-open on Linux).
    """
    try:
        import cv2
    except ImportError:
        console.print("[yellow]opencv not available — cannot extract frame.[/yellow]")
        return

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_s * fps))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        console.print(f"[yellow]Could not extract frame at {_fmt_ts(timestamp_s)}[/yellow]")
        return

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, prefix="vocolith_frame_") as f:
        tmp = f.name

    cv2.imwrite(tmp, frame)
    console.print(f"[dim]Frame at {_fmt_ts(timestamp_s)} → {tmp}[/dim]")

    # Try to open with system viewer
    for viewer in ["xdg-open", "feh", "display", "eog", "gimp"]:
        try:
            subprocess.Popen([viewer, tmp])
            break
        except FileNotFoundError:
            continue
    else:
        console.print(f"[yellow]No image viewer found. Frame saved to: {tmp}[/yellow]")


# ── Apply rename ──────────────────────────────────────────────────────────────

def _apply_rename(label: str, new_name: str, output_dir: Path, config=None,
                  speaker_id: str | None = None,
                  current_display: str | None = None) -> None:
    """
    Rename the speaker in SQLite and transcript.md.

    speaker_id: when provided, update SQLite by speaker_id (reliable).
                Falls back to searching by label (old behaviour).
    current_display: when provided, use this for transcript.md text replacement
                     instead of label.  Required when label is a diarization ID
                     (e.g. "SPEAKER_01") but the transcript uses the display name
                     (e.g. "Speaker_3").
    """
    if config is None:
        from vocolith.config import load_config
        config = load_config()

    from vocolith.storage.db import init_db
    from vocolith.storage.speaker_store import SpeakerStore

    db_path = Path(config.storage.profiles_dir) / config.storage.db_filename
    if db_path.exists():
        conn = init_db(db_path)
        store = SpeakerStore(conn)
        if speaker_id:
            conn.execute(
                "UPDATE speakers SET display_name=? WHERE speaker_id=?",
                (new_name, speaker_id),
            )
            conn.commit()
        else:
            profile = store.find_by_name(label)
            if profile:
                conn.execute(
                    "UPDATE speakers SET display_name=? WHERE speaker_id=?",
                    (new_name, profile.speaker_id),
                )
                conn.commit()
        log.debug("Renamed %s -> %s in SQLite", label, new_name)

    # Update transcript.md — use current_display when the label is a raw
    # diarization ID that doesn't appear in the rendered transcript text.
    search_name = current_display or label
    transcript_path = output_dir / "transcript.md"
    if transcript_path.exists():
        text = transcript_path.read_text(encoding="utf-8")
        updated = text.replace(f"**{search_name}**:", f"**{new_name}**:")
        updated = updated.replace(f"**{search_name}** (", f"**{new_name}** (")
        if updated != text:
            transcript_path.write_text(updated, encoding="utf-8")
            log.debug("Updated transcript.md: %s -> %s", search_name, new_name)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _apply_split_to_transcript_file(
    label: str,
    assignment: dict[int, str],
    output_dir: Path,
    config=None,
    current_display: str | None = None,
) -> None:
    """
    Apply a segment-by-segment split to transcript.md on disk.

    current_display: when provided, use this for transcript text matching
                     instead of label.  Needed when label is a raw diarization
                     ID (e.g. "SPEAKER_01") but the transcript uses the display
                     name (e.g. "Speaker_3").
    """
    manifest_path = _find_manifest(output_dir)
    if not manifest_path:
        return
    manifest = json.loads(manifest_path.read_text())
    seg_map = {s["segment_id"]: s for s in manifest.get("segments", [])}

    if config is None:
        from vocolith.config import load_config
        config = load_config()
    from vocolith.storage.db import init_db
    from vocolith.storage.speaker_store import SpeakerStore
    from pathlib import Path as _Path
    db_path = _Path(config.storage.profiles_dir) / config.storage.db_filename
    conn = init_db(db_path)
    store = SpeakerStore(conn)
    for name in set(assignment.values()):
        if not (store.find_by_name(name) or store.find_by_alias(name)):
            from vocolith.models.speaker import SpeakerProfile
            store.save(SpeakerProfile(display_name=name))

    transcript_path = output_dir / "transcript.md"
    if not transcript_path.exists():
        return

    # The transcript uses display_name (e.g. "Speaker_3"), not the diarization
    # label (e.g. "SPEAKER_01"), so use current_display for text matching.
    search_name = current_display or label

    lines = transcript_path.read_text(encoding="utf-8").splitlines()
    updated_lines = []
    for line in lines:
        for seg_id, name in assignment.items():
            seg = seg_map.get(seg_id)
            if not seg:
                continue
            ts = _fmt_ts(seg.get("start", 0))
            if f"[{ts}]" in line and f"**{search_name}**" in line:
                line = line.replace(f"**{search_name}**", f"**{name}**")
                break
        updated_lines.append(line)

    transcript_path.write_text("\n".join(updated_lines), encoding="utf-8")


def _find_manifest(output_dir: Path) -> Path | None:
    """Search for diarization.json under output_dir or its debug/ subdir."""
    # Also search the configured debug_dir if it differs from output_dir/debug
    candidates = [
        output_dir / "diarization.json",
        output_dir / "debug" / "diarization.json",
        output_dir / ".." / "debug" / "diarization.json",
    ]
    # Try the global debug_dir from config (e.g. /tmp/vocolith/)
    try:
        from vocolith.config import load_config
        cfg = load_config()
        if cfg.pipeline.debug_dir:
            candidates.append(Path(cfg.pipeline.debug_dir) / "diarization.json")
    except Exception:
        pass

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.exists():
                log.debug("Found manifest: %s", resolved)
                return resolved
        except Exception:
            continue

    log.warning(
        "diarization.json not found. Searched:\n%s",
        "\n".join(f"  {c}" for c in candidates),
    )
    return None


def _fmt_ts(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

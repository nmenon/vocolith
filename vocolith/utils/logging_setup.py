# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Rich-based logging configuration."""
import logging
from rich.console import Console
from rich.logging import RichHandler

console = Console()

# Third-party loggers that are always noisy regardless of level
_ALWAYS_QUIET = (
    "httpx", "httpcore", "urllib3", "filelock", "PIL",
    "whisperx", "pyannote", "lightning", "torch",
)


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """
    Configure logging verbosity.

    Default (no flags): WARNING only — silences all INFO noise; the Rich
      progress bars and status() calls handle user-facing output.
    --verbose: INFO — useful details without internal stage chatter.
    --debug:   DEBUG — everything, for diagnosing failures.
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    # basicConfig() is a no-op when the root logger already has handlers
    # (e.g. set by an imported library).  Use force=True to override any
    # existing handler configuration and guarantee our RichHandler is active.
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                rich_tracebacks=True,
                show_path=debug,        # show file:line only in debug mode
                show_time=verbose or debug,
            )
        ],
        force=True,
    )

    # Third-party loggers always at WARNING regardless of user level
    for noisy in _ALWAYS_QUIET:
        logging.getLogger(noisy).setLevel(logging.WARNING)

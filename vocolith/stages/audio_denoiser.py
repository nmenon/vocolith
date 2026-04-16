# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Stage 2: Spectral noise reduction before transcription."""
from __future__ import annotations
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def denoise_audio(input_wav: Path, output_dir: Path,
                   stationary: bool = False) -> Path:
    """
    Apply spectral gating noise reduction to a WAV file.

    Args:
        input_wav:  Path to the raw WAV file.
        output_dir: Directory for the cleaned output file.
        stationary: If False (recommended), uses non-stationary noise
                    estimation (better for meetings with variable background).

    Returns:
        Path to the denoised WAV file.
    """
    try:
        import noisereduce as nr
        import soundfile as sf
        import numpy as np
    except ImportError as exc:
        log.warning("noisereduce not installed (%s). Skipping denoising.", exc)
        return input_wav

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wav = output_dir / "audio_clean.wav"

    log.debug("Denoising audio (stationary=%s)...", stationary)
    try:
        data, rate = sf.read(str(input_wav))

        # noisereduce expects float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        reduced = nr.reduce_noise(y=data, sr=rate, stationary=stationary,
                                   prop_decrease=0.75)

        sf.write(str(output_wav), reduced, rate)
        log.debug("Denoised audio -> %s", output_wav.name)
        return output_wav
    except Exception as exc:
        log.warning("Denoising failed (%s) — using raw audio unchanged.", exc)
        return input_wav

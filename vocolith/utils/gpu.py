# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""GPU detection and model configuration helpers."""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

_VALID_COMPUTE_TYPES = {"float16", "float32", "int8", "int8_float16"}
_VALID_MODEL_SIZES = {"tiny", "base", "small", "medium", "large-v2", "large-v3"}


def _get_cuda_info() -> tuple[bool, float]:
    """Return (cuda_available, vram_gb) in a single torch query."""
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return True, vram
    except Exception:
        pass
    return False, 0.0


def get_device() -> str:
    """Return 'cuda' if CUDA is available, else 'cpu'."""
    cuda_ok, _ = _get_cuda_info()
    return "cuda" if cuda_ok else "cpu"


def get_vram_gb() -> float:
    """Return available GPU VRAM in GB, or 0.0 on CPU-only systems."""
    _, vram = _get_cuda_info()
    return vram


def get_whisper_config(
    model_size_override: str | None = None,
    compute_type_override: str | None = None,
    batch_size_override: int | None = None,
) -> dict:
    """
    Auto-select WhisperX model size, compute type, and batch size based on
    available hardware. Explicit overrides take precedence over auto-selection.

    Returns a dict with keys: device, model_size, compute_type, batch_size, vram_gb.
    """
    cuda_ok, vram_gb = _get_cuda_info()
    device = "cuda" if cuda_ok else "cpu"

    # ── Model size ────────────────────────────────────────────────────────────
    if model_size_override and model_size_override != "auto":
        if model_size_override not in _VALID_MODEL_SIZES:
            log.warning(
                "Unknown model size '%s'; valid options: %s. Falling back to auto.",
                model_size_override, sorted(_VALID_MODEL_SIZES),
            )
            model_size_override = None

    if model_size_override and model_size_override != "auto":
        model_size = model_size_override
    elif vram_gb >= 6.0:
        # float16 large-v2 (~3.1GB) + inference buffers fit comfortably
        model_size = "large-v2"
    elif vram_gb >= 4.0:
        # int8 large-v2 (~1.6GB) + inference fits on most 4-6GB cards
        # Falls back automatically if OOM occurs (see transcriber.py ladder)
        model_size = "large-v2"
    elif vram_gb >= 2.0:
        model_size = "medium"
    else:
        model_size = "small"

    # ── Compute type ──────────────────────────────────────────────────────────
    if compute_type_override and compute_type_override != "auto":
        if compute_type_override not in _VALID_COMPUTE_TYPES:
            log.warning(
                "Unknown compute_type '%s'; valid options: %s. Falling back to auto.",
                compute_type_override, sorted(_VALID_COMPUTE_TYPES),
            )
            compute_type_override = None

    if compute_type_override and compute_type_override != "auto":
        compute_type = compute_type_override
    else:
        if device != "cuda":
            compute_type = "int8"
        elif vram_gb >= 5.0:
            # Plenty of headroom — full float16 precision
            compute_type = "float16"
        else:
            # <5 GB VRAM: large-v2 float16 = ~3.1GB weights alone, leaves no headroom.
            # int8 = ~1.6GB weights, nearly identical quality for speech recognition.
            compute_type = "int8"

    # ── Batch size ────────────────────────────────────────────────────────────
    # large-v2 weights ~3GB; leave headroom for inference activations:
    #   <4.5 GB  → batch 8  (T400 4GB: model + batch fits)
    #   ≥4.5 GB  → batch 16
    if batch_size_override is not None:
        batch_size = max(1, int(batch_size_override))
    elif vram_gb >= 4.5:
        batch_size = 16
    elif vram_gb >= 3.5:
        # T400/similar ~4 GB cards: batch=8 OOMs in practice due to mel-spectrogram
        # activation buffers for long audio (1h+). Use batch=4 as a safe default.
        # The degradation ladder in transcriber.py handles further fallback if needed.
        batch_size = 4
    elif vram_gb >= 2.0:
        batch_size = 4
    else:
        batch_size = 2

    cfg = {
        "device": device,
        "model_size": model_size,
        "compute_type": compute_type,
        "batch_size": batch_size,
        "vram_gb": round(vram_gb, 1),
    }

    # Always-visible hardware summary — user-facing eye candy
    try:
        from vocolith.utils.progress import status as _status
        if device == "cuda":
            try:
                import torch
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "GPU"
            _status(
                f"[bold cyan]{gpu_name}[/bold cyan] {vram_gb:.1f} GB VRAM  "
                f"[green]{model_size}[/green] {compute_type} batch={batch_size}"
            )
        else:
            _status(f"[yellow]CPU[/yellow] mode — [green]{model_size}[/green] {compute_type} batch={batch_size}")
    except Exception:
        pass

    log.debug(
        "Hardware: %s (%.1f GB VRAM) -> model=%s  compute=%s  batch=%d",
        device.upper(), vram_gb, model_size, compute_type, batch_size,
    )
    return cfg

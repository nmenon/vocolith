# SPDX-License-Identifier: GPL-2.0-only
# Copyright (C) 2026 Texas Instruments Incorporated - https://www.ti.com/
"""Configuration loader using pydantic-settings + YAML."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings  # noqa: F401


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    format: str = "wav"


class DenoiserConfig(BaseModel):
    enabled: bool = True
    stationary: bool = False


class TranscriptionConfig(BaseModel):
    model_size: str = "auto"
    language: str | None = None
    batch_size: Any = "auto"
    compute_type: str = "auto"
    huggingface_token: str | None = None


class DiarizationConfig(BaseModel):
    enabled: bool = True
    min_speakers: int = 1
    max_speakers: int = 10
    # Suppress noisy pyannote internal warnings that don't affect output:
    #   - "std(): degrees of freedom <= 0" (short speech segments)
    #   - TF32 ReproducibilityWarning (pyannote disables TF32 for determinism)
    suppress_warnings: bool = True


class FrameSamplingConfig(BaseModel):
    interval_seconds: float = 5.0
    change_threshold: float = 0.002  # screen shares have tiny frame diffs
    min_frames: int = 6               # always keep at least this many frames
    top_strip_pct: float = 0.25
    bottom_strip_pct: float = 0.15


class OcrConfig(BaseModel):
    languages: list[str] = Field(default_factory=lambda: ["en"])
    confidence_threshold: float = 0.5
    min_name_freq: int = 2
    extract_terminology: bool = True
    # Number of parallel OCR worker processes (CPU-only).
    # 0 = auto: half of CPU cores, capped at 6.
    # 1 = sequential fallback.
    ocr_workers: int = 0
    # When True and CUDA is available, bypass the worker pool and use a single
    # GPU-backed reader instead (sequential but ~7x faster per frame than CPU).
    ocr_gpu_worker: bool = True
    # Names seen in more than this fraction of total frames are static UI noise
    # (browser tabs, window titles, always-on overlays) and are discarded.
    # 0.8 = appears in >80% of frames → noise.  Set 1.0 to disable.
    ocr_noise_freq_threshold: float = 0.8
    # GPU/CPU frame split when both are active.
    # 0.0 = auto: balance GPU(~1 s/frame) vs CPU(~7 s/N_workers) wall time.
    # 0.0–1.0: explicit fraction of frames assigned to GPU (e.g. 0.6 = 60% GPU).
    ocr_gpu_frame_ratio: float = 0.0
    ui_blacklist: list[str] = Field(default_factory=lambda: [
        "Mute", "Unmute", "Video", "Participants", "Chat", "Share",
        "Record", "Leave", "End", "Reactions", "More", "Settings",
        "Raise Hand", "View", "Pin",
    ])


class FaceRecognitionConfig(BaseModel):
    tolerance: float = 0.6
    model: str = "hog"
    min_face_height_px: int = 30


class SpeakerResolutionConfig(BaseModel):
    # Voice d-vector thresholds
    voice_similarity_threshold_high: float = 0.92  # trust immediately, skip all other signals
    voice_similarity_threshold: float = 0.89        # accept if no better signal found
    # OCR / addressee
    ocr_match_confidence: float = 0.70
    face_match_threshold: float = 0.60
    addressee_min_votes: int = 2
    # Reference transcript cross-referencing (e.g. Teams .vtt export).
    # Single threshold, used both ways: coverage >= this = confident clean
    # match (name directly); coverage < this with --room-attendees set and
    # no explicit --room-label = presume shared room mic (route to split).
    reference_match_threshold: float = 0.6
    # Require user confirmation before committing speaker names.
    # ALL strategies shown — voice HIGH with a confidence badge, others without.
    # Disable with --no-confirm for fully automated runs (e.g. batch processing).
    confirm_auto_identified: bool = True


class LlmConfig(BaseModel):
    model: str = "gpt-4o-mini"
    use_local: bool = False
    local_base_url: str = "http://localhost:11434/v1"
    local_model: str = "mistral:7b"
    max_tokens: int = 32000
    temperature: float = 0.1          # low = more deterministic, fewer hallucinations
    max_transcript_chars: int = 100000
    chunk_overlap_chars: int = 500
    verify_notes: bool = True          # run a second LLM pass to cross-check claims
    # SSL verify: True for public endpoints; False for self-signed/corporate proxies
    # Can also override via OPENAI_API_KEY + OPENAI_API_MODEL + OPENAI_API_PROVIDER env vars
    ssl_verify: bool = False


class StorageConfig(BaseModel):
    profiles_dir: str = "~/.cache/vocolith"
    db_filename: str = "speakers.db"
    chroma_dir: str = "chroma"


class PipelineConfig(BaseModel):
    enable_face_recognition: bool = False
    enable_ocr: bool = True
    parallel_video_audio: bool = False  # sequential by default; avoids GPU VRAM contention
    output_dir: str | None = None    # None = create folder next to input video
    debug_dir: str | None = None     # None = <output_dir>/debug/; set e.g. /tmp/vocolith


class TemplatesConfig(BaseModel):
    default: str = "standard"
    # Templates generated on every run (in addition to any --template flag).
    # Set to an empty list to disable multi-output and only use --template.
    run: list[str] = Field(default_factory=lambda: [
        "executive_summary",
        "email_notes",
        "detailed_technical_discussion_notes",
    ])
    user_templates_dir: str = "~/.config/vocolith/templates"


class MeetingTypeConfig(BaseModel):
    """A named meeting type that maps to a specific set of templates."""
    templates: list[str]
    description: str = ""


class AppConfig(BaseModel):
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    denoiser: DenoiserConfig = Field(default_factory=DenoiserConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    frame_sampling: FrameSamplingConfig = Field(default_factory=FrameSamplingConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    face_recognition: FaceRecognitionConfig = Field(default_factory=FaceRecognitionConfig)
    speaker_resolution: SpeakerResolutionConfig = Field(default_factory=SpeakerResolutionConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    templates: TemplatesConfig = Field(default_factory=TemplatesConfig)
    # Named meeting types — each maps an alias to a list of templates.
    # Use with: vocolith process meeting.mp4 --meeting-type <alias>
    meeting_types: dict[str, MeetingTypeConfig] = Field(default_factory=lambda: {
        "standup": MeetingTypeConfig(
            templates=["executive_summary", "email_notes"],
            description="Daily standup — short summary + email recap",
        ),
        "design_review": MeetingTypeConfig(
            templates=["executive_summary", "detailed_technical_discussion_notes", "design_review"],
            description="Technical design review — exec summary + full technical notes",
        ),
        "customer_call": MeetingTypeConfig(
            templates=["executive_summary", "email_notes"],
            description="External customer or partner call",
        ),
        "team_sync": MeetingTypeConfig(
            templates=["executive_summary", "email_notes", "detailed_technical_discussion_notes"],
            description="Team sync — all three standard outputs",
        ),
    })


def load_config(config_path: Path | None = None) -> AppConfig:
    """
    Load config from YAML file, then apply environment variable overrides
    (prefix MD_, e.g. MD_LLM__MODEL overrides llm.model).

    Search order:
      1. --config <path>  (explicit CLI override)
      2. ~/.config/vocolith/config.yaml  (user config)
      3. ./config.yaml                   (current directory, dev convenience)
      4. <package_root>/config.yaml      (bundled defaults, last resort)
    """
    data: dict = {}

    # Find config file
    search_paths: list[Path] = []
    if config_path:
        search_paths.append(Path(config_path))
    search_paths += [
        Path.home() / ".config" / "vocolith" / "config.yaml",
        Path.cwd() / "config.yaml",
        Path(__file__).parent.parent / "config.yaml",
    ]

    for path in search_paths:
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            import logging
            logging.getLogger(__name__).debug("Config loaded from %s", path)
            break

    # Environment variable overrides (MD_LLM__MODEL -> llm.model)
    for key, val in os.environ.items():
        if key.startswith("MD_"):
            parts = key[3:].lower().split("__")
            _set_nested(data, parts, val)

    # Inject HF token from env if not in config
    if not data.get("transcription", {}).get("huggingface_token"):
        hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if hf_token:
            data.setdefault("transcription", {})["huggingface_token"] = hf_token

    cfg = AppConfig(**data)

    # Expand ~ in paths so callers always get absolute paths
    cfg.storage.profiles_dir = str(Path(cfg.storage.profiles_dir).expanduser())
    if cfg.pipeline.output_dir:
        cfg.pipeline.output_dir = str(Path(cfg.pipeline.output_dir).expanduser())
    if cfg.pipeline.debug_dir:
        cfg.pipeline.debug_dir = str(Path(cfg.pipeline.debug_dir).expanduser())
    if cfg.templates.user_templates_dir:
        cfg.templates.user_templates_dir = str(
            Path(cfg.templates.user_templates_dir).expanduser()
        )

    return cfg


def _set_nested(d: dict, keys: list[str], value: str) -> None:
    """Set a nested dict value from a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

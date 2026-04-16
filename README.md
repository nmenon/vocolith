[//]: # (SPDX-License-Identifier: GPL-2.0-only)
[//]: # (Copyright \(C\) 2026 Texas Instruments Incorporated - https://www.ti.com/)

<p align="center">
  <img src="vocolith_icon.png" width="120" alt="Vocolith icon" />
</p>

<h1 align="center">Vocolith</h1>
<p align="center"><em>Local meeting transcription with speaker identification and AI-powered notes</em></p>
<p align="center">
  <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="version 0.1.0" />
  <img src="https://img.shields.io/badge/status-WIP-yellow" alt="Work In Progress" />
  <img src="https://img.shields.io/badge/license-GPL--2.0-green" alt="GPL-2.0" />
</p>

> **⚠️ Work In Progress — v0.1.0**
> This project is under active development. APIs, CLI flags, config keys, and
> storage formats may change without notice between versions. Not recommended
> for production use yet.

---

Vocolith processes meeting recordings (MP4, WebM) entirely on your desktop — no cloud ASR, no data leaving your machine (except transcript text for LLM notes).

## Features

- **Transcription**: WhisperX (large-v2) with word-level timestamps, forced or auto language detection
- **Speaker diarization**: pyannote.audio — who speaks when
- **Voice speaker ID**: resemblyzer 256-dim d-vectors, persistent across meetings via ChromaDB; per-segment audio loading for reliable embeddings on long recordings
- **Transcript name mining**: extracts candidate speaker names from conversation patterns automatically
- **Video frame OCR**: temporal correlation of participant names from Zoom/WebEx/Teams overlays
- **Terminology boost**: OCR extracts domain terms (LPDDR, ADAS, etc.) as Whisper hotwords
- **Addressee inference**: "Alice, can you...?" → next speaker is likely Alice
- **Speaker confirmation wizard**: review every identification before it's committed; correct or split diarization errors segment-by-segment; bulk-assign with fast-path prompt or `r NAME` shortcut; color-coded confidence (green/yellow/red)
- **DB consistency check**: `vocolith profiles check` validates pairwise embedding similarities and flags false-positive risks with remediation advice before each run
- **AI meeting notes**: structured notes via LLM (OpenAI-compatible, or local Ollama)
- **Multiple note formats**: generates executive summary, email notes, and detailed technical notes in one run
- **Jinja2 templates**: 10 built-in styles; drop `.md.j2` + `.guidance.txt` to add your own
- **Meeting type aliases**: name recurring meeting patterns once, apply with `--meeting-type`
- **Anti-hallucination**: source quotes, timestamp citations, LLM verification pass, null-first fields

---

## Quick Start

```bash
# 1. System dependencies
sudo apt-get install ffmpeg libopenblas-dev

# 2. Python dependencies
pip install -r requirements.txt
pip install git+https://github.com/m-bain/whisperX.git
pip install pyannote.audio

# 3. Install vocolith
pip install -e .

# 4. Set credentials
export OPENAI_API_KEY=sk-...
export OPENAI_API_MODEL=gpt-4o-mini
export OPENAI_API_PROVIDER=https://api.openai.com/v1   # or any OpenAI-compatible URL
export HUGGINGFACE_TOKEN=hf_...                        # for speaker diarization

# 5. Accept the pyannote model licence (one-time, in browser)
#    https://huggingface.co/pyannote/speaker-diarization-community-1

# 6. Process a meeting
vocolith process meeting.mp4
```

---

## Output

Each run creates a timestamped folder **next to the input video** by default:

```
~/Videos/standup_20260409_1430/
├── executive_summary.md                    # 3-5 sentence leadership recap
├── email_notes.md                          # paste-ready email
├── detailed_technical_discussion_notes.md  # full technical notes with source quotes
├── transcript.md                           # timestamped transcript with speaker labels
└── debug/
    ├── audio_raw.wav                       # extracted audio
    ├── audio_clean.wav                     # denoised audio
    └── diarization.json                    # manifest: paths + all segments
```

Override locations:
```bash
vocolith process meeting.mp4 --output-dir ~/notes/   # put files here instead
vocolith process meeting.mp4 --debug-dir /tmp/vocolith/  # debug artifacts here
```

`debug/` defaults to `/tmp/vocolith` (configured in `config.yaml`), keeping your output folder clean.

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `OPENAI_API_KEY` | API key for LLM notes | Yes (for notes) |
| `OPENAI_API_MODEL` | Model name e.g. `gpt-4o-mini`, `claude-sonnet-4-5` | No |
| `OPENAI_API_PROVIDER` | Full base URL or short name: `openai` \| `anthropic` \| `ollama` | No |
| `ANTHROPIC_AUTH_TOKEN` | Fallback API key if `OPENAI_API_KEY` not set | No |
| `HUGGINGFACE_TOKEN` | pyannote diarization model download | Yes (speaker labels) |
| `HTTP_PROXY` / `HTTPS_PROXY` | Corporate proxy for LLM calls | If behind proxy |

---

## Full CLI Reference

### `vocolith process` — main command

```
vocolith process [OPTIONS] VIDEO
```

| Option | Short | Description |
|---|---|---|
| `--output-dir PATH` | `-o` | Exact directory for transcript + notes (default: `<video_dir>/<stem>_<ts>/`) |
| `--debug-dir PATH` | | Intermediate files: WAVs, frames, diarization.json (default: from config) |
| `--config PATH` | `-c` | Path to a custom config.yaml |
| `--model-size TEXT` | | Whisper model: `tiny\|base\|small\|medium\|large-v2\|auto` |
| `--language TEXT` | | Force ISO 639-1 code e.g. `en` (default: auto-detect) |
| `--llm-model TEXT` | | Override LLM model for this run |
| `--attendees TEXT` | `-a` | Comma-separated expected attendees: `"Alice Smith, Bob Jones"` |
| `--template TEXT` | `-t` | Add an extra template to this run |
| `--meeting-type TEXT` | `-m` | Meeting type alias (overrides `templates.run`) |
| `--confirm` / `--no-confirm` | | Show speaker ID review wizard before writing transcript (default: on) |
| `--identify` | | After processing, run the identify wizard for unresolved Speaker_N speakers |
| `--dry-run` | | Transcribe only; skip LLM note generation |
| `--no-faces` | | Skip face recognition |
| `--no-ocr` | | Skip OCR name extraction |
| `--verbose` | `-v` | Debug logging |

**Examples:**

```bash
# Standard run — output next to video, confirmation wizard on
vocolith process meeting.mp4

# Audio-only meeting (no camera, no screen share)
vocolith process meeting.mp4 --no-faces --no-ocr

# Hint who will be in the meeting (helps speaker ID before voices are enrolled)
vocolith process meeting.mp4 --attendees "Alice Smith, Bob Jones, Carol Wu"

# Use a meeting type alias for predictable template sets
vocolith process meeting.mp4 --meeting-type standup
vocolith process meeting.mp4 --meeting-type design_review

# Add one extra template on top of the default set
vocolith process meeting.mp4 --template brainstorm

# Skip confirmation wizard (batch processing, CI)
vocolith process meeting.mp4 --no-confirm

# Transcribe only — no LLM calls
vocolith process meeting.mp4 --dry-run

# Force English, use smaller model, direct output
vocolith process meeting.mp4 --language en --model-size small --output-dir ~/today/

# Local LLM via Ollama
vocolith process meeting.mp4 --llm-model mistral:7b
```

---

### `vocolith identify` — post-run speaker wizard

Run after a previous `vocolith process` when speakers remain as `Speaker_N`:

```
vocolith identify OUTPUT_DIR [--config PATH]
```

The wizard shows each unresolved speaker with 3 sample transcript snippets and offers:

| Key | Action |
|---|---|
| `[Enter]` | Accept auto-identified name (or skip for `Speaker_N`) |
| `[p]` | Play audio clip via `ffplay` |
| `[f]` | Show video frame at that timestamp via system image viewer |
| `[n]` | Cycle to next set of samples |
| `[g]` | **Go segment-by-segment** — for diarization errors where one label contains two people |
| `[s]` | Skip this speaker |
| type name | Rename to what you type |

**Segment-by-segment mode** (`[g]`): when sample 1 sounds like Alice but sample 2 sounds like Bob, go through each segment individually and assign a different name per segment. The wizard detects the split and creates a new speaker profile for the second person.

Also available inline during `process`:

```bash
# Run wizard automatically after processing
vocolith process meeting.mp4 --identify
```

---

### `vocolith profiles` — manage speaker profiles

```bash
vocolith profiles list              # show all stored speakers with meeting count
vocolith profiles check             # validate embedding quality + flag false-positive risks
vocolith profiles play "Alice Smith"                 # play stored voice sample
vocolith profiles rename "Speaker_1" "Alice Smith"   # name an unidentified speaker
vocolith profiles delete "Alice Smith"               # remove speaker + all embeddings
vocolith profiles clear             # wipe ALL profiles (asks for confirmation)
```

#### `vocolith profiles check` — DB consistency validation

Run before processing a new meeting to catch embedding quality problems early:

```
Profile health
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┓
┃ Name           ┃ Meetings ┃ Status ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━┩
│ Alice Smith    │ 3        │ OK     │
│ Bob Jones      │ 1        │ OK     │
└────────────────┴──────────┴────────┘

Pairwise similarity warnings
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Speaker A    ┃ Speaker B   ┃ Similarity ┃ Risk                ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ Alice Smith  │ Bob Jones   │ 0.9372     │ ✗ FP risk (>=0.92)  │
└──────────────┴─────────────┴────────────┴─────────────────────┘

ERRORS (1):
  • 'Alice Smith' vs 'Bob Jones' sim=0.9372 — above HIGH threshold
Recommended remediation:
  FP 0.9372  'Alice Smith' ↔ 'Bob Jones'
    → Delete weaker profile: vocolith profiles delete "Bob Jones"
    → Re-enroll 'Bob Jones' in next meeting run
```

Exit code 0 = clean. Exit code 1 = errors found (FP risks, orphan profiles, sparse embeddings). Thresholds are configurable: `--warn 0.89 --fail 0.92`.

Speaker voice embeddings are stored in ChromaDB at `~/.cache/vocolith/`. Once you rename a `Speaker_N`, that voice is automatically recognised in future meetings.

---

### `vocolith templates` — manage note templates

```bash
vocolith templates list   # show all templates with full path, source (built-in/user), active default
```

---

### `vocolith meeting-types` — manage meeting type aliases

```bash
vocolith meeting-types list   # show all aliases with template lists and descriptions
```

---

## Speaker Identification

### How it works

Every diarized label (`SPEAKER_00`, `SPEAKER_01`, …) goes through up to 6 strategies, stopping at the first confident match:

```
For each SPEAKER_N label:
│
├─ 1. Voice HIGH (≥0.92 cosine similarity)
│     Biometric match against stored profile.
│     → Accept immediately. No other strategy checked.
│     auto_confirmed=True in wizard (green badge, fast Enter).
│
│   NO high-confidence voice match:
│
├─ 2. OCR name — temporal correlation
│     Which names were visible in video frames during this speaker's turns?
│     Accepts name if visible in ≥50% of frames during speaker's windows.
│
├─ 3. Addressee inference
│     Scans transcript for "Alice, can you...?" patterns.
│     Candidate names come from: OCR + --attendees + names mined from transcript.
│     Requires pattern to repeat ≥ addressee_min_votes times (default: 2).
│
├─ 4. Voice STANDARD (≥0.89)
│     Same voice comparison, lower confidence threshold.
│     Only accepted if strategies 2 and 3 both failed.
│
├─ 5. Multi-signal agreement
│     If OCR/addressee found a name AND voice (0.85+) agrees → boost confidence.
│
├─ 6. Face recognition (optional, disabled by default)
│
└─ 7. Fallback → "Speaker_N"
       Voice embedding IS saved to ChromaDB.
       Next meeting: strategy 1 or 4 will match as "Speaker_N".
       Rename with: vocolith profiles rename "Speaker_N" "Alice Smith"
```

### Name candidate sources

Addressee inference needs candidate names to search for. Sources (merged automatically):

1. **OCR** — names read from Zoom/WebEx/Teams video tiles
2. **`--attendees` flag** — names you provide on the command line
3. **Transcript mining** — automatic extraction using:
   - Addressee patterns: `"Alice, can you..."`, `"Thanks Bob"`, `"ask Carol"`
   - Repeated proper nouns: capitalised words appearing 3+ times across speaker turns

This means addressee inference works even on a fresh meeting with cameras off and no `--attendees` — as long as participants address each other by name in conversation.

### The confirmation wizard

By default (`confirm_auto_identified: true`), after speaker resolution and before the transcript is written, the wizard shows every identification for review:

```
Speaker Confirmation — 4 speaker(s)  (1 high-confidence, 3 need review)

╭─ SPEAKER_00 → Alice Chen  ✓ HIGH CONFIDENCE  via voice(high,0.95) ─╮  ← green
│ [00:02]  So Dilanka, we had more than a number of discussions...         │
│ [00:53]  Is that correct, Bob?                                         │
╰──────────────────────────────────────────────────────────────────────────╯
[Enter] Accept · type to correct · [p] Play · [f] Frame · [g] Segment-by-segment · [s] Skip

╭─ SPEAKER_01 → Bob Martinez  ?  via ocr ─╮  ← yellow
│ [00:57]  Yeah, that's correct.            │
╰───────────────────────────────────────────╯
```

- **Voice HIGH ≥0.92** (green border): Enter confirms instantly
- **Voice STD ≥0.89** (yellow border): shown for review; may be a false match — play clips before confirming
- **OCR / addressee** (orange border): text-based signal; less reliable for identity
- **`[g]` segment-by-segment**: use when one label contains two different people
  - **Fast-path**: shown first — Enter to assign all to current name, type a name to bulk-assign all to it, `n` to review one-by-one
  - **`r NAME`**: at any segment, type `r Alice Smith` to assign current + all remaining segments to that person

Disable for automated/batch runs:

```bash
vocolith process meeting.mp4 --no-confirm
# or in config.yaml: speaker_resolution.confirm_auto_identified: false
```

### What to do with Speaker_N labels

```bash
# After a run, any unresolved speakers get Speaker_N labels
vocolith profiles list               # see all stored speakers

# Option 1: rename immediately
vocolith profiles rename "Speaker_1" "Alice Smith"

# Option 2: run the identify wizard (shows audio + video evidence)
vocolith identify ./standup_20260409_1430/

# Option 3: re-process the same video — voice is now enrolled, will match
vocolith process meeting.mp4 --attendees "Alice Smith, Bob Jones"
```

---

## Note Templates

### Built-in templates

| Template | Best for |
|---|---|
| `executive_summary` | 3–5 sentences for leadership — outcome + next steps, no jargon |
| `email_notes` | Paste-ready email recap — decisions + actions |
| `detailed_technical_discussion_notes` | Full technical notes — rationale, specs, verbatim source quotes |
| `standard` | Balanced general-purpose notes |
| `standup` | Daily standups — Yesterday / Today / Blockers per person |
| `design_review` | Technical design reviews — proposals, decisions, rationale |
| `one_on_one` | 1:1 meetings — feedback + goals progress |
| `brainstorm` | Ideation sessions — ideas generated and shortlisted |
| `email_summary` | Short variant of `email_notes` |
| `detailed` | Short name for `detailed_technical_discussion_notes` |

### Multi-template output (default)

Every run generates multiple note files. The default set is configured in `templates.run`:

```
standup_20260409_1430/
├── executive_summary.md
├── email_notes.md
├── detailed_technical_discussion_notes.md
└── transcript.md
```

Change the default set in `~/.config/vocolith/config.yaml`:

```yaml
templates:
  run:
    - executive_summary
    - email_notes
```

Or use a meeting type alias to select a per-meeting-pattern set.

### Meeting type aliases

Define recurring meeting patterns once, use with `--meeting-type`:

```bash
vocolith process meeting.mp4 --meeting-type standup
vocolith process meeting.mp4 --meeting-type design_review
vocolith meeting-types list    # see all aliases
```

Define your own in `~/.config/vocolith/config.yaml`:

```yaml
meeting_types:
  lpddr_debug:
    description: "LPDDR debug session with Synopsis"
    templates:
      - detailed_technical_discussion_notes
      - email_notes

  customer_call:
    description: "External partner call"
    templates:
      - executive_summary
      - email_notes
```

`--meeting-type` overrides `templates.run` for that run only.

### Creating a custom template

Every template is two files in `~/.config/vocolith/templates/`:

```
~/.config/vocolith/templates/
├── my_template.md.j2        # Jinja2 layout — controls how notes look
└── my_template.guidance.txt # Plain text LLM instructions — controls what gets extracted
```

#### Layout file (`my_template.md.j2`)

Available Jinja2 variables:

| Variable | Type | Content |
|---|---|---|
| `title` | `str` | Inferred meeting title |
| `date` | `str` | ISO date (YYYY-MM-DD) |
| `duration_minutes` | `int` | Meeting length |
| `attendees` | `list[str]` | Participant names |
| `summary` | `str` | LLM-generated summary |
| `agenda_items` | `list[str]` | Main topics discussed |
| `key_topics` | `list[str]` | 5–10 key terms or themes |
| `decisions` | `list[Decision]` | `.description`, `.decided_by`, `.context`, `.source_quote`, `.timestamp` |
| `action_items` | `list[ActionItem]` | `.description`, `.assignee`, `.due_date`, `.priority`, `.source_quote`, `.timestamp` |
| `follow_up_questions` | `list[str]` | Unresolved questions |
| `extra` | `dict` | Template-specific fields populated by guidance |

Minimal example:

```jinja2
# {{ title }} — {{ date }}
Attendees: {{ attendees | join(", ") }}

{{ summary }}

{% for item in action_items %}
- [ ] {{ item.description }}{% if item.assignee %} ({{ item.assignee }}){% endif %}
{% endfor %}
```

#### Guidance file (`my_template.guidance.txt`)

Plain text injected into the LLM system prompt. Controls what the LLM extracts.

**Include:** tone, length, what to focus on, what to put in `extra.*`
**Do not include:** formatting instructions (that's the `.md.j2`), invented content, prompts over ~200 words

Examples:

```
# email_notes.guidance.txt
Write the summary as 2-3 plain sentences for a team email.
Focus on what was decided and who needs to do what next.
Keep the entire summary under 80 words.
```

```
# detailed_technical_discussion_notes.guidance.txt
Write a thorough technical summary (4-6 paragraphs).
Include exact technical terms, version numbers, and spec references mentioned.
Capture the engineering reasoning behind decisions, not just the outcome.
Flag unvalidated assumptions with [?].
```

```bash
vocolith process meeting.mp4 --template my_template
vocolith templates list   # verify it appears
```

User templates override built-in templates of the same name.

---

## Accuracy & Hallucination Prevention

Vocolith applies multiple layers to keep notes factual:

| Layer | What it does |
|---|---|
| Low temperature (`0.1`) | Near-deterministic extraction |
| Hard system prompt | 10 explicit rules: omit rather than guess, `null` over invented dates, `[?]` when uncertain |
| Source quotes | Every `decision` and `action_item` carries a verbatim `source_quote` and approximate `timestamp` |
| Verification pass | Second LLM call cross-checks every claim against up to 32 000 chars of transcript |
| Null-first fields | `assignee`, `due_date`, `priority` are `null` unless explicitly stated in the meeting |

### What the LLM is instructed never to do

- Invent action item owners not explicitly assigned in the meeting
- Add due dates unless a specific date was spoken
- Set priority unless urgency was stated ("ASAP", "before Friday", etc.)
- Fill gaps with "common sense" completions

### `[?]` in output

If the LLM is uncertain it appends `[?]` to the field — e.g. `"Alice will update the spec [?]"`. Review these before sharing.

---

## GPU / CPU

Whisper model is selected automatically based on available VRAM:

| VRAM | Whisper model | Compute | Batch |
|---|---|---|---|
| ≥3.5 GB | `large-v2` | float16 | 16 |
| ≥2 GB | `medium` | float16 | 8 |
| CPU only | `small` | int8 | 4 |

Override: `vocolith process meeting.mp4 --model-size small`

---

## Configuration

### Config file search order

1. `--config <path>` — explicit CLI flag (highest priority)
2. `~/.config/vocolith/config.yaml` — **user config** (recommended for personalisation)
3. `./config.yaml` — current directory (dev/project-specific)
4. `<package>/config.yaml` — bundled defaults (fallback)

Create a user config:

```bash
mkdir -p ~/.config/vocolith
cp $(python3 -c "import vocolith, pathlib; print(pathlib.Path(vocolith.__file__).parent.parent / 'config.yaml')") \
   ~/.config/vocolith/config.yaml
```

### `MD_` environment variable overrides

Every config key is overridable at runtime without editing the file:

```bash
MD_LLM__MODEL=gpt-4o vocolith process meeting.mp4
MD_STORAGE__PROFILES_DIR=/data/vocolith vocolith process meeting.mp4
MD_SPEAKER_RESOLUTION__CONFIRM_AUTO_IDENTIFIED=false vocolith process meeting.mp4
```

Format: `MD_<SECTION>__<KEY>=value` (double underscore as separator).

### All settings

#### `pipeline`
| Key | Default | Description |
|---|---|---|
| `enable_face_recognition` | `false` | Enable face recognition (requires `pip install vocolith[faces]`) |
| `enable_ocr` | `true` | Extract names and terminology from video frames |
| `parallel_video_audio` | `false` | Run audio first (GPU), then video/OCR — avoids VRAM contention on cards <8 GB |
| `output_dir` | `null` | `null` = create `<stem>_<ts>/` next to input video; set a path to collect all runs in one place |
| `debug_dir` | `"/tmp/vocolith"` | Intermediate files: WAVs, sampled frames, diarization.json |

#### `audio`
| Key | Default | Description |
|---|---|---|
| `sample_rate` | `16000` | WAV sample rate for ffmpeg extraction (Hz) |
| `channels` | `1` | Mono recommended for transcription |
| `format` | `wav` | Intermediate audio format |

#### `denoiser`
| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Spectral noise reduction before transcription |
| `stationary` | `false` | Non-stationary mode handles variable meeting noise |

#### `transcription`
| Key | Default | Description |
|---|---|---|
| `model_size` | `auto` | `auto` picks by VRAM, or `tiny/base/small/medium/large-v2` |
| `language` | `"en"` | Force language code; `null` = auto-detect (risky on compressed audio — auto-detect can misfire) |
| `batch_size` | `auto` | `auto` = 16 (GPU) / 4 (CPU) |
| `compute_type` | `auto` | `auto` = float16 (GPU) / int8 (CPU) |
| `huggingface_token` | `null` | Prefer `HUGGINGFACE_TOKEN` env var |

#### `diarization`
| Key | Default | Description |
|---|---|---|
| `enabled` | `true` | Speaker diarization (requires `HUGGINGFACE_TOKEN`) |
| `min_speakers` | `1` | Minimum speakers hint |
| `max_speakers` | `10` | Maximum speakers hint — effective value is `max(len(attendees), max_speakers)` so the attendee list never forces a cap below what's configured |
| `suppress_warnings` | `true` | Suppress cosmetic pyannote warnings (TF32, std() degrees-of-freedom) |

#### `frame_sampling`
| Key | Default | Description |
|---|---|---|
| `interval_seconds` | `5` | Sample one frame every N seconds |
| `change_threshold` | `0.002` | Skip frame if mean pixel diff < this (screen shares: ~0.001–0.005) |
| `min_frames` | `6` | Always keep at least this many frames |
| `top_strip_pct` | `0.25` | Top fraction of frame for name overlays (Zoom/WebEx thumbnails) |
| `bottom_strip_pct` | `0.15` | Bottom fraction |

#### `ocr`
| Key | Default | Description |
|---|---|---|
| `languages` | `["en"]` | EasyOCR languages e.g. `["en", "fr"]` |
| `confidence_threshold` | `0.5` | Minimum EasyOCR confidence |
| `min_name_freq` | `2` | Name must appear in this many frames to count |
| `extract_terminology` | `true` | Extract domain terms as Whisper hotwords |
| `ocr_workers` | `0` | CPU worker processes. `0` = auto (½ × CPU cores, capped at 6). `1` = sequential. GPU path sets workers to `ocr_workers - 1` for CPU side |
| `ocr_gpu_worker` | `true` | Use GPU reader in main process when CUDA is available. Runs concurrently with CPU workers |
| `ocr_gpu_frame_ratio` | `0.0` | Fraction of frames assigned to GPU in hybrid mode. `0.0` = auto-balance GPU (1 s/frame) vs CPU (7 s/frame) wall time |
| `ocr_noise_freq_threshold` | `0.8` | Discard names visible in >80% of frames (browser tabs, persistent UI). Set `1.0` to disable |
| `ui_blacklist` | *(list)* | Meeting platform UI labels to reject |

#### `face_recognition`
| Key | Default | Description |
|---|---|---|
| `tolerance` | `0.6` | Face comparison distance threshold |
| `model` | `hog` | `hog` (CPU) or `cnn` (GPU, better for small tiles) |
| `min_face_height_px` | `30` | Ignore faces smaller than this |

#### `speaker_resolution`
| Key | Default | Description |
|---|---|---|
| `voice_similarity_threshold_high` | `0.92` | Voice match ≥ this → accept immediately, skip all other strategies |
| `voice_similarity_threshold` | `0.89` | Voice match ≥ this → accept if OCR and addressee are silent |
| `ocr_match_confidence` | `0.70` | OCR-to-speaker temporal overlap threshold |
| `face_match_threshold` | `0.60` | Face cosine similarity threshold |
| `addressee_min_votes` | `2` | Pattern must repeat this many times before assigning identity |
| `multi_signal_agree_boost` | `true` | Voice (medium) + OCR/addressee agreeing on same person → higher confidence |
| `confirm_auto_identified` | `true` | Show all identifications for review before writing transcript; disable with `--no-confirm` |

#### `llm`
| Key | Default | Description |
|---|---|---|
| `model` | `gpt-4o-mini` | Model name; overridden by `OPENAI_API_MODEL` |
| `use_local` | `false` | Use local Ollama |
| `local_base_url` | `http://localhost:11434/v1` | Ollama base URL |
| `local_model` | `mistral:7b` | Ollama model name |
| `max_tokens` | `32000` | Max LLM response tokens |
| `temperature` | `0.1` | Low = deterministic factual extraction |
| `max_transcript_chars` | `100000` | Chunk transcript if longer |
| `chunk_overlap_chars` | `500` | Overlap between chunks |
| `verify_notes` | `true` | Second LLM pass to remove unsupported claims |
| `ssl_verify` | `true` | Set `false` for self-signed / corporate proxy endpoints |

#### `storage`
| Key | Default | Description |
|---|---|---|
| `profiles_dir` | `~/.cache/vocolith` | Root for speaker profiles (SQLite + ChromaDB) |
| `db_filename` | `speakers.db` | SQLite metadata filename |
| `chroma_dir` | `chroma` | ChromaDB subdirectory |

#### `templates`
| Key | Default | Description |
|---|---|---|
| `default` | `standard` | Fallback when `run` is empty and no `--template` given |
| `run` | `[executive_summary, email_notes, detailed_technical_discussion_notes]` | Templates generated every run |
| `user_templates_dir` | `~/.config/vocolith/templates` | Directory for custom `.md.j2` files |

#### `meeting_types`

Named aliases mapping to lists of templates. Each entry:

| Key | Description |
|---|---|
| `templates` | Template names to generate |
| `description` | Label shown in `vocolith meeting-types list` |

Built-in aliases: `standup`, `design_review`, `customer_call`, `team_sync`.

---

## Installation Notes

### System dependencies

```bash
sudo apt-get install ffmpeg libopenblas-dev
```

### Optional: face recognition

```bash
sudo apt-get install cmake liblapack-dev
pip install dlib face-recognition
# Then enable in config.yaml:
# pipeline.enable_face_recognition: true
```

### Optional: audio playback in identify wizard

```bash
# ffplay is part of ffmpeg (usually already installed):
sudo apt-get install ffmpeg
```

---

## Storage & Privacy

All speaker data is stored **locally** on your machine:

| Path | Content |
|---|---|
| `~/.cache/vocolith/speakers.db` | Speaker names, aliases, meeting history |
| `~/.cache/vocolith/chroma/` | Voice embeddings (resemblyzer d-vectors) |
| `<output_dir>/*.md` | Transcripts and notes |
| `/tmp/vocolith/` | Debug artifacts — cleared on reboot |

**What leaves your machine:** transcript text is sent to the configured LLM endpoint for note generation. Voice embeddings are never transmitted.

```bash
vocolith profiles clear   # wipe all speaker profiles and embeddings
```

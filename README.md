# Cat.li

**Cat.li** is an end-to-end drum analysis and music education platform. Paste any YouTube URL, and Cat.li isolates the drum track, transcribes every hit using a neural drum transcription model, renders the result as sheet music notation and an event scatter plot, and generates a personalized practice lesson using a local LLM.

---

## Architecture

```
YouTube URL
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI Backend  (Python 3.11 · CUDA · RTX 5070)           │
│                                                              │
│  1. yt-dlp + ffmpeg   →  mono WAV @ 44 100 Hz               │
│  2. Demucs htdemucs   →  isolated drums stem                 │
│  3. ADTOF Frame_RNN   →  timestamped hit events (MIDI JSON)  │
│  4. Ollama qwen2.5    →  personalized drum lesson            │
│                                                              │
└──────────────────┬───────────────────────────────────────────┘
                   │  Supabase  drum_transcriptions
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  Next.js 16 Frontend  (React 19 · Tailwind v4)              │
│                                                              │
│  usePipeline hook       →  poll 4-step progress             │
│  useTranscription hook  →  fetch events + metadata          │
│  SheetMusic             →  MusicXML → OSMD notation          │
│  DrumNotation           →  SVG event scatter plot            │
│  MusicTeacherLesson     →  react-markdown lesson card        │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Audio download | `yt-dlp` + `ffmpeg` |
| Drum stem separation | Meta Demucs `htdemucs` |
| Drum transcription | ADTOF-pytorch `Frame_RNN` (F1 ~88.5%) |
| MIDI export | `mido` |
| AI lesson generation | Ollama · `qwen2.5:14b` |
| Backend API | FastAPI + Uvicorn |
| Database | Supabase (PostgreSQL + JSONB) |
| Frontend framework | Next.js 16 (Turbopack) · React 19 |
| CSS | Tailwind CSS v4 |
| Sheet music notation | MusicXML 3.1 (stdlib) + OpenSheetMusicDisplay |
| Event visualization | SVG scatter plot (inline React) |
| Markdown rendering | `react-markdown` |
| GPU | NVIDIA RTX 5070 · CUDA 12.x · PyTorch 2.x |

---

## Project Structure

```
drum-transcriber/
├── backend/
│   ├── main.py                          # FastAPI app, CORS, route registration
│   ├── requirements.txt
│   └── app/
│       ├── core/
│       │   ├── config.py                # Env loading, CUDA device, output dirs
│       │   ├── job_store.py             # In-memory pipeline job state
│       │   └── supabase_client.py       # Admin client (service role key)
│       ├── api/routes/
│       │   ├── pipeline.py              # POST /pipeline/start  GET /pipeline/{id}
│       │   ├── audio.py                 # POST /audio/extract  /separate  /process
│       │   ├── analysis.py              # POST /audio/analyze
│       │   ├── teacher.py               # POST /audio/teach
│       │   └── notation.py              # GET  /notation/musicxml/{job_id}  /midi/{job_id}
│       ├── models/
│       │   └── adtof/
│       │       └── transcriber.py       # ADTOF Frame_RNN wrapper (lazy singleton)
│       └── services/
│           ├── audio_extractor.py       # yt-dlp subprocess wrapper
│           ├── stem_separator.py        # Demucs async wrapper
│           ├── drum_analyzer.py         # Full transcription pipeline
│           ├── transcription_store.py   # Supabase update-or-insert
│           ├── musicxml_builder.py      # MusicXML 3.1 generator (stdlib only)
│           └── music_teacher.py         # Semantic log builder + Ollama lesson
├── frontend/
│   ├── app/
│   │   ├── page.tsx                     # Main page (URL input → analysis view)
│   │   ├── layout.tsx
│   │   └── globals.css                  # Tailwind v4 + Cat.li pastel tokens
│   ├── components/
│   │   ├── SheetMusic.tsx               # OSMD sheet music renderer
│   │   ├── DrumNotation.tsx             # SVG event scatter plot (BD/SD/TT/HH/CY+RD)
│   │   ├── MusicTeacherLesson.tsx       # Markdown lesson on notepad background
│   │   ├── PipelineProgress.tsx         # 4-step progress bar
│   │   └── CatLiLogo.tsx               # SVG logo
│   ├── hooks/
│   │   ├── usePipeline.ts               # Poll /pipeline/{id} status
│   │   └── useTranscription.ts          # Fetch transcription by URL or job_id
│   └── lib/
│       ├── supabase.ts                  # Browser Supabase client
│       └── api.ts                       # API wrapper + MIDI download helper
└── adtof_pytorch/                       # Optional local ADTOF checkout for development
```

---

## Pipeline — Step by Step

### Step 1 — Audio Extraction

`POST /api/audio/extract`

`audio_extractor.py` spawns `yt-dlp` to download the best audio-only stream and pipes it into `ffmpeg`, producing a **16-bit mono PCM WAV at 44 100 Hz** under `backend/audio_files/<job_id>.wav`.

### Step 2 — Drum Stem Separation

`POST /api/audio/separate`

`stem_separator.py` loads Meta's **Demucs `htdemucs`** model (cached globally after first load) and separates the mix into four stems: drums, bass, vocals, other. Only the drums stem is written to `backend/stems/<job_id>_drums.wav`. The separation runs in `asyncio.run_in_executor` to avoid blocking FastAPI's event loop.

### Step 3 — ADTOF Transcription

`POST /api/audio/analyze`  ·  implemented in `drum_analyzer.py`

This is the core analysis step. It runs five sequential operations on the drums stem:

**3a. BPM + time signature detection** (`librosa.beat.beat_track`)

Detects tempo from the drum stem. Applies two corrections:
- Halves/doubles the result to keep BPM in the 60–200 range
- Checks a 3/4× candidate to catch librosa's known 4/3× tempo lock-on bias

If the auto-detection is wrong, the user can supply manual overrides via the frontend (BPM and/or time signature fields). When `override_bpm` is set, auto-detected tempo is replaced and beat times are recomputed from the override value so the quantisation grid stays correct. `override_beats_per_bar` and `override_beat_unit` replace the numerator/denominator respectively. All three are optional and independent — any combination works.

**3b. ADTOF Frame_RNN inference** (`app/models/adtof/transcriber.py`)

Runs the pre-trained ADTOF Frame_RNN neural network on the full drum stem. ADTOF is a CRNN trained on 359 hours of real acoustic drum recordings. It outputs onset times per instrument class:

| ADTOF class | GM MIDI note |
|---|---|
| Bass drum (35) | 36 (Kick) |
| Snare drum (38) | 38 (Snare) |
| Tom (47) | 45 (Low Tom) |
| Hi-hat (42) | 42 (Closed HH) |
| Cymbal (49) | 49 (Crash/Ride) |

The model weights (~30 MB) are bundled inside the `adtof_pytorch` package — no external download needed.

**3c. Velocity from RMS** (`_rms_at`, `_rms_to_velocity`)

For each onset, a 40 ms window of the drum stem is extracted and its RMS energy computed. RMS values are min-max normalized per instrument class and mapped to MIDI velocity (30–115 range, tuned per class).

**3d. 16th-note quantization** (`_quantize`)

Events are snapped to a uniform 16th-note grid anchored to the detected beat phase:
- Grid unit: `median_IBI / 4` (inter-beat-interval from librosa beat times)
- Phase anchor: `beat_times[0] % grid_unit` — the sub-beat offset at which the grid starts
- Duplicate events at the same (time, note) slot are deduplicated

The `grid_phase` value (not `beat_times[0]`) is stored in metadata so the frontend can correctly align events to measure boundaries.

**3e. Ghost note tagging** (`_tag_ghosts`)

Snare hits below the 20th percentile of snare velocities are flagged as ghost notes.

**3f. MIDI export** (`_build_midi`)

A standard MIDI file is built with `mido` using tempo and time signature metadata. Each event maps to a `note_on` / `note_off` pair on GM percussion channel 9 with a 16th-note duration.

**Event JSON format** (stored in Supabase `events` JSONB column):
```json
{ "note": 36, "time": 0.406, "velocity": 95, "type": "kick", "ghost": false, "duration": 0.05 }
```

**Metadata JSON format** (stored in Supabase `metadata` JSONB column):
```json
{
  "bpm": 97.5,
  "time_signature": "4/4",
  "beats_per_bar": 4,
  "beat_unit": 4,
  "duration": 264.08,
  "event_count": 1721,
  "grid_phase": 0.098685,
  "grid_subdivisions": 4
}
```

### Step 4 — AI Lesson Generation

`POST /api/audio/teach`  ·  implemented in `music_teacher.py`

`build_semantic_log` converts the event list into a human-readable 16th-note grid (first 60 seconds), then `generate_lesson` sends it to a local **Ollama `qwen2.5:14b`** instance. The system prompt instructs the model to return four Markdown sections: groove analysis, rudiments, ghost note mastery, and coordination exercises. The lesson is persisted to the `exercises` column in Supabase.

---

## Frontend

### `usePipeline` hook

Polls `GET /api/pipeline/{id}` every 3 seconds until `status` is `"complete"` or `"error"`. Returns `{ status, step, pct, message, db_job_id }`.

### `useTranscription` hook

Queries Supabase's `drum_transcriptions` table using the browser anon key. Accepts a full YouTube URL (regex extracts the 11-char video ID for flexible `ilike` matching) or a direct `job_id`. Returns `{ data, loading, error }`.

### `SheetMusic` component

Fetches MusicXML from `GET /api/notation/musicxml/{job_id}`, then dynamically imports and renders it with **OpenSheetMusicDisplay (OSMD)**. Displays the YouTube video title (fetched via YouTube's oEmbed API) as the score title. Percussion staff uses two voices: Voice 1 stems-up (SD/HH/TOM/CYM) and Voice 2 stems-down (BD). Ghost notes are rendered with parenthesised noteheads per MusicXML 3.1 convention.

The MusicXML is generated server-side in `musicxml_builder.py` using only Python's stdlib `xml.etree` — no extra dependencies. Events are grouped into measures using the stored `grid_phase` offset, then serialised as a two-voice percussion score.

### `DrumNotation` component

Renders a horizontally scrollable SVG scatter plot mirroring the ADTOF visualization style:

| Row | GM notes | Symbol | Color |
|---|---|---|---|
| CY+RD | 49, 51 | ★ | Orange |
| HH | 42, 46 | × | Purple |
| TT | 45, 47, 48 | ● | Green |
| SD | 38, 40 | ● | Dark |
| BD | 35, 36 | ● | Blue |

Scale: 80 px/second. Ghost notes rendered as smaller, faded purple dots.

### `MusicTeacherLesson` component

Renders the `exercises` Markdown string using `react-markdown` with custom component renderers. Section headers (`h2`) become colour-coded banners; `pre` blocks render ASCII notation exercises in dark monospace.

### Export MIDI button

An **Export MIDI** button appears in the results meta-pill row once a transcription is loaded. Clicking it calls `downloadMidi(jobId)` from `lib/api.ts`, which fetches `GET /api/notation/midi/{job_id}` as a blob and triggers a browser download of `{job_id}.mid`.

The backend endpoint reconstructs the GM MIDI file on the fly from the stored events and metadata using `mido` (`_build_midi` in `drum_analyzer.py`). The file uses GM percussion channel 9 with correct tempo, time signature, and per-hit velocities.

---

## Supabase Schema

```sql
create table if not exists drum_transcriptions (
    id          uuid        primary key default gen_random_uuid(),
    job_id      text        not null,
    youtube_url text,
    event_count integer     not null,
    events      jsonb       not null,
    metadata    jsonb,
    exercises   text,
    created_at  timestamptz not null default now()
);
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- NVIDIA GPU with CUDA 12.x (CPU fallback works, significantly slower)
- [Ollama](https://ollama.ai) running locally with `qwen2.5:14b` pulled
- `ffmpeg` and `yt-dlp` on `PATH`

### Install

```bash
git clone <repo-url>
cd drum-transcriber

# Python environment
python -m venv venv
venv/Scripts/activate          # Windows
source venv/bin/activate        # macOS/Linux
pip install -r backend/requirements.txt   # includes ADTOF-pytorch + bundled weights

# Frontend
cd frontend && npm install && cd ..
```

### Environment variables

`drum-transcriber/.env`:
```env
SUPABASE_URL=https://<ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
```

`drum-transcriber/frontend/.env.local`:
```env
NEXT_PUBLIC_SUPABASE_URL=https://<ref>.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY=sb_publishable_...
```

### Run

```bash
# Terminal 1 — backend
cd backend && uvicorn main:app --reload

# Terminal 2 — frontend
cd frontend && npm run dev
```

Open **http://localhost:3000**, paste a YouTube URL, click **Analyze**.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/pipeline/start` | Start full 4-step pipeline (accepts optional `override_bpm`, `override_beats_per_bar`, `override_beat_unit`) |
| `GET` | `/api/pipeline/{id}` | Poll pipeline progress |
| `POST` | `/api/audio/extract` | Download YouTube audio → WAV |
| `POST` | `/api/audio/separate` | Demucs drum stem isolation |
| `POST` | `/api/audio/process` | Extract + separate in one call |
| `POST` | `/api/audio/analyze` | ADTOF transcription → events JSON (accepts optional BPM/time sig overrides) |
| `POST` | `/api/audio/teach` | Generate + save LLM lesson |
| `GET` | `/api/notation/musicxml/{job_id}` | Generate + stream MusicXML 3.1 |
| `GET` | `/api/notation/midi/{job_id}` | Generate + download GM MIDI file |

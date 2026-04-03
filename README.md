# Cat.li 🐱🥁

**Cat.li** is an end-to-end drum analysis and music education platform. It takes any YouTube URL, isolates the drum track, transcribes every hit to a timestamped event list, renders the result as sheet music notation, and generates a personalised lesson using a local LLM.

---

## Architecture Overview

```
YouTube URL
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  FastAPI Backend  (Python · CUDA · RTX 5070)            │
│                                                         │
│  1. yt-dlp + ffmpeg  →  16-bit mono WAV @ 44 100 Hz    │
│  2. Demucs htdemucs  →  isolated drums stem             │
│  3. DrumAnalyzer     →  timestamped hit events (JSON)   │
│  4. MusicTeacher     →  16th-note grid → Ollama lesson  │
│                                                         │
└────────────────┬────────────────────────────────────────┘
                 │  Supabase (drum_transcriptions table)
                 ▼
┌─────────────────────────────────────────────────────────┐
│  Next.js 16 Frontend  (React 19 · Tailwind v4)         │
│                                                         │
│  useTranscription hook  →  fetch by YouTube URL        │
│  DrumNotation           →  VexFlow staff + playhead    │
│  MusicTeacherLesson     →  react-markdown lesson card  │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Audio download | `yt-dlp` + `ffmpeg` |
| Source separation | Meta Demucs (`htdemucs`) |
| Drum transcription | `librosa` onset detection + NumPy FFT |
| AI lesson generation | Ollama · `qwen2.5:14b` |
| Backend API | FastAPI + Uvicorn |
| Database | Supabase (PostgreSQL + JSONB) |
| Frontend framework | Next.js 16 (Turbopack) · React 19 |
| CSS | Tailwind CSS v4 |
| Notation rendering | VexFlow 4 |
| Markdown rendering | react-markdown |
| GPU | NVIDIA RTX 5070 · CUDA 12.1 · PyTorch 2.x |

---

## Project Structure

```
drum-transcriber/
├── backend/
│   ├── main.py                         # FastAPI app, CORS, route registration
│   ├── requirements.txt
│   └── app/
│       ├── core/
│       │   ├── config.py               # Env loading, CUDA device detection
│       │   └── supabase_client.py      # Admin client (service role key)
│       ├── api/routes/
│       │   ├── audio.py                # /extract  /separate  /process
│       │   ├── analysis.py             # /analyze
│       │   └── teacher.py              # /teach  (Phase 3)
│       └── services/
│           ├── audio_extractor.py      # yt-dlp subprocess wrapper
│           ├── stem_separator.py       # Demucs async wrapper
│           ├── drum_analyzer.py        # FFT transcriber
│           ├── transcription_store.py  # Supabase insert + exercises update
│           └── music_teacher.py        # Grid builder + Ollama lesson generator
├── frontend/
│   ├── app/
│   │   ├── page.tsx                    # Main page (URL input → analysis view)
│   │   ├── layout.tsx
│   │   └── globals.css                 # Tailwind v4 + Cat.li pastel tokens
│   ├── components/
│   │   ├── CatLiLogo.tsx               # SVG cat-playing-drums logo
│   │   ├── DrumNotation.tsx            # VexFlow 2-measure staff + playhead
│   │   └── MusicTeacherLesson.tsx      # Markdown lesson on notepad background
│   ├── hooks/
│   │   └── useTranscription.ts         # Supabase fetch by URL or job_id
│   └── lib/
│       └── supabase.ts                 # Browser Supabase client
├── trigger_lesson.py                   # One-shot lesson generation script
└── verify_cuda.py                      # CUDA sanity check
```

---

## Phase 1 — Audio Extraction

**Endpoint:** `POST /api/audio/extract`

`audio_extractor.py` spawns `yt-dlp` as a subprocess to download the best audio-only stream from a YouTube URL and immediately pipes it into `ffmpeg` for conversion to a **16-bit mono PCM WAV at 44 100 Hz**. The output is saved under a UUID `job_id` in `backend/audio_files/`.

```python
cmd = [
    YTDLP_PATH, "--extract-audio", "--audio-format", "wav",
    "--postprocessor-args", "ffmpeg:-ar 44100 -ac 1 -sample_fmt s16",
    ...
]
```

---

## Phase 2 — Drum Transcription

### Stem Separation — `POST /api/audio/separate`

`stem_separator.py` loads Meta's **Demucs `htdemucs`** model (cached globally after first load) and runs source separation on the raw WAV. The model outputs four stems: drums, bass, vocals, other. Only the drums stem is retained, written to `backend/stems/<job_id>_drums.wav`.

The long-running separation is wrapped in `asyncio.run_in_executor` to avoid blocking FastAPI's event loop. The GPU is used via `get_torch_device()` in `config.py`:

```python
def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

### FFT Transcription — `POST /api/audio/analyze`

`DrumAnalyzer` in `drum_analyzer.py` uses a fully **vectorised NumPy pipeline** (no Python loops over individual hits):

1. **Onset detection** — `librosa.onset.onset_detect` with spectral flux and backtracking identifies candidate hit times.
2. **Analysis windows** — 50 ms windows are extracted around each onset.
3. **Batch FFT** — All windows are transformed simultaneously via `np.fft.rfft`.
4. **Frequency-band classification** — Each hit is classified by its dominant energy band:

   | Class | Frequency band |
   |---|---|
   | `kick` | 0 – 100 Hz |
   | `snare` | 200 – 3 000 Hz |
   | `hat` | 5 000 Hz+ |

5. **Ghost note detection** — Snare hits whose RMS energy is more than 15 dB below the peak quartile average are reclassified as `snare_ghost`.
6. **Velocity normalisation** — RMS energy is min-max normalised to [0, 1].

The result is a `list[{time, type, velocity}]` saved to Supabase via `transcription_store.save_transcription`.

### Supabase Schema

```sql
create table drum_transcriptions (
    id          uuid primary key default gen_random_uuid(),
    job_id      text not null,
    youtube_url text,
    event_count integer not null,
    events      jsonb not null,    -- [{time, type, velocity}, ...]
    exercises   text,              -- Markdown lesson (Phase 3)
    created_at  timestamptz not null default now()
);
```

---

## Phase 3 — Music Teacher

### Grid Builder — `summarize_transcription(events, duration=60)`

`music_teacher.py` maps the first 60 seconds of events onto a human-readable **16th-note grid**:

1. **BPM estimation** — Hi-hat inter-onset intervals (IOIs) are computed, quantised to 5 ms bins, and the modal bin is used to derive the tempo (trying 16th, 8th, and quarter-note interpretations to land in the 60–240 BPM range).
2. **Slot quantisation** — Each event is snapped to `round(time / sixteenth_duration)`.
3. **Grid rendering** — Four instrument rows per measure (`K` / `S` / `g` / `H`), beat boundaries marked with `|`.

```
BPM ~130.4  |  K=Kick  S=Snare  g=Ghost  H=Hi-hat

M01:  K |X...|....|X...|....|
       S |....|X...|....|X...|
       g |..X.|..X.|..X.|..X.|
       H |X.X.|X.X.|X.X.|X.X.|
```

### LLM Lesson — `generate_lesson(job_id, grid)`

The grid is sent to a **local Ollama instance** (`qwen2.5:14b`) via the `ollama` Python async client. The system prompt instructs the model to act as a world-class drum instructor and return exactly four Markdown sections:

1. **Overall Transcription** — time signature, tempo, linear vs. pocket feel
2. **Key Rudiments** — sticking patterns (Paradiddle, Double Stroke, etc.) + 2 exercises
3. **Ghost Note Mastery** — snare dynamics analysis + 2 chatter-note exercises
4. **Groove & Coordination** — kick/snare relationship + 1 limb-independence exercise

The lesson is persisted back to Supabase in the `exercises` column via `save_exercises(job_id, lesson)`.

**Trigger script:**

```bash
cd drum-transcriber
venv/Scripts/python trigger_lesson.py
```

**API endpoint:** `POST /api/audio/teach  { "job_id": "..." }`

---

## Phase 4 — Frontend (Cat.li)

### `useTranscription` hook

Queries Supabase's `drum_transcriptions` table using the **anon (publishable) key**. Accepts a full YouTube URL — a regex extracts the 11-character video ID and uses an `ilike` query for flexible matching. Returns `{ data, loading, error }` with automatic cancellation on re-render.

### `DrumNotation` component

Uses **VexFlow 4** (dynamically imported to avoid SSR issues) to render a 5-line percussion staff:

- **Two voices** — Voice 1 stems-down (kick on `f/4`), Voice 2 stems-up (snare `c/5`, hi-hat `g/5`).
- **Grid quantisation** mirrors the backend Python logic (BPM estimation → 16th-note slots).
- **Playhead** — an absolutely-positioned `div` animated via `requestAnimationFrame`. Position is calculated from `stave.getNoteStartX()` and the tempo-derived measure duration.
- Hi-hat notes are coloured `#7C6FCD`; ghost notes `#C4B5FD`.

### `MusicTeacherLesson` component

Renders the `exercises` Markdown string using **react-markdown** with fully custom component renderers:

- `h2` → coloured pastel banners (blue / green / pink / orange per section keyword)
- `pre` → dark monospace code blocks for ASCII notation exercises
- `code` → inline purple pill for sticking patterns

The card uses a `repeating-linear-gradient` CSS trick to simulate lined notepad paper.

### Theming

Tailwind v4's `@theme inline` block defines `--color-catli-*` CSS variables that Tailwind generates utility classes for (`bg-catli-bg`, `text-catli-purple-dark`, etc.). No `tailwind.config.js` needed.

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- NVIDIA GPU with CUDA 12.x (CPU fallback works, significantly slower)
- [Ollama](https://ollama.ai) installed and running
- `ffmpeg` and `yt-dlp` on `PATH`

### 1. Clone & install

```bash
git clone <repo-url>
cd drum-transcriber

# Python venv
python -m venv venv
venv/Scripts/activate          # Windows
pip install -r backend/requirements.txt

# Node
cd frontend && npm install && cd ..
```

### 2. Pull the LLM

```bash
ollama pull qwen2.5:14b
```

### 3. Environment variables

Create `drum-transcriber/.env`:

```env
SUPABASE_URL=https://<ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY=sb_publishable_...
```

Create `drum-transcriber/frontend/.env.local`:

```env
NEXT_PUBLIC_SUPABASE_URL=https://<ref>.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY=sb_publishable_...
```

### 4. Supabase migration

Run once in the Supabase SQL editor:

```sql
create table drum_transcriptions (
    id          uuid primary key default gen_random_uuid(),
    job_id      text not null,
    youtube_url text,
    event_count integer not null,
    events      jsonb not null,
    exercises   text,
    created_at  timestamptz not null default now()
);
```

### 5. Run

```bash
# Backend (Terminal 1)
cd backend && uvicorn main:app --reload

# Frontend (Terminal 2)
cd frontend && npm run dev

# Generate a lesson for a track (Terminal 3)
python trigger_lesson.py
```

Open **http://localhost:3000**, paste a YouTube URL, and click **View Analysis**.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/audio/extract` | Download YouTube audio → WAV |
| `POST` | `/api/audio/separate` | Demucs drum stem isolation |
| `POST` | `/api/audio/process` | Extract + separate in one call |
| `POST` | `/api/audio/analyze` | FFT transcription → events JSON |
| `POST` | `/api/audio/teach` | Generate + save LLM lesson |

---

## Verify CUDA

```bash
venv/Scripts/python verify_cuda.py
# CUDA available : True
# Device name    : NVIDIA GeForce RTX 5070
```

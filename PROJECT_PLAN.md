# Project Plan: Cat.li Drum Transcriber

**Goal:** A web application that transcribes drum performances from YouTube videos with high accuracy — including ghost notes, dynamics, and correct rhythmic placement — and generates personalized practice lessons using a local LLM.

---

## Current Status

All four phases are complete and working end-to-end.

---

## Phase 1 — Pipeline Infrastructure ✅

**Goal:** Get audio from YouTube into a format ready for analysis.

1. YouTube audio extraction via `yt-dlp` + `ffmpeg` → 16-bit mono WAV @ 44 100 Hz
2. Drum stem isolation via Meta Demucs `htdemucs` → `<job_id>_drums.wav`
3. 4-step pipeline endpoint (`POST /api/pipeline/start`) with background task + progress polling
4. Supabase `drum_transcriptions` table for persisting results

**Key decision:** Use Demucs `htdemucs` (not MDX23C-DrumSep) for stem separation. Demucs is simpler to deploy, produces clean drum isolation, and doesn't require separate model weight downloads.

---

## Phase 2 — ADTOF Transcription Engine ✅

**Goal:** Accurately detect and classify every drum hit with correct timing.

**Approach tried and abandoned:** librosa onset detection + NumPy FFT frequency-band classification. This approach could not distinguish overlapping instruments reliably (kick bleed into snare, cymbal/ride confusion) and had no notion of rhythmic structure.

**Current approach:** ADTOF-pytorch `Frame_RNN` neural network.

ADTOF is a CRNN (Convolutional Recurrent Neural Network) trained on 359 hours of real acoustic drum recordings (MDB-Drums++). It processes the full drum stem directly and outputs onset times for 5 instrument classes simultaneously:

| Class | GM MIDI |
|---|---|
| Bass drum | 36 (Kick) |
| Snare | 38 |
| Tom | 45 |
| Hi-hat | 42 |
| Cymbal | 49 |

Post-processing pipeline:
1. **Velocity** — RMS energy at each onset window, normalized per instrument
2. **16th-note quantization** — anchored to `grid_phase = beat_times[0] % grid_unit` (not raw beat_times[0], which is the first detected beat, not the grid origin)
3. **Ghost note tagging** — snare hits below the 20th percentile velocity
4. **MIDI export** — standard MIDI file via `mido` for download/playback

**Key lesson:** Accuracy came from switching to a neural model (ADTOF) that understands drum context, not from iterating on frequency-band heuristics.

---

## Phase 3 — Music Teacher ✅

**Goal:** Generate actionable practice content from the transcription.

`build_semantic_log` converts the event list into a 16th-note ASCII grid (first 60 seconds). This grid is sent to a local **Ollama `qwen2.5:14b`** instance via the Python `ollama` async client. The LLM returns four Markdown sections:

1. Overall transcription (time signature, tempo, feel)
2. Key rudiments (sticking patterns + 2 exercises)
3. Ghost note mastery (dynamics analysis + 2 exercises)
4. Groove and coordination (kick/snare relationship + 1 exercise)

The lesson is stored in the Supabase `exercises` column.

---

## Phase 4 — Visualization ✅

**Goal:** Display the transcription results clearly and accurately.

Two views are available, toggled in the UI:

**View 1 — Sheet Music (default):** MusicXML rendered by OpenSheetMusicDisplay (OSMD).

`musicxml_builder.py` converts stored events + metadata into a valid MusicXML 3.1 document using Python's stdlib `xml.etree` — no extra dependencies. Events are grouped into measures using `grid_phase`, then serialised as a two-voice percussion score (Voice 1 stems-up for SD/HH/TOM/CYM, Voice 2 stems-down for BD). Ghost notes use parenthesised noteheads. The MusicXML is served from `GET /api/notation/musicxml/{job_id}` and rendered in the browser by OSMD (dynamically imported to keep it out of the initial bundle). The YouTube video title is shown as the score title via YouTube's oEmbed API.

**Why this approach works where VexFlow didn't:** VexFlow is a low-level rendering primitive — the application must solve all the notation math (measure buckets, slot assignment, voice balancing, rest filling) before VexFlow can draw anything. Any bug in that math cascades visually. MusicXML is a structured format that encodes all of this explicitly, and OSMD is a purpose-built renderer that handles edge cases correctly.

**View 2 — Event Plot:** SVG scatter plot, directly inspired by the ADTOF repository's own visualization. Each event's `time` maps linearly to x at 80 px/second. Five horizontal lanes (BD/SD/TT/HH/CY+RD). No quantization math — the backend times are trusted directly. Useful for verifying raw transcription accuracy.

---

## Known Limitations

- **Open vs. closed hi-hat not distinguished** — ADTOF outputs a single hi-hat class (GM 42). Open hi-hat (GM 46) is not detected separately.
- **Single tom class** — All toms map to GM 45 (Low Tom). No rack/floor tom distinction.
- **Cymbal not differentiated** — Crash and ride both map to GM 49.
- **BPM edge cases** — Librosa's beat tracker can lock onto 4/3× the true tempo in certain grooves. A heuristic correction is in place but not perfect.

---

## Potential Next Steps

### Hi-hat open/closed detection
ADTOF's 5-class model merges open and closed hi-hat. A post-processing step could try to infer open hits from the RMS envelope shape (open hi-hats decay slower).

### Playback sync
Add an audio player for the original YouTube track with a moving playhead on the scatter plot synchronized to audio time.

### User accounts
Store transcription history per user via Supabase Auth.

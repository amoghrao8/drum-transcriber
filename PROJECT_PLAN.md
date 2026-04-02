# Project: DrumScribe AI

**Goal:** A web application that transcribes drum performances from YouTube videos with high accuracy (including ghost notes and dynamics) and provides personalized practice routines.

## Technical Stack

- **Frontend:** Next.js (TypeScript), Tailwind CSS, VexFlow (for notation)
- **Backend:** FastAPI (Python 3.11)
- **Processing:** FFmpeg, yt-dlp (downloading), Meta Demucs (stem separation)
- **Analysis:** Librosa, Madmom, or custom PyTorch models for onset detection
- **Database/Auth:** Supabase (for storing transcription history and user exercises)

## Phase 1: The Pipeline ✅

1. Link Frontend to Backend via a secure API.
2. Implement YouTube audio extraction using `yt-dlp`.
3. Integrate `Demucs` to isolate the `drums.wav` stem from the audio.

## Phase 2: The Analysis Engine ✅

1. Develop an onset detection algorithm that identifies Kick, Snare, and Hi-hat.
2. Implement dynamics detection: Categorize hits by velocity (Ghost notes vs. Accents).
3. Convert detected hits into a MIDI-like JSON structure.

## Phase 3: The Music Teacher

1. Create a "Pattern Analyzer" that identifies the core rudiments used in the song.
2. Integration with an LLM (Claude API) to generate 3-5 specific exercises (hand-foot coordination, fills) based on the song's difficulty.

## Phase 4: Visualization

1. Render the transcribed JSON into sheet music using VexFlow.
2. Create a "Practice Mode" dashboard for the user.

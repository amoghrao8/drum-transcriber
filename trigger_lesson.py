"""
One-shot script: generate and display a Music Teacher lesson for the
test track with YouTube ID 42kM4ua-Gt8.

Run from the drum-transcriber directory:

    cd drum-transcriber
    python trigger_lesson.py

Prerequisites
─────────────
1. Supabase: run the migration below once in the Supabase SQL editor:
       ALTER TABLE drum_transcriptions ADD COLUMN exercises text;

2. The track must already be in drum_transcriptions (Phase 1 + 2 complete).

3. Ollama must be running locally with llama4:14b pulled:
       ollama serve          (in a separate terminal if not already running)
       ollama pull llama4:14b

4. The backend venv must be active (packages installed).
"""

import asyncio
import os
import sys
from pathlib import Path

# ── resolve backend on sys.path so we can import app.* directly ─────────────
BACKEND = Path(__file__).parent / "backend"
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
# Try monorepo root first, then drum-transcriber root
_env = Path(__file__).parent.parent / ".env"
if not _env.exists():
    _env = Path(__file__).parent / ".env"
load_dotenv(_env)

from app.core.supabase_client import supabase                     # noqa: E402
from app.services.music_teacher import summarize_transcription, generate_lesson  # noqa: E402
from app.services.transcription_store import save_exercises       # noqa: E402

VIDEO_ID = "42kM4ua-Gt8"


async def main() -> None:
    print(f"[1/4] Looking up transcription for video ID: {VIDEO_ID}")

    # Supabase: find the most-recent row whose youtube_url contains the video ID
    result = (
        supabase.table("drum_transcriptions")
        .select("job_id, youtube_url, event_count, events")
        .like("youtube_url", f"%{VIDEO_ID}%")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if not result.data:
        sys.exit(
            f"\n✗  No transcription found for video ID '{VIDEO_ID}'.\n"
            "   Make sure you have run the full pipeline (extract → separate → analyze)\n"
            "   for https://www.youtube.com/watch?v=42kM4ua-Gt8 first.\n"
        )

    row = result.data[0]
    job_id: str      = row["job_id"]
    events: list     = row["events"]
    event_count: int = row["event_count"]
    yt_url: str      = row.get("youtube_url", "n/a")

    print(f"   Found job_id : {job_id}")
    print(f"   YouTube URL  : {yt_url}")
    print(f"   Events       : {event_count}")

    # ── Build 16th-note grid ─────────────────────────────────────────────────
    print("\n[2/4] Building 16th-note grid (first 60 s)…")
    grid = summarize_transcription(events, duration=60.0)
    print("\n-- Grid preview (first 20 lines) ----------------------------------")
    for line in grid.splitlines()[:20]:
        print(line)
    print("   …")

    # ── Call Ollama ──────────────────────────────────────────────────────────
    print(f"\n[3/4] Sending grid to Ollama ({VIDEO_ID=}, model=llama4:14b)…")
    print("   (This may take 20–60 s depending on hardware)")
    try:
        lesson = await generate_lesson(job_id, grid)
    except RuntimeError as exc:
        sys.exit(f"\n✗  Ollama call failed: {exc}\n"
                 "   Is Ollama running?  Try: ollama serve\n"
                 "   Is the model pulled? Try: ollama pull llama4:14b\n")

    # ── Save to Supabase ─────────────────────────────────────────────────────
    print("\n[4/4] Saving lesson to Supabase (exercises column)…")
    try:
        await save_exercises(job_id, lesson)
        print("   Saved.")
    except RuntimeError as exc:
        print(f"   ⚠  Could not save to Supabase: {exc}")
        print("   (lesson still printed below)")

    # ── Print lesson ─────────────────────────────────────────────────────────
    separator = "=" * 70
    print(f"\n{separator}")
    print("  MUSIC TEACHER LESSON")
    print(separator)
    print(lesson)
    print(separator)


if __name__ == "__main__":
    asyncio.run(main())

"""
Music Teacher route.

POST /api/audio/teach
    - Fetches the stored transcription for a job_id from Supabase
    - Builds a 16th-note grid from the events
    - Sends the grid to local Ollama (llama4:14b) for lesson generation
    - Saves the lesson back to drum_transcriptions.exercises
    - Returns the Markdown lesson
"""

import asyncio
from functools import partial

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.supabase_client import supabase
from app.services.music_teacher import summarize_transcription, generate_lesson
from app.services.transcription_store import save_exercises

router = APIRouter(prefix="/audio", tags=["music-teacher"])


class TeachRequest(BaseModel):
    job_id: str


class TeachResponse(BaseModel):
    job_id: str
    grid_preview: str   # first 500 chars of grid for quick inspection
    lesson: str


@router.post("/teach", response_model=TeachResponse)
async def teach(body: TeachRequest):
    """
    Generate a personalised drum lesson for a previously analysed track.

    Requires a row in drum_transcriptions with the given job_id.
    The generated lesson is persisted to the exercises column.
    """
    # 1. Fetch transcription row from Supabase
    result = (
        supabase.table("drum_transcriptions")
        .select("job_id, events")
        .eq("job_id", body.job_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        raise HTTPException(
            status_code=404,
            detail=f"No transcription found for job_id '{body.job_id}'. "
                   "Run /api/audio/analyze first.",
        )

    row = result.data[0]
    events: list[dict] = row["events"]

    # 2. Build 16th-note grid (CPU-bound but fast; run in thread for safety)
    loop = asyncio.get_event_loop()
    grid: str = await loop.run_in_executor(
        None, partial(summarize_transcription, events, 60.0)
    )

    # 3. Generate lesson via Ollama (network I/O, can be slow)
    try:
        lesson = await generate_lesson(body.job_id, grid)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # 4. Persist to Supabase
    try:
        await save_exercises(body.job_id, lesson)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return TeachResponse(
        job_id=body.job_id,
        grid_preview=grid[:500],
        lesson=lesson,
    )

"""
Full pipeline endpoint — runs all 4 steps in a BackgroundTask and exposes
a status endpoint that the frontend polls every 5 seconds.

POST /api/pipeline/start   → { pipeline_id }
GET  /api/pipeline/{id}    → PipelineStatus
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from functools import partial

from fastapi import APIRouter, BackgroundTasks, HTTPException

log = logging.getLogger(__name__)
from pydantic import BaseModel

from app.core.job_store import create_job, get_job, update_job
from app.services.audio_extractor import extract_audio
from app.services.stem_separator import separate_stems
from app.services.drum_analyzer import transcribe as transcribe_drums
from app.services.transcription_store import save_transcription, save_exercises
from app.services.music_teacher import build_semantic_log, generate_lesson
from app.core.config import STEMS_OUTPUT_DIR

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


# ---------------------------------------------------------------------------
# Pipeline runner (runs in BackgroundTasks thread pool)
# ---------------------------------------------------------------------------

async def _run(pipeline_id: str, youtube_url: str, *,
               override_bpm: float | None = None,
               override_beats_per_bar: int | None = None,
               override_beat_unit: int | None = None) -> None:
    """Execute the 4-step pipeline, writing progress to the job store."""

    def progress(**kw):
        update_job(pipeline_id, **kw)

    try:
        # ── Step 1: Extract audio ──────────────────────────────────────────
        progress(
            status="running", step=1,
            step_name="Downloading audio",
            pct=5,
            message="yt-dlp is downloading and converting the YouTube stream to 44.1 kHz WAV.",
        )
        audio = await extract_audio(youtube_url)
        job_id: str = audio["job_id"]
        progress(pct=20, db_job_id=job_id)

        # ── Step 2: Separate stems ─────────────────────────────────────────
        progress(
            step=2,
            step_name="Isolating drum track",
            pct=22,
            message="Meta Demucs (htdemucs) is separating the drums from the mix. "
                    "This is the slowest step — the model weights may download on first run.",
        )
        from pathlib import Path
        await separate_stems(Path(audio["file_path"]), job_id)
        progress(pct=60)

        # ── Step 3: Transcribe hits ────────────────────────────────────────
        progress(
            step=3,
            step_name="Transcribing hits",
            pct=62,
            message="ADTOF Frame_RNN neural transcription — detecting kick, snare, "
                    "hi-hat, toms, and cymbals. Quantising to 16th-note grid.",
        )
        drums_wav = STEMS_OUTPUT_DIR / f"{job_id}_drums.wav"
        loop = asyncio.get_event_loop()
        events, midi_obj, metadata = await loop.run_in_executor(
            None, partial(transcribe_drums, drums_wav,
                          override_bpm=override_bpm,
                          override_beats_per_bar=override_beats_per_bar,
                          override_beat_unit=override_beat_unit)
        )
        await save_transcription(job_id, events, youtube_url, metadata)
        progress(pct=75)

        # ── Step 4: Generate lesson ────────────────────────────────────────
        progress(
            step=4,
            step_name="Generating AI lesson",
            pct=77,
            message="Building semantic MIDI log and sending it to Qwen 2.5 "
                    "to generate your personalised drum lesson.",
        )
        semantic_log = build_semantic_log(events, metadata)
        lesson = await generate_lesson(job_id, semantic_log)
        await save_exercises(job_id, lesson)

        progress(
            status="complete",
            step=4,
            step_name="Complete",
            pct=100,
            message="Analysis complete! Your lesson is ready.",
        )

    except BaseException as exc:
        tb = traceback.format_exc()
        log.error("Pipeline %s failed:\n%s", pipeline_id, tb)
        # Write to file so errors are visible regardless of log config
        try:
            with open("pipeline_error.log", "a") as f:
                f.write(f"\n=== Pipeline {pipeline_id} ===\n{tb}\n")
        except Exception:
            pass
        error_msg = str(exc).strip() or repr(exc) or type(exc).__name__
        update_job(
            pipeline_id,
            status="error",
            message=f"Failed at step {get_job(pipeline_id).step if get_job(pipeline_id) else '?'}: {type(exc).__name__}",
            error=error_msg,
        )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    youtube_url: str
    override_bpm: float | None = None
    override_beats_per_bar: int | None = None
    override_beat_unit: int | None = None


class StartResponse(BaseModel):
    pipeline_id: str


class PipelineStatus(BaseModel):
    pipeline_id: str
    status: str
    step: int
    step_name: str
    pct: int
    message: str
    error: str | None
    db_job_id: str | None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/start", response_model=StartResponse)
async def start_pipeline(body: StartRequest, background_tasks: BackgroundTasks):
    """
    Kick off the full pipeline asynchronously.
    Returns a pipeline_id the client can use to poll /pipeline/{id}.
    """
    pipeline_id = uuid.uuid4().hex
    create_job(pipeline_id, body.youtube_url)
    background_tasks.add_task(_run, pipeline_id, body.youtube_url,
                              override_bpm=body.override_bpm,
                              override_beats_per_bar=body.override_beats_per_bar,
                              override_beat_unit=body.override_beat_unit)
    return StartResponse(pipeline_id=pipeline_id)


@router.get("/{pipeline_id}", response_model=PipelineStatus)
def get_status(pipeline_id: str):
    """Poll this endpoint to track pipeline progress."""
    job = get_job(pipeline_id)
    if not job:
        raise HTTPException(status_code=404, detail="Pipeline job not found.")
    return PipelineStatus(
        pipeline_id=job.pipeline_id,
        status=job.status,
        step=job.step,
        step_name=job.step_name,
        pct=job.pct,
        message=job.message,
        error=job.error,
        db_job_id=job.db_job_id,
    )

import asyncio
from functools import partial
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import STEMS_OUTPUT_DIR
from app.services.drum_analyzer import DrumAnalyzer

router = APIRouter(prefix="/audio", tags=["analysis"])

# Single shared analyzer instance — frequency masks computed once at startup.
_analyzer = DrumAnalyzer()


class AnalyzeRequest(BaseModel):
    job_id: str


class DrumEvent(BaseModel):
    time: float
    type: str
    velocity: float


class AnalyzeResponse(BaseModel):
    job_id: str
    event_count: int
    events: list[DrumEvent]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_drums(body: AnalyzeRequest):
    """
    Run FFT-based onset detection and instrument classification on the drums
    stem produced by /separate or /process.

    Returns a timestamped list of drum hit events with type and velocity.
    """
    drums_wav = STEMS_OUTPUT_DIR / f"{body.job_id}_drums.wav"
    if not drums_wav.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Drums stem not found for job_id '{body.job_id}'. "
                   "Run /api/audio/separate first.",
        )

    loop = asyncio.get_event_loop()
    try:
        # Run in thread pool — librosa.load + NumPy are CPU-bound
        events = await loop.run_in_executor(
            None, partial(_analyzer.transcribe, drums_wav)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return AnalyzeResponse(
        job_id=body.job_id,
        event_count=len(events),
        events=events,
    )

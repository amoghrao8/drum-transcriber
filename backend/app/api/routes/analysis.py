import asyncio
from functools import partial

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import STEMS_OUTPUT_DIR
from app.services.drum_analyzer import transcribe

router = APIRouter(prefix="/audio", tags=["analysis"])


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
    Run MDX23C-DrumSep onset detection on the drums stem.
    """
    drums_wav = STEMS_OUTPUT_DIR / f"{body.job_id}_drums.wav"
    if not drums_wav.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Drums stem not found for job_id '{body.job_id}'.",
        )

    loop = asyncio.get_event_loop()
    try:
        events, _, _ = await loop.run_in_executor(
            None, partial(transcribe, drums_wav)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return AnalyzeResponse(
        job_id=body.job_id,
        event_count=len(events),
        events=[{"time": e["time"], "type": e["type"], "velocity": e["velocity"]} for e in events],
    )

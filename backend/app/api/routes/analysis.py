import asyncio
from functools import partial

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import STEMS_OUTPUT_DIR
from app.services.drum_analyzer import transcribe

router = APIRouter(prefix="/audio", tags=["analysis"])


class AnalyzeRequest(BaseModel):
    job_id: str
    override_bpm: float | None = None
    override_beats_per_bar: int | None = None
    override_beat_unit: int | None = None
    quantize: bool = False


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
    Run ADTOF transcription on the drums stem.
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
            None, partial(transcribe, drums_wav,
                          override_bpm=body.override_bpm,
                          override_beats_per_bar=body.override_beats_per_bar,
                          override_beat_unit=body.override_beat_unit,
                          quantize=body.quantize)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    return AnalyzeResponse(
        job_id=body.job_id,
        event_count=len(events),
        events=[{"time": e["time"], "type": e["type"], "velocity": e["velocity"]} for e in events],
    )

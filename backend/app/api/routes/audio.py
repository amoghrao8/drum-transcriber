from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl

from app.core.config import AUDIO_OUTPUT_DIR
from app.services.audio_extractor import extract_audio
from app.services.stem_separator import separate_stems

router = APIRouter(prefix="/audio", tags=["audio"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ExtractionRequest(BaseModel):
    youtube_url: HttpUrl


class ExtractionResponse(BaseModel):
    job_id: str
    file_name: str
    size_bytes: int
    message: str


class SeparationRequest(BaseModel):
    job_id: str


class SeparationResponse(BaseModel):
    job_id: str
    drums_file: str
    sample_rate: int
    size_bytes: int
    device_used: str
    message: str


class ProcessResponse(BaseModel):
    job_id: str
    # extraction
    raw_file: str
    raw_size_bytes: int
    # separation
    drums_file: str
    sample_rate: int
    drums_size_bytes: int
    device_used: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/extract", response_model=ExtractionResponse)
async def extract_youtube_audio(body: ExtractionRequest):
    """
    Download a YouTube video's audio track and convert to a 16-bit mono WAV
    at 44100 Hz. Returns a job_id to pass into /separate.
    """
    try:
        result = await extract_audio(str(body.youtube_url))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ExtractionResponse(
        job_id=result["job_id"],
        file_name=result["file_name"],
        size_bytes=result["size_bytes"],
        message="Audio extracted. Pass job_id to /api/audio/separate.",
    )


@router.post("/separate", response_model=SeparationResponse)
async def separate_audio_stems(body: SeparationRequest):
    """
    Run Demucs (htdemucs) on a previously extracted WAV identified by job_id
    and isolate the drums stem. Uses MPS (Apple GPU) when available.
    """
    input_wav = AUDIO_OUTPUT_DIR / f"{body.job_id}.wav"

    try:
        result = await separate_stems(input_wav, body.job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return SeparationResponse(
        job_id=result["job_id"],
        drums_file=result["drums_file"],
        sample_rate=result["sample_rate"],
        size_bytes=result["size_bytes"],
        device_used=result["device_used"],
        message="Drums stem isolated successfully. Ready for onset detection.",
    )


@router.post("/process", response_model=ProcessResponse)
async def process_youtube_url(body: ExtractionRequest):
    """
    Combined endpoint: extract audio from YouTube then immediately run Demucs
    stem separation. Returns metadata for both the raw audio and drums stem.
    """
    # Step 1 — extract
    try:
        extraction = await extract_audio(str(body.youtube_url))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Step 2 — separate
    input_wav = Path(extraction["file_path"])
    try:
        separation = await separate_stems(input_wav, extraction["job_id"])
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ProcessResponse(
        job_id=extraction["job_id"],
        raw_file=extraction["file_name"],
        raw_size_bytes=extraction["size_bytes"],
        drums_file=separation["drums_file"],
        sample_rate=separation["sample_rate"],
        drums_size_bytes=separation["size_bytes"],
        device_used=separation["device_used"],
        message="Pipeline complete. Drums stem ready for onset detection.",
    )

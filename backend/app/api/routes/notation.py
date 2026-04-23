"""
Notation endpoints.

GET /api/notation/musicxml/{job_id}
  Returns a MusicXML 3.1 document for the stored transcription.

GET /api/notation/midi/{job_id}
  Returns a GM MIDI file (.mid) for the stored transcription.

GET /api/notation/clonehero/{job_id}
  Returns a Clone Hero chart zip for the stored transcription.

GET /api/notation/clonehero/{job_id}/stream
  SSE endpoint — streams build progress then the final zip download URL.
"""
import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from app.core.supabase_client import supabase
from app.core.config import AUDIO_OUTPUT_DIR, STEMS_OUTPUT_DIR
from app.services.musicxml_builder import build_musicxml
from app.services.drum_analyzer import _build_midi, midi_to_bytes
from app.services.clonehero_builder import build_chart_zip

log = logging.getLogger(__name__)
router = APIRouter(prefix="/notation", tags=["notation"])

# In-memory cache for built zips (short-lived, cleared after download)
_zip_cache: dict[str, bytes] = {}


@router.get("/musicxml/{job_id}")
def get_musicxml(job_id: str):
    """
    Fetch stored events + metadata for job_id and return MusicXML.
    """
    resp = (
        supabase.table("drum_transcriptions")
        .select("events, metadata")
        .eq("job_id", job_id)
        .limit(1)
        .execute()
    )

    if not resp.data:
        raise HTTPException(
            status_code=404,
            detail=f"No transcription found for job_id '{job_id}'.",
        )

    row      = resp.data[0]
    events   = row.get("events")   or []
    metadata = row.get("metadata") or {}

    if not events:
        raise HTTPException(status_code=422, detail="Transcription has no events.")

    xml_str = build_musicxml(events, metadata)

    return Response(
        content=xml_str,
        media_type="application/xml",
        headers={"Content-Disposition": f'inline; filename="{job_id}.musicxml"'},
    )


@router.get("/midi/{job_id}")
def get_midi(job_id: str):
    """
    Fetch stored events + metadata for job_id and return a GM MIDI file.
    """
    resp = (
        supabase.table("drum_transcriptions")
        .select("events, metadata")
        .eq("job_id", job_id)
        .limit(1)
        .execute()
    )

    if not resp.data:
        raise HTTPException(
            status_code=404,
            detail=f"No transcription found for job_id '{job_id}'.",
        )

    row      = resp.data[0]
    events   = row.get("events")   or []
    metadata = row.get("metadata") or {}

    if not events:
        raise HTTPException(status_code=422, detail="Transcription has no events.")

    bpm           = metadata.get("bpm", 120.0)
    beats_per_bar = metadata.get("beats_per_bar", 4)
    beat_unit     = metadata.get("beat_unit", 4)

    midi = _build_midi(events, bpm, beats_per_bar, beat_unit)
    midi_bytes = midi_to_bytes(midi)

    return Response(
        content=midi_bytes,
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.mid"'},
    )


@router.get("/clonehero/{job_id}")
def get_clonehero(job_id: str):
    """
    Fetch stored events + metadata for job_id and return a Clone Hero chart zip.
    """
    resp = (
        supabase.table("drum_transcriptions")
        .select("events, metadata")
        .eq("job_id", job_id)
        .limit(1)
        .execute()
    )

    if not resp.data:
        raise HTTPException(
            status_code=404,
            detail=f"No transcription found for job_id '{job_id}'.",
        )

    row      = resp.data[0]
    events   = row.get("events")   or []
    metadata = row.get("metadata") or {}

    if not events:
        raise HTTPException(status_code=422, detail="Transcription has no events.")

    zip_bytes = build_chart_zip(
        events, metadata, song_name=job_id,
        song_wav=AUDIO_OUTPUT_DIR / f"{job_id}.wav",
        drums_wav=STEMS_OUTPUT_DIR / f"{job_id}_drums.wav",
        drumless_wav=STEMS_OUTPUT_DIR / f"{job_id}_drumless.wav",
    )

    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_clonehero.zip"'},
    )


@router.get("/clonehero/{job_id}/stream")
async def stream_clonehero_build(job_id: str):
    """
    SSE endpoint that streams build progress, then emits a download_token
    the client can use to fetch the finished zip from /clonehero/download/{token}.
    """
    resp = (
        supabase.table("drum_transcriptions")
        .select("events, metadata")
        .eq("job_id", job_id)
        .limit(1)
        .execute()
    )

    if not resp.data:
        raise HTTPException(status_code=404, detail="No transcription found.")

    row = resp.data[0]
    events = row.get("events") or []
    metadata = row.get("metadata") or {}

    if not events:
        raise HTTPException(status_code=422, detail="Transcription has no events.")

    async def _generate():
        progress_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()

        def on_progress(pct: int, msg: str):
            progress_queue.put_nowait((pct, msg))

        loop = asyncio.get_event_loop()

        # Run the (blocking) zip build in a thread
        build_task = loop.run_in_executor(
            None,
            lambda: build_chart_zip(
                events, metadata, song_name=job_id,
                song_wav=AUDIO_OUTPUT_DIR / f"{job_id}.wav",
                drums_wav=STEMS_OUTPUT_DIR / f"{job_id}_drums.wav",
                drumless_wav=STEMS_OUTPUT_DIR / f"{job_id}_drumless.wav",
                on_progress=on_progress,
            ),
        )

        # Drain progress events while the build is running
        while not build_task.done():
            try:
                pct, msg = await asyncio.wait_for(progress_queue.get(), timeout=0.3)
                yield f"data: {json.dumps({'pct': pct, 'message': msg})}\n\n"
            except asyncio.TimeoutError:
                pass

        zip_bytes = await build_task

        # Drain any remaining queued progress
        while not progress_queue.empty():
            pct, msg = progress_queue.get_nowait()
            yield f"data: {json.dumps({'pct': pct, 'message': msg})}\n\n"

        # Store zip in cache and emit download token
        token = uuid.uuid4().hex
        _zip_cache[token] = zip_bytes
        yield f"data: {json.dumps({'pct': 100, 'message': 'Ready', 'download_token': token})}\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.get("/clonehero/download/{token}")
def download_clonehero_zip(token: str):
    """Fetch a previously built Clone Hero zip by its short-lived token."""
    zip_bytes = _zip_cache.pop(token, None)
    if not zip_bytes:
        raise HTTPException(status_code=404, detail="Download expired or not found.")
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="clonehero.zip"'},
    )

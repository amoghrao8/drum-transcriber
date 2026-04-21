"""
Notation endpoints.

GET /api/notation/musicxml/{job_id}
  Returns a MusicXML 3.1 document for the stored transcription.

GET /api/notation/midi/{job_id}
  Returns a GM MIDI file (.mid) for the stored transcription.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.core.supabase_client import supabase
from app.services.musicxml_builder import build_musicxml
from app.services.drum_analyzer import _build_midi, midi_to_bytes

router = APIRouter(prefix="/notation", tags=["notation"])


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

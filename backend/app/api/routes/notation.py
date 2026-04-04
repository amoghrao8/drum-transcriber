"""
Notation endpoints.

GET /api/notation/musicxml/{job_id}
  Returns a MusicXML 3.1 document for the stored transcription.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app.core.supabase_client import supabase
from app.services.musicxml_builder import build_musicxml

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

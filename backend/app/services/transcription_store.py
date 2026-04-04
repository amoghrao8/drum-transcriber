"""
Persists drum transcription results to the `drum_transcriptions` Supabase table.

Schema (run in Supabase SQL editor to migrate):

    create table if not exists drum_transcriptions (
        id               uuid primary key default gen_random_uuid(),
        job_id           text not null,
        youtube_url      text,
        event_count      integer not null,
        events           jsonb  not null,
        metadata         jsonb,
        exercises        text,
        created_at       timestamptz not null default now()
    );

    -- Add metadata column to existing table:
    ALTER TABLE drum_transcriptions ADD COLUMN IF NOT EXISTS metadata JSONB;

Events column format (MIDI-first, each element):
    { "note": 36, "time": 0.25, "velocity": 90,
      "type": "kick", "ghost": false, "duration": 0.05 }

Metadata column format:
    { "bpm": 120.0, "time_signature": "4/4",
      "beats_per_bar": 4, "beat_unit": 4,
      "duration": 185.3, "event_count": 1247 }
"""
from typing import Any, Optional

from app.core.supabase_client import supabase


async def save_transcription(
    job_id: str,
    events: list[dict],
    youtube_url: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Upsert a transcription record keyed on job_id.
    If a row with this job_id already exists it is updated in place,
    preventing duplicate rows for the same pipeline run.
    """
    payload: dict[str, Any] = {
        "job_id":      job_id,
        "youtube_url": youtube_url,
        "event_count": len(events),
        "events":      events,
    }
    if metadata:
        payload["metadata"] = metadata

    # Try UPDATE first (row already exists for this job_id)
    update_resp = (
        supabase.table("drum_transcriptions")
        .update(payload)
        .eq("job_id", job_id)
        .execute()
    )
    if update_resp.data:
        return update_resp.data[0]

    # No existing row — INSERT
    insert_resp = (
        supabase.table("drum_transcriptions")
        .insert(payload)
        .execute()
    )
    if not insert_resp.data:
        raise RuntimeError(
            f"Supabase insert returned no data. Response: {insert_resp}"
        )
    return insert_resp.data[0]


async def save_exercises(job_id: str, exercises: str) -> dict:
    """
    Update the exercises column for an existing transcription row.
    Raises RuntimeError if no matching row is found.
    """
    response = (
        supabase.table("drum_transcriptions")
        .update({"exercises": exercises})
        .eq("job_id", job_id)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            f"save_exercises: no row found for job_id '{job_id}'."
        )
    return response.data[0]

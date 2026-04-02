"""
Persists drum transcription results to the `drum_transcriptions` Supabase table.

Uses the admin client (Service Role Key) so writes succeed regardless of any
Row Level Security policies on the table.

Expected table schema (run once in the Supabase SQL editor):

    create table drum_transcriptions (
        id          uuid primary key default gen_random_uuid(),
        job_id      text not null,
        youtube_url text,
        event_count integer not null,
        events      jsonb  not null,
        created_at  timestamptz not null default now()
    );
"""
from typing import Optional

from app.core.supabase_client import supabase


async def save_transcription(
    job_id: str,
    events: list[dict],
    youtube_url: Optional[str] = None,
) -> dict:
    """
    Insert a transcription record and return the created row.

    Raises RuntimeError if the insert fails.
    """
    payload = {
        "job_id":      job_id,
        "youtube_url": youtube_url,
        "event_count": len(events),
        "events":      events,
    }

    response = supabase.table("drum_transcriptions").insert(payload).execute()

    if not response.data:
        raise RuntimeError(
            f"Supabase insert returned no data. Response: {response}"
        )

    return response.data[0]

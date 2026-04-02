"""
Backend-only Supabase admin client.

Initialized with the Service Role Key, which grants full database access
and bypasses Row Level Security. This module must never be imported by or
exposed to the frontend.
"""
from supabase import create_client, Client

from app.core.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY


def _create_admin_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env "
            "before the Supabase client can be initialized."
        )
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# Singleton — created once when this module is first imported.
supabase: Client = _create_admin_client()

"""
Supabase client stub — flight table sync removed.
The flights table is no longer maintained; change tracking is handled in-memory.
"""

import os
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://vkffblxiaazkvimwilwk.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "REDACTED")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def upsert_flights(flights: list[dict], airport_code: str):
    pass


def cleanup_departed_flight(flight_number: str, airport_code: str):
    pass

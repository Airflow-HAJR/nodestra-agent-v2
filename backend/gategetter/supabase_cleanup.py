"""
Supabase flight sync: adds new flights and removes departed ones.
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
    """Insert new flights into Supabase. Skips flights that already exist."""
    sb = _get_client()

    for f in flights:
        fn = f.get("flight_number")
        if not fn:
            continue
        try:
            sb.table("flights").upsert({
                "flight_number": fn,
                "airport_departing_id": airport_code,
                "airport_arriving_id": f.get("destination_iata"),
                "terminal": f.get("terminal"),
                "gate": f.get("gate"),
            }, on_conflict="flight_number").execute()
        except Exception as e:
            print(f"  [{airport_code}] Supabase upsert error for {fn}: {e}")


def cleanup_departed_flight(flight_number: str, airport_code: str):
    """Remove a departed flight and its passenger links from Supabase."""
    sb = _get_client()

    try:
        # 1. Get passenger IDs linked to this flight at this airport
        fp_resp = sb.table("flight_passengers") \
            .select("id, passenger_id") \
            .eq("flight_id", flight_number) \
            .eq("airport_id", airport_code) \
            .execute()

        passenger_ids = [row["passenger_id"] for row in (fp_resp.data or [])]

        # 2. Delete flight_passengers records
        sb.table("flight_passengers") \
            .delete() \
            .eq("flight_id", flight_number) \
            .eq("airport_id", airport_code) \
            .execute()

        # 3. Delete orphaned passengers (not linked to any other flight)
        for pid in passenger_ids:
            other = sb.table("flight_passengers") \
                .select("id") \
                .eq("passenger_id", pid) \
                .limit(1) \
                .execute()
            if not other.data:
                sb.table("passengers").delete().eq("id", pid).execute()

        # 4. Delete the flight itself
        sb.table("flights") \
            .delete() \
            .eq("flight_number", flight_number) \
            .execute()

        print(f"  [{airport_code}] Cleaned up Supabase: {flight_number} ({len(passenger_ids)} passenger links removed)")

    except Exception as e:
        print(f"  [{airport_code}] Supabase cleanup error for {flight_number}: {e}")

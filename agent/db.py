"""
Supabase client and map data loading.
Everything that touches the database lives here.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

from supabase import Client, create_client

import os

from agent.config import SUPABASE_KEY, SUPABASE_URL

SUPABASE_CONFIGURED: bool = bool(SUPABASE_URL and SUPABASE_KEY)
_supabase_reachable: bool = True  # set False on first ConnectError; never reset


def supabase_ok() -> bool:
    return SUPABASE_CONFIGURED and _supabase_reachable


def mark_supabase_unreachable() -> None:
    global _supabase_reachable
    _supabase_reachable = False


_client: Optional[Client] = None
_service_client: Optional[Client] = None
_MAP_CACHE: dict[str, tuple[list[dict], float]] = {}
_MAP_LOCK = threading.Lock()
_MAP_TTL_SECONDS = 3600


def get_client() -> Client:
    """Anon-key client — used for map reads."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


def get_service_client() -> Client:
    """Service-role client — bypasses RLS, used for analytics writes."""
    global _service_client
    if _service_client is None:
        service_key = os.environ.get("SUPABASE_SERVICE_KEY") or SUPABASE_KEY
        if not SUPABASE_URL or not service_key:
            raise RuntimeError("SUPABASE_URL must be set in .env")
        _service_client = create_client(SUPABASE_URL, service_key)
    return _service_client


def _slim_levels(rows: list[dict]) -> list[dict]:
    """Extract and slim down map levels from Supabase rows."""
    levels = []
    for row in rows:
        gm = row.get("graph_map") or {}
        if not gm.get("pois") and not gm.get("waypoints"):
            continue
        levels.append({
            "name": row["name"],
            "pois": [
                {
                    "id": p["id"],
                    "name": p.get("name", ""),
                    "type": p.get("type", ""),
                    "x": p["x"],
                    "y": p["y"],
                    "waypointId": p.get("waypointId", ""),
                    "linkedPortalIds": [lp for lp in p.get("linkedPortalIds", []) if lp],
                }
                for p in gm.get("pois", [])
            ],
            "waypoints": [
                {"id": w["id"], "x": w["x"], "y": w["y"]}
                for w in gm.get("waypoints", [])
            ],
            "edges": [
                {"from": e["from"], "to": e["to"], "weight": e.get("weight", 1)}
                for e in gm.get("edges", [])
            ],
        })
    return levels


def get_map_levels_cached(airport_id: str) -> Optional[list[dict]]:
    with _MAP_LOCK:
        entry = _MAP_CACHE.get(airport_id)
        if entry and time.time() < entry[1]:
            return entry[0]
    return None


def put_map_levels_cache(airport_id: str, levels: list[dict]) -> None:
    with _MAP_LOCK:
        _MAP_CACHE[airport_id] = (levels, time.time() + _MAP_TTL_SECONDS)


def invalidate_map_cache(airport_id: str) -> None:
    with _MAP_LOCK:
        _MAP_CACHE.pop(airport_id, None)


def load_map_levels(airport_id: str) -> list[dict]:
    """Fetch map levels: in-process cache -> Supabase."""
    cached = get_map_levels_cached(airport_id)
    if cached is not None:
        return cached

    db = get_client()
    resp = (
        db.table("maps")
        .select("name, sort_order, graph_map")
        .eq("airport_id", airport_id)
        .order("sort_order")
        .execute()
    )
    levels = _slim_levels(resp.data or [])
    put_map_levels_cache(airport_id, levels)
    return levels

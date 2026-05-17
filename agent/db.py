"""
Supabase client and map data loading.
Everything that touches the database lives here.
"""
from __future__ import annotations

from typing import Optional

from supabase import Client, create_client

from agent.config import SUPABASE_KEY, SUPABASE_URL
from agent.moss import get_map_levels_cached, put_map_levels_cache

_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


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


def load_map_levels(airport_id: str) -> list[dict]:
    """Fetch map levels: L1 dict → moss → Supabase."""
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

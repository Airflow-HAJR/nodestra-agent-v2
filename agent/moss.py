"""
Moss.dev integration — semantic POI search.

Moss is a semantic search engine, NOT a key-value cache.
It lets us find POIs by meaning ("coffee place" → "Starbucks") instead of
exact string matching.

Map graph data (edges, waypoints, coordinates) is cached in a plain
in-process dict — moss is only for the search layer.

Usage flow:
  1. On /update: call index_airport_pois() to build the moss index from loaded levels
  2. On find_poi: call search_pois() to get semantically ranked matches
  3. If moss is unavailable / unconfigured: falls back to Levenshtein in map_engine
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from typing import Optional

from agent.config import MOSS_PROJECT_ID, MOSS_PROJECT_KEY

_client = None
_client_lock = threading.Lock()

# Simple in-process map cache (the real graph data — edges, pois, waypoints)
# This is a plain dict, nothing to do with moss.
_MAP_CACHE: dict[str, tuple[list[dict], float]] = {}
_MAP_LOCK = threading.Lock()
_MAP_TTL = 3600


def _get_client():
    global _client
    if not is_available():
        return None
    with _client_lock:
        if _client is None:
            from moss import MossClient
            _client = MossClient(MOSS_PROJECT_ID, MOSS_PROJECT_KEY)
    return _client


def is_available() -> bool:
    return bool(MOSS_PROJECT_ID and MOSS_PROJECT_KEY)


def _index_name(airport_id: str) -> str:
    return f"pois-{airport_id.lower()}"


def _run_async(coro):
    """Run an async coroutine from sync code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# In-process map graph cache (no moss — just a dict)
# ---------------------------------------------------------------------------

def get_map_levels_cached(airport_id: str) -> Optional[list[dict]]:
    with _MAP_LOCK:
        entry = _MAP_CACHE.get(airport_id)
        if entry and time.time() < entry[1]:
            return entry[0]
    return None


def put_map_levels_cache(airport_id: str, levels: list[dict]) -> None:
    with _MAP_LOCK:
        _MAP_CACHE[airport_id] = (levels, time.time() + _MAP_TTL)


def invalidate_map_cache(airport_id: str) -> None:
    with _MAP_LOCK:
        _MAP_CACHE.pop(airport_id, None)


# ---------------------------------------------------------------------------
# Moss POI index (semantic search)
# ---------------------------------------------------------------------------

def index_airport_pois(airport_id: str, levels: list[dict]) -> int:
    """
    Build / rebuild the moss semantic index for an airport's POIs.
    Call this once from the /update endpoint after loading from Supabase.
    Returns number of POIs indexed, or 0 if moss is not configured.
    """
    client = _get_client()
    if client is None:
        return 0

    from moss import DocumentInfo

    docs = []
    for level in levels:
        level_name = level.get("name", "")
        for poi in level.get("pois", []):
            name = poi.get("name", "").strip()
            if not name:
                continue
            poi_type = poi.get("type", "")
            text = f"{name} {poi_type} {level_name}".strip()
            docs.append(DocumentInfo(
                id=poi["id"],
                text=text,
                metadata={
                    "name": name,
                    "type": poi_type,
                    "level": level_name,
                    "airport_id": airport_id,
                },
            ))

    if not docs:
        return 0

    idx = _index_name(airport_id)

    async def _build():
        await client.create_index(idx, docs, "moss-minilm")
        await client.load_index(idx)

    try:
        _run_async(_build())
        return len(docs)
    except Exception as e:
        print(f"[MOSS] index build failed: {e}")
        return 0


def ensure_index_loaded(airport_id: str) -> bool:
    """Load the moss index into memory if it's not already loaded."""
    client = _get_client()
    if client is None:
        return False
    try:
        _run_async(client.load_index(_index_name(airport_id)))
        return True
    except Exception:
        return False


def search_pois(airport_id: str, query: str, top_k: int = 10) -> list[dict]:
    """
    Semantic POI search. Returns list of dicts with id, name, type, level, score.
    Returns [] if moss is not configured or index not found — caller falls back to Levenshtein.
    """
    client = _get_client()
    if client is None:
        return []

    from moss import QueryOptions

    async def _query():
        return await client.query(
            _index_name(airport_id),
            query,
            QueryOptions(top_k=top_k, alpha=0.6),
        )

    try:
        results = _run_async(_query())
        out = []
        for doc in results.docs:
            out.append({
                "id": doc.id,
                "name": doc.metadata.get("name", ""),
                "type": doc.metadata.get("type", ""),
                "_level_name": doc.metadata.get("level", ""),
                "_score": doc.score,
            })
        return out
    except Exception as e:
        print(f"[MOSS] search failed: {e}")
        return []

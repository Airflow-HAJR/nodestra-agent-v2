from __future__ import annotations

import asyncio
import concurrent.futures

from agent.config import MOSS_PROJECT_ID, MOSS_PROJECT_KEY

_moss_client = None
_moss_indexes_ready: set[str] = set()
_moss_indexes_failed: set[str] = set()
_INDEX_POIS = "oakland-pois"
_INDEX_FLIGHTS = "oakland-flights"
_THRESHOLD = 0.8


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


async def _get_moss():
    global _moss_client
    if _moss_client is None:
        if not MOSS_PROJECT_ID or not MOSS_PROJECT_KEY:
            return None
        from moss import MossClient

        _moss_client = MossClient(MOSS_PROJECT_ID, MOSS_PROJECT_KEY)
    return _moss_client


async def _ensure_loaded(client, index: str) -> bool:
    if index in _moss_indexes_failed:
        return False
    try:
        await client.load_index(index)
        _moss_indexes_ready.add(index)
        return True
    except Exception:
        _moss_indexes_failed.add(index)
        return False


async def _search_single_index_async(query: str, index: str, top_k: int = 7) -> dict:
    from moss import QueryOptions

    client = await _get_moss()
    if not client:
        return {
            "query": query,
            "index": index,
            "results": [],
            "message": "MOSS not configured.",
        }

    if index not in _moss_indexes_ready:
        loaded = await _ensure_loaded(client, index)
        if not loaded:
            return {
                "query": query,
                "index": index,
                "results": [],
                "message": f"Failed to load Moss index '{index}'.",
            }

    try:
        res = await client.query(index, query, QueryOptions(top_k=top_k))
    except Exception as e:
        return {
            "query": query,
            "index": index,
            "results": [],
            "message": f"Moss query failed for '{index}': {e}",
        }

    hits: list[dict] = []
    for doc in (res.docs or []):
        if doc.score > _THRESHOLD and doc.text:
            hits.append({"text": doc.text, "score": doc.score})

    hits.sort(key=lambda r: r["score"], reverse=True)
    return {
        "query": query,
        "index": index,
        "results": [{"text": h["text"], "score": round(h["score"], 3)} for h in hits],
    }


def search_moss_pois(query: str, top_k: int = 7) -> dict:
    """Semantic search only over the Oakland POI Moss index."""
    return _run_async(_search_single_index_async(query=query, index=_INDEX_POIS, top_k=top_k))


def search_moss_flights(query: str, top_k: int = 7) -> dict:
    """Semantic search only over the Oakland flight Moss index."""
    return _run_async(_search_single_index_async(query=query, index=_INDEX_FLIGHTS, top_k=top_k))

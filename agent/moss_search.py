from __future__ import annotations

import asyncio
import threading

from agent.config import MOSS_PROJECT_ID, MOSS_PROJECT_KEY

_INDEX_POIS = "oakland-pois"
_INDEX_FLIGHTS = "oakland-flights"
_THRESHOLD = 0.8

# ── Persistent event loop ────────────────────────────────────────────────────
# A single background thread runs one event loop for the lifetime of the
# process. All Moss coroutines are submitted to it via run_coroutine_threadsafe,
# so load_index and every subsequent query share the same loop → the same
# asyncio.to_thread executor → the Rust IndexManager sees consistent state and
# has_index() returns True, enabling in-memory (~1-10ms) queries.

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            t = threading.Thread(target=_loop.run_forever, daemon=True)
            t.start()
    return _loop


def _run(coro):
    future = asyncio.run_coroutine_threadsafe(coro, _get_loop())
    return future.result()


# ── Client & index state ─────────────────────────────────────────────────────

_moss_client = None
_moss_indexes_ready: set[str] = set()
_moss_indexes_failed: set[str] = set()


async def _get_moss():
    global _moss_client
    if _moss_client is None:
        if not MOSS_PROJECT_ID or not MOSS_PROJECT_KEY:
            return None
        from moss import MossClient
        _moss_client = MossClient(MOSS_PROJECT_ID, MOSS_PROJECT_KEY)
    return _moss_client


_REFRESH_INTERVAL_SECONDS = 30  # how often the in-memory index polls for hydration updates


async def _ensure_loaded(client, index: str) -> bool:
    if index in _moss_indexes_failed:
        return False
    if index in _moss_indexes_ready:
        return True
    try:
        await client.load_index(
            index,
            auto_refresh=True,
            polling_interval_in_seconds=_REFRESH_INTERVAL_SECONDS,
        )
        _moss_indexes_ready.add(index)
        return True
    except Exception:
        _moss_indexes_failed.add(index)
        return False


# ── Public search functions ──────────────────────────────────────────────────

async def _search_async(query: str, index: str, top_k: int) -> dict:
    from moss import QueryOptions

    client = await _get_moss()
    if not client:
        return {"query": query, "index": index, "results": [], "message": "MOSS not configured."}

    if not await _ensure_loaded(client, index):
        return {"query": query, "index": index, "results": [], "message": f"Failed to load Moss index '{index}'."}

    try:
        res = await client.query(index, query, QueryOptions(top_k=top_k))
    except Exception as e:
        return {"query": query, "index": index, "results": [], "message": f"Moss query failed: {e}"}

    hits = sorted(
        [{"text": doc.text, "score": doc.score} for doc in (res.docs or []) if doc.score > _THRESHOLD and doc.text],
        key=lambda r: r["score"],
        reverse=True,
    )
    return {
        "query": query,
        "index": index,
        "results": [{"text": h["text"], "score": round(h["score"], 3)} for h in hits],
    }


def search_moss_pois(query: str, top_k: int = 7) -> dict:
    """Semantic search over the Oakland POI Moss index (in-memory, ~1-10ms)."""
    return _run(_search_async(query=query, index=_INDEX_POIS, top_k=top_k))


def search_moss_flights(query: str, top_k: int = 7) -> dict:
    """Semantic search over the Oakland flight Moss index (in-memory, ~1-10ms)."""
    return _run(_search_async(query=query, index=_INDEX_FLIGHTS, top_k=top_k))


# ── Eager preload at import time ─────────────────────────────────────────────
# Load both indexes into the persistent loop immediately so the first query
# is also in-memory fast. Failures are silenced — the search functions
# handle missing indexes gracefully.

async def _preload():
    client = await _get_moss()
    if not client:
        return
    for index in [_INDEX_POIS, _INDEX_FLIGHTS]:
        await _ensure_loaded(client, index)


try:
    _run(_preload())
except Exception:
    pass

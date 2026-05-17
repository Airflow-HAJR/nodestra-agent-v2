from __future__ import annotations

import asyncio
import concurrent.futures

from agent.config import DEFAULT_AIRPORT, MOSS_PROJECT_ID, MOSS_PROJECT_KEY
from agent.db import get_client

_moss_client = None
_INDEX_POIS = "oakland-pois"
_INDEX_FLIGHTS = "oakland-flights"


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


def _doc_info(doc_id: str, text: str, metadata: dict) -> object:
    from moss import DocumentInfo

    return DocumentInfo(id=doc_id, text=text, metadata=metadata)


def _poi_to_doc(poi: dict, level_name: str) -> dict | None:
    name = (poi.get("name") or "").strip()
    if not name:
        return None

    keywords = [kw for kw in (poi.get("keywords") or []) if kw]
    parts = [
        f"{name} ({poi.get('type') or 'location'}) on {level_name}.",
        f"Keywords: {', '.join(keywords)}." if keywords else None,
        poi.get("description"),
        f"Rated {poi['rating']}/5." if poi.get("rating") else None,
        f"Current deal: {poi['discount']}" if poi.get("discount") else None,
    ]
    return _doc_info(
        doc_id=f"poi_{poi['id']}",
        text=" ".join(part for part in parts if part),
        metadata={
            "type": poi.get("type"),
            "level": level_name,
            "poi_id": poi["id"],
        },
    )


async def _index_pois_async(airport_id: str) -> dict:
    client = await _get_moss()
    if not client:
        return {"ok": False, "index": _INDEX_POIS, "message": "MOSS not configured."}

    db = get_client()
    resp = (
        db.table("maps")
        .select("name, sort_order, graph_map")
        .eq("airport_id", airport_id)
        .order("sort_order")
        .execute()
    )

    docs = []
    for row in resp.data or []:
        graph_map = row.get("graph_map") or {}
        for poi in graph_map.get("pois", []) or []:
            doc = _poi_to_doc(poi, row["name"])
            if doc is not None:
                docs.append(doc)

    if not docs:
        return {"ok": False, "index": _INDEX_POIS, "message": f"No POIs found for airport_id={airport_id}."}

    try:
        await client.load_index(_INDEX_POIS)
        await client.add_docs(_INDEX_POIS, docs)
        action = "upserted"
    except Exception:
        await client.create_index(_INDEX_POIS, docs)
        action = "created"

    return {"ok": True, "index": _INDEX_POIS, "action": action, "docs_indexed": len(docs), "airport_id": airport_id}


def _flight_to_doc(flight: dict) -> dict | None:
    flight_number = flight.get("flight_number")
    if not flight_number:
        return None

    parts = [
        f"Flight {flight_number}",
        f"to {flight['airport_arriving_id']}" if flight.get("airport_arriving_id") else None,
        f"Gate {flight['gate']}" if flight.get("gate") else None,
        f"Terminal {flight['terminal']}" if flight.get("terminal") else None,
        f"departing from {flight['airport_departing_id']}" if flight.get("airport_departing_id") else None,
    ]
    return _doc_info(
        doc_id=f"flight_{flight_number}",
        text=", ".join(part for part in parts if part) + ".",
        metadata={
            "flight_number": flight_number,
            "gate": flight.get("gate"),
            "terminal": flight.get("terminal"),
            "airport_arriving_id": flight.get("airport_arriving_id"),
            "airport_departing_id": flight.get("airport_departing_id"),
        },
    )


async def _index_flights_async(airport_id: str) -> dict:
    client = await _get_moss()
    if not client:
        return {"ok": False, "index": _INDEX_FLIGHTS, "message": "MOSS not configured."}

    db = get_client()
    resp = (
        db.table("flights")
        .select("flight_number, airport_arriving_id, airport_departing_id, terminal, gate")
        .eq("airport_departing_id", airport_id)
        .execute()
    )

    docs = [doc for flight in (resp.data or []) if (doc := _flight_to_doc(flight)) is not None]
    if not docs:
        return {"ok": False, "index": _INDEX_FLIGHTS, "message": f"No flights found for airport_id={airport_id}."}

    try:
        await client.load_index(_INDEX_FLIGHTS)
        await client.add_docs(_INDEX_FLIGHTS, docs)
        action = "upserted"
    except Exception:
        await client.create_index(_INDEX_FLIGHTS, docs)
        action = "created"

    return {"ok": True, "index": _INDEX_FLIGHTS, "action": action, "docs_indexed": len(docs), "airport_id": airport_id}


def index_moss_pois(airport_id: str = DEFAULT_AIRPORT) -> dict:
    return _run_async(_index_pois_async(airport_id=airport_id))


def index_moss_flights(airport_id: str = DEFAULT_AIRPORT) -> dict:
    return _run_async(_index_flights_async(airport_id=airport_id))

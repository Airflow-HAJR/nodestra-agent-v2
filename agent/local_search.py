"""
Local vector search for flight and store data.
Backed by ChromaDB on disk. Swap _get_client() to migrate to Supabase pgvector.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"
_DB_DIR = Path(__file__).parent.parent / ".chroma_db"

_emb_fn = DefaultEmbeddingFunction()
_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(_DB_DIR))
    return _client


def _flight_to_text(f: dict) -> str:
    return (
        f"Flight {f['flight_number']} ({f['airline']}) to {f['destination']}. "
        f"Departs {f['departure_time']}, boarding at {f.get('boarding_time', 'TBD')}. "
        f"{f.get('terminal', '')} {f.get('gate', '')}. Status: {f.get('status', 'Unknown')}."
    )


def _store_to_text(s: dict) -> str:
    parts = [
        f"{s['name']} — {s['category']}. Located in {s.get('terminal', '?')}, {s.get('gate_area', '')}.",
        f"Hours: {s.get('hours', 'unknown')}.",
        s.get('description', ''),
    ]
    if s.get('dietary'):
        parts.append(f"Dietary info: {s['dietary']}")
    if s.get('perks'):
        parts.append(f"Card perks & loyalty: {s['perks']}")
    parts.append(f"Payment: {s.get('accepts', 'major cards')}.")
    return " ".join(parts)


def _build_collection(name: str, data_file: str, to_text) -> chromadb.Collection:
    client = _get_client()
    records: list[dict] = json.loads((_DATA_DIR / data_file).read_text())
    collection = client.get_or_create_collection(name, embedding_function=_emb_fn)
    if collection.count() != len(records):
        if collection.count() > 0:
            client.delete_collection(name)
            collection = client.create_collection(name, embedding_function=_emb_fn)
        collection.add(
            documents=[to_text(r) for r in records],
            ids=[f"{name}-{i}" for i in range(len(records))],
            metadatas=[{k: str(v) for k, v in r.items()} for r in records],
        )
        logger.info(f"Indexed {len(records)} records into '{name}'")
    return collection


def search_flights(query: str, top_k: int = 3) -> list[str]:
    try:
        col = _build_collection("oak_flights", "oak_flights.json", _flight_to_text)
        results = col.query(query_texts=[query], n_results=min(top_k, col.count()))
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        logger.error(f"search_flights failed: {e}")
        return []


def search_stores(query: str, top_k: int = 3) -> list[str]:
    try:
        col = _build_collection("oak_stores", "oak_stores.json", _store_to_text)
        results = col.query(query_texts=[query], n_results=min(top_k, col.count()))
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        logger.error(f"search_stores failed: {e}")
        return []

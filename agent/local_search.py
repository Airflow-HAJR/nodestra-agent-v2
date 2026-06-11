"""
Vector search for flight and store data.
Primary: Supabase pgvector via Azure OpenAI embeddings.
Fallback: keyword search over static JSON files (used when the embedding
deployment is unavailable or the vector_documents table is empty).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from agent.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    DEFAULT_AIRPORT,
)
from agent.db import get_service_client, supabase_ok

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"

# ── Text converters (shared with hydrator) ────────────────────────────────────

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
        s.get("description", ""),
    ]
    if s.get("dietary"):
        parts.append(f"Dietary info: {s['dietary']}")
    if s.get("perks"):
        parts.append(f"Card perks & loyalty: {s['perks']}")
    parts.append(f"Payment: {s.get('accepts', 'major cards')}.")
    return " ".join(parts)


# ── Keyword fallback ──────────────────────────────────────────────────────────

_static_cache: dict[str, list[str]] = {}


def _load_static(collection: str) -> list[str]:
    if collection in _static_cache:
        return _static_cache[collection]
    if collection == "flights":
        records = json.loads((_DATA_DIR / "oak_flights.json").read_text())
        texts = [_flight_to_text(r) for r in records]
    else:
        records = json.loads((_DATA_DIR / "oak_stores.json").read_text())
        texts = [_store_to_text(r) for r in records]
    _static_cache[collection] = texts
    return texts


def _keyword_search(collection: str, query: str, top_k: int) -> list[str]:
    texts = _load_static(collection)
    q_tokens = set(query.lower().replace("-", " ").split())
    scored = []
    for text in texts:
        t_lower = text.lower()
        score = sum(1 for tok in q_tokens if tok in t_lower)
        if score > 0:
            scored.append((score, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:top_k]]


# ── Vector search ─────────────────────────────────────────────────────────────

_embed_client = None
_embed_broken = False  # latched True on DeploymentNotFound so we stop retrying


def _embed(text: str) -> list[float]:
    global _embed_client, _embed_broken
    if _embed_broken:
        raise RuntimeError("embedding deployment unavailable")
    if _embed_client is None:
        from openai import AzureOpenAI
        _embed_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    try:
        resp = _embed_client.embeddings.create(
            input=text,
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        )
        return resp.data[0].embedding
    except Exception as e:
        if "DeploymentNotFound" in str(e) or "404" in str(e):
            _embed_broken = True
            logger.warning(
                "Embedding deployment '%s' not found — vector search disabled, "
                "using keyword fallback. Set AZURE_OPENAI_EMBEDDING_DEPLOYMENT to "
                "a deployed model name to enable semantic search.",
                AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            )
        raise


def _vector_search(collection: str, query: str, top_k: int) -> list[str]:
    embedding = _embed(query)
    db = get_service_client()
    resp = db.rpc("match_documents", {
        "query_embedding": embedding,
        "match_collection": collection,
        "match_airport": DEFAULT_AIRPORT,
        "match_count": top_k,
    }).execute()
    return [row["content"] for row in (resp.data or [])]


# ── Public interface ──────────────────────────────────────────────────────────

def _search(collection: str, query: str, top_k: int = 3) -> list[str]:
    if supabase_ok() and not _embed_broken:
        try:
            results = _vector_search(collection, query, top_k)
            if results:
                return results
            # Table empty — fall through to keyword search
        except Exception as e:
            if not _embed_broken:
                logger.debug("vector search failed (%s), using keyword fallback: %s", collection, e)
    return _keyword_search(collection, query, top_k)


def search_flights(query: str, top_k: int = 3) -> list[str]:
    return _search("flights", query, top_k)


def search_stores(query: str, top_k: int = 3) -> list[str]:
    return _search("stores", query, top_k)

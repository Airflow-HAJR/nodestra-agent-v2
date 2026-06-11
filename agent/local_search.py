"""
Vector search for flight and store data.
Backed by Supabase pgvector; embeddings via Azure OpenAI.
"""
from __future__ import annotations

import logging

from openai import AzureOpenAI

from agent.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    DEFAULT_AIRPORT,
)
from agent.db import get_service_client, supabase_ok

logger = logging.getLogger(__name__)

_embed_client: AzureOpenAI | None = None


def _get_embed_client() -> AzureOpenAI:
    global _embed_client
    if _embed_client is None:
        _embed_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
        )
    return _embed_client


def _embed(text: str) -> list[float]:
    resp = _get_embed_client().embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    )
    return resp.data[0].embedding


def _search(collection: str, query: str, top_k: int = 3) -> list[str]:
    if not supabase_ok():
        return []
    try:
        embedding = _embed(query)
        db = get_service_client()
        resp = db.rpc("match_documents", {
            "query_embedding": embedding,
            "match_collection": collection,
            "match_airport": DEFAULT_AIRPORT,
            "match_count": top_k,
        }).execute()
        return [row["content"] for row in (resp.data or [])]
    except Exception as e:
        logger.error("vector search failed (%s): %s", collection, e)
        return []


def search_flights(query: str, top_k: int = 3) -> list[str]:
    return _search("flights", query, top_k)


def search_stores(query: str, top_k: int = 3) -> list[str]:
    return _search("stores", query, top_k)

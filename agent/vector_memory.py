"""
Per-user semantic memory backed by Supabase pgvector.

Design for low latency:
  * At session start we fetch ALL of a user's memories once (fetch_all_memories),
    including their embeddings, and cache them in graph state.
  * Recall during the session runs entirely in-process (recall_from_cache) —
    embed the query once, cosine-rank against the cached embeddings. No DB round
    trip per turn.
  * When the agent writes a new memory (add_memory) we return the full row —
    embedding included — so the caller can append it to the cached set instead
    of re-fetching. That keeps the cache fresh for the rest of the session.

Everything degrades gracefully: if Supabase is unreachable or the embedding
deployment is missing, reads return [] and writes become no-ops.
"""
from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Optional

from agent.analytics import hash_user_id
from agent.db import (
    get_service_client as get_supabase,
    is_network_error,
    mark_supabase_unreachable,
    supabase_ok,
)
from agent.embeddings import embed, embeddings_available

logger = logging.getLogger(__name__)


# ── embedding (de)serialization ────────────────────────────────────────────────

def _parse_embedding(raw: Any) -> Optional[list[float]]:
    """pgvector comes back as a JSON string ('[0.1,0.2,...]') or a list depending
    on the client. Normalize to list[float]."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if isinstance(raw, (list, tuple)):
        try:
            return [float(x) for x in raw]
        except (ValueError, TypeError):
            return None
    return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ── session-start bulk load ─────────────────────────────────────────────────────

def fetch_all_memories(user_id: str) -> list[dict]:
    """Load every memory for a user. Returns rows shaped like
    {id, content, category, metadata, embedding: list[float] | None}."""
    if not supabase_ok():
        return []
    try:
        result = (
            get_supabase()
            .table("user_memories")
            .select("id, content, category, metadata, embedding")
            .eq("user_id_hash", hash_user_id(user_id))
            .order("created_at", desc=False)
            .execute()
        )
        rows = result.data or []
        for row in rows:
            row["embedding"] = _parse_embedding(row.get("embedding"))
        return rows
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("fetch_all_memories failed")
        return []


# ── in-session recall (no DB round trip) ────────────────────────────────────────

_STOPWORDS = {
    "the", "and", "for", "you", "your", "about", "any", "anything", "what", "which",
    "with", "have", "has", "does", "did", "can", "are", "was", "were", "that", "this",
    "remember", "know", "need", "want", "like", "some", "get", "find", "here", "there",
}


def _keyword_rank(cached: list[dict], query: str, top_k: int) -> list[dict]:
    """Fallback recall when embeddings are unavailable: rank by how many query
    words overlap each memory (substring-aware), not whole-string containment."""
    q_words = {w for w in re.split(r"\W+", query.lower()) if len(w) > 2 and w not in _STOPWORDS}
    if not q_words:
        # No topical words ("what do you remember about me?") — surface everything.
        return cached[:top_k]
    scored: list[tuple[float, dict]] = []
    for m in cached:
        c_words = {w for w in re.split(r"\W+", (m.get("content") or "").lower()) if len(w) > 2}
        overlap = sum(1 for qw in q_words if any(qw in cw or cw in qw for cw in c_words))
        if overlap:
            scored.append((overlap / len(q_words), m))
    scored.sort(key=lambda x: -x[0])
    return [m for score, m in scored if score >= 0.15][:top_k]


def recall_from_cache(
    cached: list[dict] | None,
    query: str,
    top_k: int = 5,
    threshold: float = 0.30,
) -> list[dict]:
    """Semantic search over the cached memory set. Embeds the query once and
    ranks by cosine similarity. Falls back to keyword-overlap ranking if
    embeddings are unavailable."""
    if not cached:
        return []

    if not embeddings_available():
        return _keyword_rank(cached, query, top_k)

    try:
        q_emb = embed(query)
    except Exception:
        return _keyword_rank(cached, query, top_k)

    scored: list[tuple[float, dict]] = []
    for m in cached:
        emb = m.get("embedding")
        if not emb:
            continue
        score = _cosine(q_emb, emb)
        if score >= threshold:
            scored.append((score, m))
    scored.sort(key=lambda x: -x[0])
    return [m for _, m in scored[:top_k]]


# ── writes ──────────────────────────────────────────────────────────────────────

def add_memory(
    user_id: str,
    content: str,
    category: str = "other",
    metadata: dict | None = None,
) -> Optional[dict]:
    """Embed *content* and persist it as a memory for the user. Returns the
    inserted row (embedding included) so the caller can append it to the session
    cache, or None if the write couldn't happen."""
    content = (content or "").strip()
    if not content or not supabase_ok():
        return None

    embedding: Optional[list[float]] = None
    if embeddings_available():
        try:
            embedding = embed(content)
        except Exception:
            logger.debug("embed failed for memory; storing without vector")

    row: dict[str, Any] = {
        "user_id_hash": hash_user_id(user_id),
        "content": content,
        "category": category or "other",
        "metadata": metadata or {},
    }
    if embedding is not None:
        row["embedding"] = embedding

    try:
        result = get_supabase().table("user_memories").insert(row).execute()
        inserted = (result.data or [{}])[0]
        # Return a cache-ready shape with the parsed embedding we already have.
        return {
            "id": inserted.get("id"),
            "content": content,
            "category": row["category"],
            "metadata": row["metadata"],
            "embedding": embedding,
        }
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("add_memory failed")
        return None


def is_duplicate(cached: list[dict] | None, content: str, threshold: float = 0.92) -> bool:
    """True if *content* is semantically near-identical to something already
    remembered — avoids storing the same fact twice across turns."""
    if not cached:
        return False
    content_l = content.strip().lower()
    if any(content_l == (m.get("content") or "").strip().lower() for m in cached):
        return True
    if not embeddings_available():
        return False
    try:
        emb = embed(content)
    except Exception:
        return False
    for m in cached:
        e = m.get("embedding")
        if e and _cosine(emb, e) >= threshold:
            return True
    return False

"""
Per-user memory backed by Supabase.

Memories are plain-text facts, one row per fact, keyed by the hashed user id.
There is no embedding / vector search: the LangGraph `recall_memory` node uses
the chat model to decide which memories are relevant, so storage stays simple.

  * fetch_all_memories  — load a user's memories once at session start
  * add_memory          — persist one new fact
  * is_duplicate        — guard against storing the same fact twice
  * delete_memory       — forget one fact (from the account sheet)
  * delete_all_memories — forget everything about a user

Degrades gracefully: if Supabase is unreachable, reads return [] and writes
are no-ops.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agent.analytics import hash_user_id
from agent.db import (
    get_service_client as get_supabase,
    is_network_error,
    mark_supabase_unreachable,
    supabase_ok,
)

logger = logging.getLogger(__name__)


def fetch_all_memories(user_id: str) -> list[dict]:
    """Load every memory for a user: [{id, content, category, metadata}, ...]."""
    if not supabase_ok():
        return []
    try:
        result = (
            get_supabase()
            .table("user_memories")
            .select("id, content, category, metadata")
            .eq("user_id_hash", hash_user_id(user_id))
            .order("created_at", desc=False)
            .execute()
        )
        return result.data or []
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("fetch_all_memories failed")
        return []


def add_memory(
    user_id: str,
    content: str,
    category: str = "other",
    metadata: dict | None = None,
) -> Optional[dict]:
    """Persist one fact. Returns the inserted row (cache-ready) or None on failure."""
    content = (content or "").strip()
    if not content or not supabase_ok():
        return None
    row: dict[str, Any] = {
        "user_id_hash": hash_user_id(user_id),
        "content": content,
        "category": category or "other",
        "metadata": metadata or {},
    }
    try:
        result = get_supabase().table("user_memories").insert(row).execute()
        inserted = (result.data or [{}])[0]
        return {
            "id": inserted.get("id"),
            "content": content,
            "category": row["category"],
            "metadata": row["metadata"],
        }
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("add_memory failed")
        return None


def delete_memory(user_id: str, memory_id: str) -> bool:
    """Forget one fact. The user_id_hash is part of the WHERE clause, not just
    the id, so a caller can only ever delete their own rows."""
    if not supabase_ok():
        return False
    try:
        result = (
            get_supabase()
            .table("user_memories")
            .delete()
            .eq("id", memory_id)
            .eq("user_id_hash", hash_user_id(user_id))
            .execute()
        )
        return bool(result.data)
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("delete_memory failed")
        return False


def delete_all_memories(user_id: str) -> int:
    """Forget everything stored about a user. Returns how many rows went."""
    if not supabase_ok():
        return 0
    try:
        result = (
            get_supabase()
            .table("user_memories")
            .delete()
            .eq("user_id_hash", hash_user_id(user_id))
            .execute()
        )
        return len(result.data or [])
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("delete_all_memories failed")
        return 0


def is_duplicate(cached: list[dict] | None, content: str) -> bool:
    """True if this exact fact (case-insensitive) is already remembered."""
    if not cached:
        return False
    c = content.strip().lower()
    return any(c == (m.get("content") or "").strip().lower() for m in cached)

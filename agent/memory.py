import logging
import re

from agent.analytics import _safe_data, hash_user_id, upsert_user_memory
from agent.config import DEFAULT_AIRPORT
from agent.db import is_network_error, mark_supabase_unreachable, supabase_ok, get_client as get_supabase

logger = logging.getLogger(__name__)

_RELEVANCE_THRESHOLD = 0.3


def _tokenize(text: str) -> list[str]:
    return [w for w in re.split(r"\W+", text.lower()) if w]


def _relevance(query_words: set[str], key: str, value: str) -> float:
    """Fraction of query words that substring-match any word in the key or value."""
    entry_words = set(_tokenize(key)) | set(_tokenize(value))
    matched = sum(
        1 for qw in query_words
        if any(qw in ew or ew in qw for ew in entry_words)
    )
    return matched / max(len(query_words), 1)


def search_memories(user_id: str, query: str) -> str:
    """Return user memory facts relevant to *query*, above a relevance threshold."""
    if not supabase_ok():
        return "No relevant user memories found."
    try:
        result = (
            get_supabase()
            .table("user_memory")
            .select("profile_facts, last_flight, last_location_name")
            .eq("user_id_hash", hash_user_id(user_id))
            .maybe_single()
            .execute()
        )
        data = _safe_data(result)
        if not data:
            return "No relevant user memories found."

        facts: dict[str, str] = {}
        if data.get("last_flight"):
            facts["last_flight"] = data["last_flight"]
        if data.get("last_location_name"):
            facts["last_location"] = data["last_location_name"]
        facts.update(data.get("profile_facts") or {})

        if not facts:
            return "No relevant user memories found."

        query_words = set(_tokenize(query))
        matches: list[tuple[float, str, str]] = []
        for key, value in facts.items():
            score = _relevance(query_words, key, str(value))
            if score >= _RELEVANCE_THRESHOLD:
                matches.append((score, key, value))

        if not matches:
            return "No relevant user memories found."

        matches.sort(key=lambda x: -x[0])
        return "\n".join(f"{k}: {v}" for _, k, v in matches)
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("search_memories failed")
        return "No relevant user memories found."


def get_last_location(user_id: str) -> str | None:
    if not supabase_ok():
        return None
    try:
        result = (
            get_supabase()
            .table("user_memory")
            .select("last_location")
            .eq("user_id_hash", hash_user_id(user_id))
            .maybe_single()
            .execute()
        )
        return _safe_data(result).get("last_location")
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("get_last_location failed")
        return None


def update_location(user_id: str, poi_id: str, poi_name: str) -> None:
    upsert_user_memory(
        hash_user_id(user_id),
        DEFAULT_AIRPORT,
        last_location=poi_id,
        last_location_name=poi_name,
    )


def get_last_flight(user_id: str) -> str | None:
    if not supabase_ok():
        return None
    try:
        result = (
            get_supabase()
            .table("user_memory")
            .select("last_flight")
            .eq("user_id_hash", hash_user_id(user_id))
            .maybe_single()
            .execute()
        )
        return _safe_data(result).get("last_flight")
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("get_last_flight failed")
        return None


def update_flight(user_id: str, flight_number: str) -> None:
    upsert_user_memory(
        hash_user_id(user_id),
        DEFAULT_AIRPORT,
        last_flight=flight_number,
    )


def get_user_profile(user_id: str) -> dict:
    if not supabase_ok():
        return {}
    try:
        result = (
            get_supabase()
            .table("user_memory")
            .select("profile_facts, last_flight, last_location_name")
            .eq("user_id_hash", hash_user_id(user_id))
            .maybe_single()
            .execute()
        )
        return _safe_data(result)
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("get_user_profile failed")
        return {}


def save_conversation(user_id: str, messages: list) -> None:
    pass

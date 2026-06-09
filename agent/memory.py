import logging

from agent.analytics import _safe_data, hash_user_id, upsert_user_memory
from agent.config import DEFAULT_AIRPORT
from agent.db import get_client as get_supabase

logger = logging.getLogger(__name__)


def search_memories(user_id: str, query: str) -> str:
    return "No relevant user memories found."


def get_last_location(user_id: str) -> str | None:
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
    except Exception:
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
    except Exception:
        logger.exception("get_last_flight failed")
        return None


def update_flight(user_id: str, flight_number: str) -> None:
    upsert_user_memory(
        hash_user_id(user_id),
        DEFAULT_AIRPORT,
        last_flight=flight_number,
    )


def get_user_profile(user_id: str) -> dict:
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
    except Exception:
        logger.exception("get_user_profile failed")
        return {}


def save_conversation(user_id: str, messages: list) -> None:
    pass

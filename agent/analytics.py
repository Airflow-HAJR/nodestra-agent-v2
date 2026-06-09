import hashlib
import logging
import time
import uuid
from typing import Any

from agent.config import DEFAULT_AIRPORT
from agent.db import is_network_error, mark_supabase_unreachable, supabase_ok, get_service_client as get_supabase

logger = logging.getLogger(__name__)


_TWILIO_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID namespace for Twilio SIDs


def sid_to_uuid(sid: str) -> str:
    """Convert a Twilio SID to a deterministic UUID5 so the calls table (uuid column) accepts it."""
    return str(uuid.uuid5(_TWILIO_NS, sid))


def hash_user_id(raw_phone: str) -> str:
    return hashlib.sha256(raw_phone.encode()).hexdigest()[:32]


def _safe_data(result: Any) -> dict:
    """Safely extract .data from a supabase response — handles None result."""
    if result is None:
        return {}
    return result.data or {}


def upsert_user_memory(
    user_id_hash: str,
    airport_id: str = DEFAULT_AIRPORT,
    *,
    last_flight: str | None = None,
    last_location: str | None = None,
    last_location_name: str | None = None,
    profile_facts_patch: dict[str, str] | None = None,
) -> None:
    if not supabase_ok():
        return
    try:
        sb = get_supabase()

        existing_data = _safe_data(
            sb.table("user_memory")
            .select("profile_facts, visit_count")
            .eq("user_id_hash", user_id_hash)
            .maybe_single()
            .execute()
        )
        existing_facts: dict = existing_data.get("profile_facts") or {}
        visit_count: int = (existing_data.get("visit_count") or 0) + 1
        merged_facts = {**existing_facts, **(profile_facts_patch or {})}

        row: dict[str, Any] = {
            "user_id_hash": user_id_hash,
            "airport_id": airport_id,
            "visit_count": visit_count,
            "last_seen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "profile_facts": merged_facts,
        }
        if last_flight is not None:
            row["last_flight"] = last_flight
        if last_location is not None:
            row["last_location"] = last_location
        if last_location_name is not None:
            row["last_location_name"] = last_location_name

        sb.table("user_memory").upsert(row).execute()
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("upsert_user_memory failed — analytics skipped")


def start_call(
    *,
    call_id: str,
    airport_id: str = DEFAULT_AIRPORT,
    user_id_hash: str | None,
    started_at: float,
) -> None:
    """Insert minimal calls row at session start so turns can FK-reference it."""
    if not supabase_ok():
        return
    try:
        sb = get_supabase()
        # user_memory must exist before calls (FK). Upsert a minimal row if needed.
        if user_id_hash:
            existing = _safe_data(
                sb.table("user_memory")
                .select("user_id_hash")
                .eq("user_id_hash", user_id_hash)
                .maybe_single()
                .execute()
            )
            if not existing:
                sb.table("user_memory").insert({
                    "user_id_hash": user_id_hash,
                    "airport_id": airport_id,
                    "visit_count": 1,
                    "last_seen": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "profile_facts": {},
                }).execute()
        sb.table("calls").insert({
            "call_id": sid_to_uuid(call_id),
            "airport_id": airport_id,
            "user_id_hash": user_id_hash,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
            "turn_count": 0,
            "resolved": False,
        }).execute()
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("start_call failed — analytics skipped")


def insert_turn(
    *,
    call_id: str,
    turn_number: int,
    turn_stats: dict,
    tools_used: list[str],
) -> None:
    if not supabase_ok():
        return
    try:
        sb = get_supabase()
        sb.table("turns").insert({
            "call_id": sid_to_uuid(call_id),
            "turn_number": turn_number,
            "llm_ms": turn_stats.get("llm_ms"),
            "llm_calls": turn_stats.get("llm_calls"),
            "tool_ms": turn_stats.get("tool_ms"),
            "tool_calls": turn_stats.get("tool_calls"),
            "tts_ms": turn_stats.get("tts_ms"),
            "total_ms": turn_stats.get("total_ms"),
            "other_ms": turn_stats.get("other_ms"),
            "tools_used": tools_used,
        }).execute()
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("insert_turn failed — analytics skipped")


def finish_call(
    *,
    call_id: str,
    duration_s: float,
    turn_count: int,
    timing: dict,
    flight_number: str | None,
    topics: list[str],
    resolved: bool,
    summary: str | None,
) -> None:
    """Update the calls row created by start_call() with final stats."""
    if not supabase_ok():
        return
    try:
        sb = get_supabase()
        sb.table("calls").update({
            "duration_s": round(duration_s, 2),
            "turn_count": turn_count,
            "total_llm_ms": timing.get("total_llm_ms"),
            "total_tool_ms": timing.get("total_tool_ms"),
            "total_tts_ms": timing.get("total_tts_ms"),
            "llm_calls": timing.get("llm_calls"),
            "tool_calls": timing.get("tool_calls"),
            "flight_number": flight_number,
            "topics": topics,
            "resolved": resolved,
            "summary": summary,
        }).eq("call_id", sid_to_uuid(call_id)).execute()
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("finish_call failed — analytics skipped")


# Keep insert_call as an alias for callers that do a one-shot write (e.g. SMS END)
def insert_call(
    *,
    call_id: str,
    airport_id: str = DEFAULT_AIRPORT,
    user_id_hash: str | None,
    started_at: float,
    duration_s: float,
    turn_count: int,
    timing: dict,
    flight_number: str | None,
    topics: list[str],
    resolved: bool,
    summary: str | None,
) -> None:
    if not supabase_ok():
        return
    try:
        sb = get_supabase()
        sb.table("calls").upsert({
            "call_id": sid_to_uuid(call_id),
            "airport_id": airport_id,
            "user_id_hash": user_id_hash,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at)),
            "duration_s": round(duration_s, 2),
            "turn_count": turn_count,
            "total_llm_ms": timing.get("total_llm_ms"),
            "total_tool_ms": timing.get("total_tool_ms"),
            "total_tts_ms": timing.get("total_tts_ms"),
            "llm_calls": timing.get("llm_calls"),
            "tool_calls": timing.get("tool_calls"),
            "flight_number": flight_number,
            "topics": topics,
            "resolved": resolved,
            "summary": summary,
        }).execute()
    except Exception as exc:
        if is_network_error(exc):
            mark_supabase_unreachable()
        logger.exception("insert_call failed — analytics skipped")

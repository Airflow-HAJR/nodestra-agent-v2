import json
from typing import Annotated, List, Optional

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.analytics import hash_user_id, upsert_user_memory
from agent.config import DEFAULT_AIRPORT
from agent.db import load_map_levels
from agent.map_engine import (
    dijkstra_multilevel,
    fmt_poi,
    format_route_speech,
    matching_poi_types,
    search_pois,
)
from agent.memory import get_last_location, save_conversation, update_flight, update_location


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class POIInput(BaseModel):
    q: str = Field(description="POI name like Gate 5, Escape Lounge")
    airport_id: str = Field(default=DEFAULT_AIRPORT)


class RouteInput(BaseModel):
    start: str = Field(description="start POI id (the 'id' field returned by find_poi, e.g. 'gate-CIU3')")
    end: str = Field(description="end POI id (the 'id' field returned by find_poi, e.g. 'lounge-MYAS')")
    airport_id: str = Field(default=DEFAULT_AIRPORT)


class GetNodesInput(BaseModel):
    airport_id: str = Field(default=DEFAULT_AIRPORT)
    floor: Optional[str] = Field(default=None, description="Optional floor filter, e.g. 'L1', 'L2'")


class ResolvePOIInput(BaseModel):
    name: str = Field(description="Ambiguous POI name (same name appears on multiple floors)")
    candidates: List[dict] = Field(description="List of POI candidates returned by find_poi")
    landmark: Optional[str] = Field(default=None, description="Nearby landmark hint from the user")
    floor: Optional[str] = Field(default=None, description="Floor hint if user provided one")
    airport_id: str = Field(default=DEFAULT_AIRPORT)


class FindNearestInput(BaseModel):
    source: str = Field(description="Source POI id to search from")
    poi_type: str = Field(description="Type of POI to find, e.g. 'restroom', 'gate', 'lounge'")
    airport_id: str = Field(default=DEFAULT_AIRPORT)


# ---------------------------------------------------------------------------
# Navigation tools
# ---------------------------------------------------------------------------

@tool(args_schema=POIInput)
def find_poi(q: str, airport_id: str = DEFAULT_AIRPORT):
    """Fuzzy search for a single POI name. Returns a dict with an 'id' field to pass to get_route."""
    print("[thinking]")
    levels = load_map_levels(airport_id)
    results = search_pois(levels, q)

    if not results:
        return {"found": False, "message": f"No POI found matching '{q}'."}

    best_score = results[0]["_score"]
    top = [r for r in results if r["_score"] == best_score]

    unique_names = {r["name"].lower().strip() for r in top}
    if len(top) > 1 and len(unique_names) == 1:
        return {
            "found": False,
            "ambiguous": True,
            "message": (
                f"Found {len(top)} locations named '{top[0]['name']}' on different floors. "
                "Call resolve_poi with the name and any nearby landmarks to identify which one."
            ),
            "matches": [fmt_poi(r) for r in top],
        }

    return {"found": True, **fmt_poi(top[0])}


@tool(args_schema=ResolvePOIInput)
def resolve_poi(
    name: str,
    candidates: List[dict],
    landmark: Optional[str] = None,
    floor: Optional[str] = None,
    airport_id: str = DEFAULT_AIRPORT,
):
    """Disambiguate POIs with the same name on different floors using a nearby landmark or floor hint."""
    print("[thinking]")
    levels = load_map_levels(airport_id)

    if floor:
        floor_lower = floor.lower()
        for c in candidates:
            if floor_lower in (c.get("level") or "").lower():
                return {"found": True, "id": c["id"], "name": c["name"], "type": c.get("type", ""), "level": c.get("level", "")}

    if not landmark:
        c = candidates[0]
        return {"found": True, "id": c["id"], "name": c["name"], "type": c.get("type", ""), "level": c.get("level", "")}

    lm_results = search_pois(levels, landmark)
    if not lm_results:
        c = candidates[0]
        return {"found": True, "id": c["id"], "name": c["name"], "type": c.get("type", ""), "level": c.get("level", "")}

    lm_poi = lm_results[0]
    best_candidate = None
    best_distance = float("inf")

    for c in candidates:
        result = dijkstra_multilevel(levels, c["id"], lm_poi["id"])
        if result is None:
            continue
        if result["distance"] < best_distance:
            best_distance = result["distance"]
            best_candidate = c

    if best_candidate is None:
        best_candidate = candidates[0]

    return {
        "found": True,
        "id": best_candidate["id"],
        "name": best_candidate["name"],
        "type": best_candidate.get("type", ""),
        "level": best_candidate.get("level", ""),
    }


@tool(args_schema=RouteInput)
def get_route(start: str, end: str, airport_id: str = DEFAULT_AIRPORT, state: Annotated[dict, InjectedState] = {}):
    """Get shortest path between two POIs. start and end must be POI ids returned by find_poi."""
    print("[thinking]")

    levels = load_map_levels(airport_id)
    result = dijkstra_multilevel(levels, start, end)

    if not result:
        return {"found": False, "stops": [], "level_changes": [], "distance": 0, "estimated_minutes": 0}

    raw = {
        "found": True,
        "stops": [
            {
                "step": i + 1,
                "id": p["id"],
                "name": p["name"],
                "type": p.get("type", ""),
                "level": p.get("level_name", ""),
            }
            for i, p in enumerate(result["poi_stops"])
        ],
        "level_changes": result["level_changes"],
        "distance": round(result["distance"], 4),
        "estimated_minutes": max(1, round(result["distance"] * 10)),
    }
    return format_route_speech(raw)


@tool(args_schema=GetNodesInput)
def get_nodes(airport_id: str = DEFAULT_AIRPORT, floor: Optional[str] = None):
    """List all POI nodes for an airport, optionally filtered by floor."""
    print("[thinking]")
    levels = load_map_levels(airport_id)
    nodes = []
    for lvl in levels:
        if floor and lvl["name"] != floor:
            continue
        nodes.extend(lvl["pois"])
    return nodes


@tool(args_schema=FindNearestInput)
def find_nearest(source: str, poi_type: str, airport_id: str = DEFAULT_AIRPORT, state: Annotated[dict, InjectedState] = {}):
    """Find the closest POI of a given type from a source location."""
    # Refresh source from stored location so detours always use the latest location.
    print("[thinking]")
    user_id = state.get("user_id") if state else None
    if user_id:
        fresh = get_last_location(user_id)
        if fresh and fresh != source:
            source = fresh

    levels = load_map_levels(airport_id)

    source_found = any(p["id"] == source for lvl in levels for p in lvl["pois"])
    if not source_found:
        return {"found": False, "message": f"Source POI '{source}' not found."}

    matched_types = matching_poi_types(poi_type, levels)
    if not matched_types:
        return {"found": False, "message": f"No POIs of type '{poi_type}' found."}

    candidates = [
        {**p, "_level_name": lvl["name"]}
        for lvl in levels
        for p in lvl["pois"]
        if p["id"] != source and (p.get("type") or "").lower().strip() in matched_types
    ]
    if not candidates:
        return {"found": False, "message": f"No POIs of type '{poi_type}' found."}

    best_candidate = None
    best_distance = float("inf")
    for c in candidates:
        result = dijkstra_multilevel(levels, source, c["id"])
        if result is None:
            continue
        if result["distance"] < best_distance:
            best_distance = result["distance"]
            best_candidate = c

    if best_candidate is None:
        return {"found": False, "message": f"No reachable POI of type '{poi_type}'."}

    return {
        "found": True,
        "id": best_candidate["id"],
        "name": best_candidate.get("name") or poi_type.capitalize(),
        "type": best_candidate.get("type", ""),
        "level": best_candidate["_level_name"],
        "distance": round(best_distance, 4),
        "estimated_minutes": max(1, round(best_distance * 10)),
    }


# ---------------------------------------------------------------------------
# State tool (LangGraph only — not a navigation API call)
# ---------------------------------------------------------------------------

@tool
def set_nav_state(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    final_destination: Optional[str] = None,
    current_location: Optional[str] = None,
) -> Command:
    """Update remembered navigation state.

    Call this when the user tells you a new destination or confirms they've moved to a new checkpoint.
    Pass POI ids (e.g. 'gate-J292'), not raw names. Pass only the field(s) that changed.
    """
    print("[thinking]")
    levels = load_map_levels(DEFAULT_AIRPORT)
    poi_by_id = {p["id"]: p for lvl in levels for p in lvl["pois"]}
    valid_ids = set(poi_by_id)

    errors = []
    if final_destination is not None and final_destination not in valid_ids:
        errors.append(f"'{final_destination}' is not a valid POI id — call find_poi first to get the id, then pass the 'id' field here.")
    if current_location is not None and current_location not in valid_ids:
        errors.append(f"'{current_location}' is not a valid POI id — call find_poi first to get the id, then pass the 'id' field here.")

    if errors:
        msg = "Error: " + " | ".join(errors)
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})

    existing_intents = list(state.get("active_intents") or [])
    nav_update: dict = {
        "messages": [ToolMessage(content="nav state updated", tool_call_id=tool_call_id)],
        "active_intents": list({*existing_intents, "navigate"}),
    }
    if final_destination is not None:
        nav_update["final_destination"] = final_destination
    if current_location is not None:
        nav_update["current_location"] = current_location
        user_id = state.get("user_id")
        if user_id:
            poi = poi_by_id.get(current_location, {})
            poi_name = poi.get("name") or current_location
            update_location(user_id, current_location, poi_name)
    return Command(update=nav_update)


# ---------------------------------------------------------------------------
# Flight state tool
# ---------------------------------------------------------------------------

class SetFlightInput(BaseModel):
    flight_number: str = Field(description="The user's flight number, e.g. 'UA 2345' or 'SW 891'")


@tool
def set_flight_number(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    flight_number: str,
) -> Command:
    """Save the user's flight number to state."""
    print("[thinking]")
    user_id = state.get("user_id")
    if user_id:
        update_flight(user_id, flight_number)
    return Command(update={
        "flight_number": flight_number,
        "messages": [ToolMessage(content=f"Flight number set to {flight_number}", tool_call_id=tool_call_id)],
    })


# ---------------------------------------------------------------------------
# Call end tool
# ---------------------------------------------------------------------------

@tool
def end_call(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Signal that the conversation is complete and the call should end.

    Call this when the user indicates they are done: farewell phrases
    ("bye", "that's all", "I'm good", "no thanks", "goodbye", "thanks bye", "I'm set"),
    or when you just confirmed the user arrived at their destination and they have no further requests.
    After calling this tool, deliver a brief closing message (e.g. "Safe travels!") and stop asking questions.
    """
    user_id = state.get("user_id")
    if user_id:
        save_conversation(user_id, state.get("messages", []))
    return Command(update={
        "should_end": True,
        "messages": [ToolMessage(content="call ended", tool_call_id=tool_call_id)],
    })


# ---------------------------------------------------------------------------
# User memory tools
# ---------------------------------------------------------------------------

@tool
def recall_user_memories(state: Annotated[dict, InjectedState] = {}) -> str:
    """Retrieve everything stored about this user from previous conversations —
    preferences, loyalty cards, flight history, home city, etc.
    Call this whenever the user asks what you remember about them,
    or when knowing their history would help you assist them better."""
    print("[thinking]")
    user_id = state.get("user_id") if state else None
    if not user_id:
        return "No user profile — this is a guest session with no stored memories."
    try:
        from agent.analytics import _safe_data, hash_user_id
        from agent.db import get_service_client
        result = _safe_data(
            get_service_client()
            .table("user_memory")
            .select("profile_facts, last_flight, last_location_name, visit_count, last_seen")
            .eq("user_id_hash", hash_user_id(user_id))
            .maybe_single()
            .execute()
        )
        if not result:
            return "No memories stored for this user yet."
        parts: list[str] = []
        if result.get("visit_count"):
            parts.append(f"visit_count: {result['visit_count']}")
        if result.get("last_flight"):
            parts.append(f"last_flight: {result['last_flight']}")
        if result.get("last_location_name"):
            parts.append(f"last_location: {result['last_location_name']}")
        facts: dict = result.get("profile_facts") or {}
        for k, v in facts.items():
            parts.append(f"{k}: {v}")
        return "\n".join(parts) if parts else "No memories stored for this user yet."
    except Exception as e:
        return f"Could not retrieve memories: {e}"


class UpdateUserMemoryInput(BaseModel):
    facts: dict[str, str] = Field(
        description=(
            "Key-value facts about this user to remember for future calls. "
            "E.g. {'preferred_airline': 'Southwest', 'card': 'Amex Platinum', 'home_city': 'Seattle'}"
        )
    )


@tool(args_schema=UpdateUserMemoryInput)
def update_user_memory(
    facts: dict[str, str],
    state: Annotated[dict, InjectedState] = {},
) -> str:
    """Persist anything useful learned about this user for future calls.
    Call whenever the user reveals a preference, loyalty card, home city, airline, accessibility need, etc."""
    print("[thinking]")
    user_id = state.get("user_id") if state else None
    if user_id:
        upsert_user_memory(hash_user_id(user_id), DEFAULT_AIRPORT, profile_facts_patch=facts)
    return "Got it, I'll remember that."


# ---------------------------------------------------------------------------
# Local vector search tools
# ---------------------------------------------------------------------------

class SearchFlightsInput(BaseModel):
    query: str = Field(description="Natural language query about a flight, e.g. 'United flight to Denver' or 'WN 2341'")

@tool(args_schema=SearchFlightsInput)
def search_flight_info(query: str) -> str:
    """Look up OAK departure info — gate, time, boarding status — for a specific flight or airline."""
    from agent.local_search import search_flights
    results = search_flights(query)
    if not results:
        return "No matching flights found in the departure board."
    return "\n---\n".join(results)


class SearchStoresInput(BaseModel):
    query: str = Field(description="Natural language query about stores or restaurants, e.g. 'coffee near gate 10' or 'sushi'")

@tool(args_schema=SearchStoresInput)
def search_store_info(query: str) -> str:
    """Look up OAK airport stores, restaurants, and amenities — hours, location, payment options."""
    from agent.local_search import search_stores
    results = search_stores(query)
    if not results:
        return "No matching stores or restaurants found."
    return "\n---\n".join(results)


TOOLS = [
    find_poi,
    get_route,
    get_nodes,
    resolve_poi,
    find_nearest,
    set_nav_state,
    set_flight_number,
    end_call,
    update_user_memory,
    recall_user_memories,
    search_flight_info,
    search_store_info,
]

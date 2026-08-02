import uuid
from typing import Annotated, List, Optional

import httpx
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.config import DEFAULT_AIRPORT, GATEGETTER_URL
from agent.db import load_map_levels
from agent.map_engine import (
    dijkstra_multilevel,
    fmt_poi,
    format_route_speech,
    matching_poi_types,
    search_pois,
)
from agent.map_tools import MAP_TOOLS


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


def _fmt_stop(i: int, p: dict) -> dict:
    return {
        "step": i + 1,
        "id": p["id"],
        "name": p["name"],
        "type": p.get("type", ""),
        "level": p.get("level_name", ""),
    }


@tool(args_schema=RouteInput)
def get_route(
    start: str,
    end: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    airport_id: str = DEFAULT_AIRPORT,
) -> Command:
    """Get shortest path between two POIs. start and end must be POI ids returned by find_poi."""
    print("[thinking]")

    levels = load_map_levels(airport_id)
    result = dijkstra_multilevel(levels, start, end)

    if not result:
        msg = ToolMessage(content="No route found.", tool_call_id=tool_call_id)
        return Command(update={"messages": [msg], "last_route": None, "active_segment_index": None})

    raw = {
        "found": True,
        "stops": [_fmt_stop(i, p) for i, p in enumerate(result["poi_stops"])],
        "level_changes": result["level_changes"],
        "segments": [
            {
                "level_name": seg["level_name"],
                "stops": [_fmt_stop(i, p) for i, p in enumerate(seg["stops"])],
                "portal_out": _fmt_stop(0, seg["portal_out"]) if seg["portal_out"] else None,
            }
            for seg in result["segments"]
        ],
        "distance": round(result["distance"], 4),
        "estimated_minutes": max(1, round(result["distance"] * 10)),
    }
    speech = format_route_speech(raw)
    msg = ToolMessage(content=speech, tool_call_id=tool_call_id)
    return Command(update={
        "messages": [msg],
        "last_route": raw,
        "active_segment_index": 0,
        "active_stop_index": 1,
        "route_id": str(uuid.uuid4()),
    })


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
    print("[thinking]")
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
    return Command(update={
        "flight_number": flight_number,
        "messages": [ToolMessage(content=f"Flight number set to {flight_number}", tool_call_id=tool_call_id)],
    })


# ---------------------------------------------------------------------------
# Direct flight status lookup (fast path — bypasses vector search)
# ---------------------------------------------------------------------------

class GetFlightStatusInput(BaseModel):
    flight_number: str = Field(description="Exact flight number, e.g. '5X 7849' or 'UA2345'")
    airport_id: str = Field(default=DEFAULT_AIRPORT)


@tool(args_schema=GetFlightStatusInput)
def get_flight_status(flight_number: str, airport_id: str = DEFAULT_AIRPORT) -> str:
    """Get real-time gate, departure time, and status for a specific flight number.
    Use this whenever you have an exact flight number — it queries the live departure
    board directly."""
    print("[thinking]")
    needle = flight_number.upper().replace(" ", "")
    try:
        resp = httpx.get(f"{GATEGETTER_URL}/api/data?airport={airport_id}", timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"Could not reach flight data service: {e}"

    flights = resp.json().get("flights", [])
    match = next(
        (f for f in flights if (f.get("flight_number") or "").upper().replace(" ", "") == needle),
        None,
    )
    if not match:
        return f"Flight {flight_number.upper()} not found on the {airport_id} departure board."

    dest_city = match.get("destination_city", "")
    dest_iata = match.get("destination_iata", "")
    destination = f"{dest_city} ({dest_iata})" if dest_iata else dest_city
    departs = match.get("actual_time") or match.get("scheduled_time", "TBD")
    gate_info = f"{match.get('terminal', '')} {match.get('gate', '')}".strip()
    status = match.get("status", "Unknown")

    parts = [
        f"Flight {match.get('flight_number')} ({match.get('airline', '')}) to {destination}",
        f"Departs {departs}",
    ]
    if gate_info:
        parts.append(gate_info)
    parts.append(f"Status: {status}")
    return ". ".join(parts) + "."


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
    return Command(update={
        "should_end": True,
        "messages": [ToolMessage(content="call ended", tool_call_id=tool_call_id)],
    })


# ---------------------------------------------------------------------------
# User memory
# ---------------------------------------------------------------------------
# There are no recall/save memory tools. Memory is handled by dedicated graph
# nodes (agent/graph.py): recall_memory surfaces relevant memories into the
# prompt each turn, and save_memory persists new facts after each reply.


# ---------------------------------------------------------------------------
# Flight change tracking tool
# ---------------------------------------------------------------------------

class TrackFlightInput(BaseModel):
    flight_number: str = Field(description="Flight number to track, e.g. 'UA 123' or 'SW 891'")
    phone_number: Optional[str] = Field(
        default=None,
        description="Phone number to call when the gate, status, or time changes. Defaults to the caller's current number.",
    )
    airport: str = Field(default=DEFAULT_AIRPORT, description="Airport IATA code, e.g. 'OAK'")


@tool(args_schema=TrackFlightInput)
def track_flight_changes(
    flight_number: str,
    phone_number: Optional[str] = None,
    airport: str = DEFAULT_AIRPORT,
    state: Annotated[dict, InjectedState] = {},
) -> str:
    """Register a flight for tracking and call the given phone number if the gate,
    status, or departure time changes. Confirm the flight number with the user before
    calling this. If no phone number is provided, the caller's current number is used."""
    from agent.flight_tracker import subscribe

    flight = flight_number.strip().upper().replace(" ", "")
    phone = phone_number or (state.get("user_id") if state else None)
    if not phone:
        return "I need a phone number to call when the flight changes. Which number should I use?"

    subscribe(airport.upper(), flight, phone)
    return (
        f"Done — I'm now tracking flight {flight}. "
        f"I'll call {phone} if the gate, status, or departure time changes."
    )


TOOLS = [
    find_poi,
    get_route,
    get_nodes,
    resolve_poi,
    find_nearest,
    set_nav_state,
    set_flight_number,
    # get_flight_status,  # enable when GateGetter server is running
    end_call,
    track_flight_changes,
    *MAP_TOOLS,
]

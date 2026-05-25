import json
import time
from typing import Annotated, List, Optional

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.config import DEFAULT_AIRPORT
from agent.timing import add_moss, add_supermemory
from agent.db import load_map_levels
from agent.map_engine import (
    dijkstra_multilevel,
    fmt_poi,
    format_route_speech,
    matching_poi_types,
    search_pois,
)
from agent.memory import get_last_location, get_last_flight, search_memories, update_flight, update_location
from agent.moss_indexing import index_moss_flights as moss_index_flights
from agent.moss_indexing import index_moss_pois as moss_index_pois
from agent.moss_search import search_moss_flights as moss_semantic_search_flights
from agent.moss_search import search_moss_pois as moss_semantic_search_pois


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


class SearchUserMemoryInput(BaseModel):
    query: str = Field(
        description="Natural language query describing what user preference or context to look up, "
                    "e.g. 'food preferences', 'payment cards and lounge access', 'mobility or accessibility needs'"
    )

class SearchMossPoisInput(BaseModel):
    query: str = Field(description="Semantic search query over the Oakland POI Moss index")


class SearchMossFlightsInput(BaseModel):
    query: str = Field(description="Semantic search query over the Oakland flights Moss index")


class IndexMossInput(BaseModel):
    airport_id: str = Field(default=DEFAULT_AIRPORT, description="Airport code to index, e.g. OAK")


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
    # Refresh start from state/Supermemory so the route always begins from the user's
    # actual current location, not a hallucinated or stale one from the LLM.
    print("[thinking]")
    user_id = state.get("user_id") if state else None
    state_loc = state.get("current_location") if state else None
    if state_loc and state_loc != start:
        start = state_loc
    elif user_id:
        t0 = time.time()
        fresh = get_last_location(user_id)
        add_supermemory(time.time() - t0)
        if fresh and fresh != start:
            start = fresh

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
    # Refresh source from Supermemory dynamic profile so detours always use the latest location.
    print("[thinking]")
    user_id = state.get("user_id") if state else None
    if user_id:
        t0 = time.time()
        fresh = get_last_location(user_id)
        add_supermemory(time.time() - t0)
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

    nav_update: dict = {"messages": [ToolMessage(content="nav state updated", tool_call_id=tool_call_id)]}
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
# Memory tool (Supermemory — personalization)
# ---------------------------------------------------------------------------

class SetFlightInput(BaseModel):
    flight_number: str = Field(description="The user's flight number, e.g. 'UA 2345' or 'SW 891'")


@tool
def set_flight_number(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    flight_number: str,
) -> Command:
    """Save the user's flight number to state and Supermemory so it persists across conversations."""
    print("[thinking with supermemory]")
    user_id = state.get("user_id")
    if user_id:
        t0 = time.time()
        update_flight(user_id, flight_number)
        add_supermemory(time.time() - t0)
    return Command(update={
        "flight_number": flight_number,
        "messages": [ToolMessage(content=f"Flight number set to {flight_number}", tool_call_id=tool_call_id)],
    })


@tool(args_schema=SearchUserMemoryInput)
def search_user_memory(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Look up persistent facts about this user: food preferences, payment cards, loyalty programs,
    accessibility needs, lifestyle habits, and location patterns. Call this before recommending a
    category of POI or when you want to personalize navigation for this user."""
    user_id = state.get("user_id")
    if not user_id:
        return "No user identity available — cannot retrieve personalized memories."
    print("[thinking with supermemory]")
    t0 = time.time()
    result = search_memories(user_id, query)
    add_supermemory(time.time() - t0)
    return result


@tool(args_schema=SearchMossPoisInput)
def search_moss_pois(query: str) -> str:
    """Semantic search only over the `oakland-pois` Moss index."""
    print("[thinking with moss]")
    t0 = time.time()
    result = moss_semantic_search_pois(query=query, top_k=7)
    add_moss(time.time() - t0)
    return json.dumps(result, ensure_ascii=True)


@tool(args_schema=SearchMossFlightsInput)
def search_moss_flights(query: str) -> str:
    """Semantic search only over the `oakland-flights` Moss index."""
    print("[thinking with moss]")
    t0 = time.time()
    result = moss_semantic_search_flights(query=query, top_k=7)
    add_moss(time.time() - t0)
    return json.dumps(result, ensure_ascii=True)


@tool(args_schema=IndexMossInput)
def index_moss_pois(airport_id: str = DEFAULT_AIRPORT) -> str:
    """Build or refresh the Moss POI index for the airport."""
    print("[thinking with moss]")
    return json.dumps(moss_index_pois(airport_id=airport_id), ensure_ascii=True)


@tool(args_schema=IndexMossInput)
def index_moss_flights(airport_id: str = DEFAULT_AIRPORT) -> str:
    """Build or refresh the Moss flight index for the airport."""
    print("[thinking with moss]")
    return json.dumps(moss_index_flights(airport_id=airport_id), ensure_ascii=True)


TOOLS = [
    find_poi,
    get_route,
    get_nodes,
    resolve_poi,
    find_nearest,
    set_nav_state,
    set_flight_number,
    search_user_memory,
    search_moss_pois,
    search_moss_flights,
    index_moss_pois,
    index_moss_flights,
]

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
from agent.db import load_map_levels
from agent.map_engine import (
    dijkstra_multilevel,
    fmt_poi,
    format_route_speech,
    matching_poi_types,
    search_pois,
)
from agent.memory import search_memories


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


# ---------------------------------------------------------------------------
# Navigation tools
# ---------------------------------------------------------------------------

@tool(args_schema=POIInput)
def find_poi(q: str, airport_id: str = DEFAULT_AIRPORT):
    """Fuzzy search for a single POI name. Returns a dict with an 'id' field to pass to get_route."""
    print(f"[TOOL] find_poi q={q!r}")
    t0 = time.time()

    levels = load_map_levels(airport_id)
    results = search_pois(levels, q)

    print(f"  -> {len(results)} candidates in {time.time() - t0:.2f}s")

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
    print(f"[TOOL] resolve_poi name={name!r} landmark={landmark!r} floor={floor!r}")
    t0 = time.time()

    levels = load_map_levels(airport_id)

    if floor:
        floor_lower = floor.lower()
        for c in candidates:
            if floor_lower in (c.get("level") or "").lower():
                print(f"  -> resolved by floor in {time.time() - t0:.2f}s")
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

    print(f"  -> resolved by landmark in {time.time() - t0:.2f}s")
    return {
        "found": True,
        "id": best_candidate["id"],
        "name": best_candidate["name"],
        "type": best_candidate.get("type", ""),
        "level": best_candidate.get("level", ""),
    }


@tool(args_schema=RouteInput)
def get_route(start: str, end: str, airport_id: str = DEFAULT_AIRPORT):
    """Get shortest path between two POIs. start and end must be POI ids returned by find_poi."""
    print(f"[TOOL] get_route start={start} end={end}")
    t0 = time.time()

    levels = load_map_levels(airport_id)
    result = dijkstra_multilevel(levels, start, end)

    if not result:
        print(f"  -> no route in {time.time() - t0:.2f}s")
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
    route_text = format_route_speech(raw)

    print(f"  -> route in {time.time() - t0:.2f}s")
    print(f"[ROUTE RAW] {json.dumps(raw, ensure_ascii=True)}")
    print(f"[ROUTE TEXT] {route_text}")

    return route_text


@tool(args_schema=GetNodesInput)
def get_nodes(airport_id: str = DEFAULT_AIRPORT, floor: Optional[str] = None):
    """List all POI nodes for an airport, optionally filtered by floor."""
    print(f"[TOOL] get_nodes floor={floor}")
    t0 = time.time()

    levels = load_map_levels(airport_id)
    nodes = []
    for lvl in levels:
        if floor and lvl["name"] != floor:
            continue
        nodes.extend(lvl["pois"])

    print(f"  -> {len(nodes)} nodes in {time.time() - t0:.2f}s")
    return nodes


@tool(args_schema=FindNearestInput)
def find_nearest(source: str, poi_type: str, airport_id: str = DEFAULT_AIRPORT):
    """Find the closest POI of a given type from a source location."""
    print(f"[TOOL] find_nearest source={source!r} type={poi_type!r}")
    t0 = time.time()

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

    print(f"  -> nearest in {time.time() - t0:.2f}s")
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
    final_destination: Optional[str] = None,
    current_location: Optional[str] = None,
) -> Command:
    """Update remembered navigation state.

    Call this when the user tells you a new destination or confirms they've moved to a new checkpoint.
    Pass POI ids (e.g. 'gate-J292'), not raw names. Pass only the field(s) that changed.
    """
    print(f"[TOOL] set_nav_state final={final_destination!r} current={current_location!r}")

    levels = load_map_levels(DEFAULT_AIRPORT)
    valid_ids = {p["id"] for lvl in levels for p in lvl["pois"]}

    errors = []
    if final_destination is not None and final_destination not in valid_ids:
        errors.append(f"'{final_destination}' is not a valid POI id — call find_poi first to get the id, then pass the 'id' field here.")
    if current_location is not None and current_location not in valid_ids:
        errors.append(f"'{current_location}' is not a valid POI id — call find_poi first to get the id, then pass the 'id' field here.")

    if errors:
        msg = "Error: " + " | ".join(errors)
        print(f"[TOOL] set_nav_state validation failed: {msg}")
        return Command(update={"messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]})

    update: dict = {"messages": [ToolMessage(content="nav state updated", tool_call_id=tool_call_id)]}
    if final_destination is not None:
        update["final_destination"] = final_destination
    if current_location is not None:
        update["current_location"] = current_location
    return Command(update=update)


# ---------------------------------------------------------------------------
# Memory tool (Supermemory — personalization)
# ---------------------------------------------------------------------------

@tool(args_schema=SearchUserMemoryInput)
def search_user_memory(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Look up persistent facts about this user: food preferences, payment cards, loyalty programs,
    accessibility needs, lifestyle habits, and location patterns. Call this before recommending a
    category of POI or when you want to personalize navigation for this user."""
    user_id = state.get("user_id")
    if not user_id:
        return "No user identity available — cannot retrieve personalized memories."
    print(f"[TOOL] search_user_memory query={query!r} user={user_id}")
    result = search_memories(user_id, query)
    print(f"[MEMORY] result:\n{result}")
    return result


TOOLS = [find_poi, get_route, get_nodes, resolve_poi, find_nearest, set_nav_state, search_user_memory]

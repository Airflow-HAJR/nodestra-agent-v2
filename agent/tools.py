import time
from typing import Annotated, List, Optional

import requests
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.config import BASE_URL, DEFAULT_AIRPORT, HTTP_TIMEOUT


# ================= SCHEMAS =================

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


# ================= HELPERS =================

def _short(text: str, n: int = 300) -> str:
    return text if len(text) <= n else text[:n] + f"... ({len(text)} chars)"


def _log_response(name: str, res: requests.Response, t0: float) -> None:
    print(f"  -> {res.status_code} in {time.time() - t0:.2f}s | {_short(res.text)}")


# ================= API TOOLS =================

@tool(args_schema=POIInput)
def find_poi(q: str, airport_id: str = DEFAULT_AIRPORT):
    """Fuzzy search for a single POI name. Returns a dict containing an 'id' field that should be passed to get_route."""
    print(f"[TOOL] find_poi q={q!r}")
    t0 = time.time()
    res = requests.get(
        f"{BASE_URL}/find-poi",
        params={"q": q, "airport_id": airport_id},
        timeout=HTTP_TIMEOUT,
    )
    _log_response("find_poi", res, t0)
    return res.json()


@tool(args_schema=RouteInput)
def get_route(start: str, end: str, airport_id: str = DEFAULT_AIRPORT):
    """Get shortest path between two POIs. start and end must be POI ids returned by find_poi."""
    print(f"[TOOL] get_route start={start} end={end}")
    t0 = time.time()
    res = requests.get(
        f"{BASE_URL}/route",
        params={"start_id": start, "end_id": end, "airport_id": airport_id},
        timeout=HTTP_TIMEOUT,
    )
    _log_response("get_route", res, t0)
    return res.json()


@tool(args_schema=GetNodesInput)
def get_nodes(airport_id: str = DEFAULT_AIRPORT, floor: Optional[str] = None):
    """List all POI nodes for an airport, optionally filtered by floor."""
    print(f"[TOOL] get_nodes floor={floor}")
    t0 = time.time()
    params: dict = {"airport_id": airport_id}
    if floor:
        params["floor"] = floor
    res = requests.get(f"{BASE_URL}/get-nodes", params=params, timeout=HTTP_TIMEOUT)
    _log_response("get_nodes", res, t0)
    return res.json()


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
    payload: dict = {"name": name, "candidates": candidates, "airport_id": airport_id}
    if landmark:
        payload["landmark"] = landmark
    if floor:
        payload["floor"] = floor
    res = requests.post(f"{BASE_URL}/resolve-poi", json=payload, timeout=HTTP_TIMEOUT)
    _log_response("resolve_poi", res, t0)
    return res.json()


@tool(args_schema=FindNearestInput)
def find_nearest(source: str, poi_type: str, airport_id: str = DEFAULT_AIRPORT):
    """Find the closest POI of a given type from a source location."""
    print(f"[TOOL] find_nearest source={source!r} type={poi_type!r}")
    t0 = time.time()
    res = requests.post(
        f"{BASE_URL}/find-nearest",
        json={"source_id": source, "poi_type": poi_type, "airport_id": airport_id},
        timeout=HTTP_TIMEOUT,
    )
    _log_response("find_nearest", res, t0)
    return res.json()


# ================= STATE TOOLS =================

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
    update: dict = {"messages": [ToolMessage(content="nav state updated", tool_call_id=tool_call_id)]}
    if final_destination is not None:
        update["final_destination"] = final_destination
    if current_location is not None:
        update["current_location"] = current_location
    return Command(update=update)


TOOLS = [find_poi, get_route, get_nodes, resolve_poi, find_nearest, set_nav_state]

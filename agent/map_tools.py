"""
Google Maps display tools for the airport agent.

Copy this file to agent/map_tools.py in nodestra-agent-v2.

Then:
1. Add MAP_TOOLS to the TOOLS list in agent/tools.py
2. Add bind_map_callback to agent/graph.py (see INTEGRATION NOTES below)
3. Add map action emission to server.py /web/stream handler (see INTEGRATION NOTES)
4. Add map tool prompting to agent/prompts.py (see PROMPTS NOTES)

INTEGRATION NOTES — graph.py:
-------------------------------
Import and add _map_callback_var + bind_map_callback to graph.py:

    from contextvars import ContextVar
    from contextlib import contextmanager

    _map_callback_var: ContextVar = ContextVar("map_callback", default=None)

    @contextmanager
    def bind_map_callback(callback):
        token = _map_callback_var.set(callback)
        try:
            yield
        finally:
            _map_callback_var.reset(token)

Then in map_tools.py, import _map_callback_var from agent.graph instead of defining it locally.

INTEGRATION NOTES — server.py /web/stream handler:
----------------------------------------------------
In the _agent_worker per-turn loop, wrap the _graph_reply call with map binding:

    from agent.map_tools import bind_map_callback

    map_action_q: asyncio.Queue[dict] = asyncio.Queue()

    def map_cb(action: dict) -> None:
        loop.call_soon_threadsafe(map_action_q.put_nowait, action)

    # Wrap graph call:
    with bind_speak_early_callback(early_speak_cb):
        with bind_sentence_callback(sentence_cb):
            with bind_map_callback(map_cb):
                reply = await loop.run_in_executor(...)

    # After drain_task, emit map actions:
    while not map_action_q.empty():
        action = map_action_q.get_nowait()
        await ws.send_json({"type": "map_action", "action": action})

PROMPTS NOTES — add to agent/prompts.py system prompt:
-------------------------------------------------------
When you give navigation directions to the user:
- After calling get_route, ALWAYS call show_map_directions so the user can see the route.
- After calling find_poi, call show_map_destination to show the location.
- If the user has GPS (user_lat/user_lng in state), pass those to show_map_directions.
- When navigation is complete or the user confirms arrival, call clear_map.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Hardcoded GPS coordinates for OAK airport POIs (testing only)
# Replace with Supabase data once GPS coordinates are added to the maps table.
# Approximate coordinates based on OAK airport terminal layout.
# ---------------------------------------------------------------------------

OAK_GPS: dict[str, dict] = {
    # ── Security ────────────────────────────────────────────────────────────
    "security":             {"lat": 37.7203, "lng": -122.2225, "name": "Security Checkpoint"},
    "security-checkpoint":  {"lat": 37.7203, "lng": -122.2225, "name": "Security Checkpoint"},
    "tsa":                  {"lat": 37.7203, "lng": -122.2225, "name": "TSA Security"},
    "tsa-precheck":         {"lat": 37.7204, "lng": -122.2224, "name": "TSA PreCheck"},

    # ── Terminal 1 gates (Southwest / domestic) ──────────────────────────────
    "gate-1":   {"lat": 37.7205, "lng": -122.2228, "name": "Gate 1"},
    "gate-2":   {"lat": 37.7206, "lng": -122.2226, "name": "Gate 2"},
    "gate-3":   {"lat": 37.7207, "lng": -122.2224, "name": "Gate 3"},
    "gate-4":   {"lat": 37.7208, "lng": -122.2222, "name": "Gate 4"},
    "gate-5":   {"lat": 37.7209, "lng": -122.2220, "name": "Gate 5"},
    "gate-6":   {"lat": 37.7210, "lng": -122.2218, "name": "Gate 6"},
    "gate-7":   {"lat": 37.7211, "lng": -122.2216, "name": "Gate 7"},
    "gate-8":   {"lat": 37.7212, "lng": -122.2214, "name": "Gate 8"},
    "gate-9":   {"lat": 37.7213, "lng": -122.2212, "name": "Gate 9"},
    "gate-10":  {"lat": 37.7214, "lng": -122.2210, "name": "Gate 10"},
    "gate-11":  {"lat": 37.7215, "lng": -122.2208, "name": "Gate 11"},
    "gate-12":  {"lat": 37.7216, "lng": -122.2206, "name": "Gate 12"},

    # ── Terminal 2 gates (United / international) ──────────────────────────
    "gate-20":  {"lat": 37.7219, "lng": -122.2195, "name": "Gate 20"},
    "gate-21":  {"lat": 37.7220, "lng": -122.2193, "name": "Gate 21"},
    "gate-22":  {"lat": 37.7221, "lng": -122.2191, "name": "Gate 22"},
    "gate-23":  {"lat": 37.7222, "lng": -122.2189, "name": "Gate 23"},
    "gate-24":  {"lat": 37.7223, "lng": -122.2187, "name": "Gate 24"},
    "gate-25":  {"lat": 37.7224, "lng": -122.2185, "name": "Gate 25"},
    "gate-26":  {"lat": 37.7225, "lng": -122.2183, "name": "Gate 26"},
    "gate-27":  {"lat": 37.7226, "lng": -122.2181, "name": "Gate 27"},
    "gate-28":  {"lat": 37.7227, "lng": -122.2179, "name": "Gate 28"},

    # ── Lounges ─────────────────────────────────────────────────────────────
    "escape-lounge":    {"lat": 37.7220, "lng": -122.2190, "name": "Escape Lounge"},
    "lounge":           {"lat": 37.7220, "lng": -122.2190, "name": "Escape Lounge"},
    "lounge-myas":      {"lat": 37.7220, "lng": -122.2190, "name": "Escape Lounge"},

    # ── Food & drink ────────────────────────────────────────────────────────
    "starbucks":        {"lat": 37.7210, "lng": -122.2215, "name": "Starbucks"},
    "coffee":           {"lat": 37.7210, "lng": -122.2215, "name": "Coffee"},
    "food-court":       {"lat": 37.7208, "lng": -122.2220, "name": "Food Court"},
    "restaurant":       {"lat": 37.7209, "lng": -122.2218, "name": "Restaurant"},
    "bar":              {"lat": 37.7211, "lng": -122.2213, "name": "Bar"},

    # ── Facilities ──────────────────────────────────────────────────────────
    "restroom":         {"lat": 37.7209, "lng": -122.2221, "name": "Restroom"},
    "bathroom":         {"lat": 37.7209, "lng": -122.2221, "name": "Restroom"},
    "atm":              {"lat": 37.7207, "lng": -122.2223, "name": "ATM"},
    "info":             {"lat": 37.7204, "lng": -122.2226, "name": "Information Desk"},
    "information":      {"lat": 37.7204, "lng": -122.2226, "name": "Information Desk"},
    "charging":         {"lat": 37.7212, "lng": -122.2212, "name": "Charging Station"},
    "elevator":         {"lat": 37.7206, "lng": -122.2224, "name": "Elevator"},
    "escalator":        {"lat": 37.7207, "lng": -122.2222, "name": "Escalator"},

    # ── Arrivals / departures ────────────────────────────────────────────────
    "baggage-claim":    {"lat": 37.7200, "lng": -122.2230, "name": "Baggage Claim"},
    "baggage":          {"lat": 37.7200, "lng": -122.2230, "name": "Baggage Claim"},
    "arrivals":         {"lat": 37.7201, "lng": -122.2229, "name": "Arrivals"},
    "departures":       {"lat": 37.7202, "lng": -122.2227, "name": "Departures"},

    # ── Ground transport ─────────────────────────────────────────────────────
    "rideshare":        {"lat": 37.7198, "lng": -122.2232, "name": "Rideshare Pickup"},
    "taxi":             {"lat": 37.7198, "lng": -122.2232, "name": "Taxi Stand"},
    "bart":             {"lat": 37.7196, "lng": -122.2235, "name": "BART Station"},
    "rental-car":       {"lat": 37.7195, "lng": -122.2240, "name": "Rental Car Center"},
}


# ---------------------------------------------------------------------------
# Context variable for emitting map actions from tools
# ---------------------------------------------------------------------------

_map_callback_var: ContextVar[Optional[Callable[[dict], None]]] = ContextVar(
    "map_callback", default=None
)


@contextmanager
def bind_map_callback(callback: Callable[[dict], None]) -> Iterator[None]:
    """Context manager that makes map tools emit actions via callback."""
    token = _map_callback_var.set(callback)
    try:
        yield
    finally:
        _map_callback_var.reset(token)


def _emit(action: dict) -> None:
    cb = _map_callback_var.get()
    if cb:
        cb(action)


def _lookup_gps(poi_id: str) -> Optional[dict]:
    """Find GPS coords for a POI id: exact match, then prefix/substring match."""
    key = poi_id.lower().strip()
    if key in OAK_GPS:
        return OAK_GPS[key]
    for k, v in OAK_GPS.items():
        if key.startswith(k) or k.startswith(key):
            return v
    for k, v in OAK_GPS.items():
        if k in key or key in k:
            return v
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ShowDestinationInput(BaseModel):
    poi_id: str = Field(
        description="POI id from find_poi (e.g. 'gate-5', 'escape-lounge', 'security')"
    )
    destination_name: str = Field(
        description="Human-readable name to display on the map (e.g. 'Gate 5', 'Escape Lounge')"
    )


@tool(args_schema=ShowDestinationInput)
def show_map_destination(poi_id: str, destination_name: str) -> str:
    """Show a destination pin on the user's map screen.

    Call this after find_poi when the user asks where something is — it puts a
    visual pin on the map so the user can see the location at a glance.
    Use the poi_id returned by find_poi as the first argument."""
    print("[map] show_destination")
    gps = _lookup_gps(poi_id)
    if not gps:
        return f"GPS coordinates not available for {destination_name} yet."
    _emit({
        "type": "show_destination",
        "destination": {"name": destination_name, "lat": gps["lat"], "lng": gps["lng"]},
    })
    return f"Map updated — showing {destination_name}."


class ShowDirectionsInput(BaseModel):
    destination_poi_id: str = Field(
        description="Destination POI id from find_poi"
    )
    destination_name: str = Field(
        description="Human-readable destination name (e.g. 'Gate 5')"
    )
    user_lat: Optional[float] = Field(
        default=None,
        description="User's current GPS latitude — pass if available from state",
    )
    user_lng: Optional[float] = Field(
        default=None,
        description="User's current GPS longitude — pass if available from state",
    )


@tool(args_schema=ShowDirectionsInput)
def show_map_directions(
    destination_poi_id: str,
    destination_name: str,
    user_lat: Optional[float] = None,
    user_lng: Optional[float] = None,
) -> str:
    """Show walking directions on the user's map screen.

    Call this after get_route when giving navigation instructions — it shows
    the route visually so the user can follow along on the map.
    Pass user_lat/user_lng if the user's GPS location is known."""
    print("[map] show_directions")
    gps = _lookup_gps(destination_poi_id)
    if not gps:
        return f"GPS coordinates not available for {destination_name} yet."

    action: dict = {
        "type": "show_directions",
        "destination": {"name": destination_name, "lat": gps["lat"], "lng": gps["lng"]},
    }
    if user_lat is not None and user_lng is not None:
        action["origin"] = {"lat": user_lat, "lng": user_lng}

    _emit(action)
    return f"Directions shown on map to {destination_name}."


class ShowRouteInput(BaseModel):
    stops: list[str] = Field(
        description="Ordered list of POI ids along the route (from get_route stops)"
    )
    stop_names: list[str] = Field(
        description="Human-readable names for each stop (same order as stops)"
    )


@tool(args_schema=ShowRouteInput)
def show_map_route(stops: list[str], stop_names: list[str]) -> str:
    """Show a multi-stop route on the user's map screen.

    Call this for complex routes with several named waypoints — it draws the
    full path so the user can see each checkpoint they'll pass through."""
    print("[map] show_route")
    resolved: list[dict] = []
    for poi_id, name in zip(stops, stop_names):
        gps = _lookup_gps(poi_id)
        if gps:
            resolved.append({"name": name, "lat": gps["lat"], "lng": gps["lng"]})

    if len(resolved) < 2:
        return "Not enough GPS coordinates available to draw route."

    _emit({"type": "show_route", "stops": resolved})
    return f"Route shown on map ({len(resolved)} stops)."


@tool
def clear_map() -> str:
    """Clear the map display.

    Call this when navigation is complete, the user has arrived at their
    destination, or the user explicitly asks to close the map."""
    print("[map] clear")
    _emit({"type": "clear"})
    return "Map cleared."


MAP_TOOLS = [show_map_destination, show_map_directions, show_map_route, clear_map]

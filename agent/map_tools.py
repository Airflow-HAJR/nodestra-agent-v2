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
from typing import Annotated, Callable, Iterator, Optional

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.config import DEFAULT_AIRPORT
from agent.db import load_map_levels

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


# ---------------------------------------------------------------------------
# Real POI ids (e.g. "gate-7WH7") come from the live Supabase map graph, whose
# POIs only carry a normalized (x, y) floor-plan position — not GPS. This
# bounding box calibrates that [0,1] plan grid onto real-world lat/lng so
# real POIs (not just the OAK_GPS demo set above) can be pinned on the map.
# (0, 0) = plan's top-left corner, (1, 1) = bottom-right.
# ---------------------------------------------------------------------------

_AIRPORT_GPS_BOUNDS: dict[str, dict] = {
    "OAK": {"nw_lat": 37.7145, "nw_lng": -122.2230, "se_lat": 37.7105, "se_lng": -122.2160},
}


def _plan_to_latlng(airport_id: str, x: float, y: float) -> Optional[dict]:
    bounds = _AIRPORT_GPS_BOUNDS.get(airport_id)
    if not bounds:
        return None
    return {
        "lat": bounds["nw_lat"] + y * (bounds["se_lat"] - bounds["nw_lat"]),
        "lng": bounds["nw_lng"] + x * (bounds["se_lng"] - bounds["nw_lng"]),
    }


def _lookup_gps(poi_id: str, airport_id: str = DEFAULT_AIRPORT) -> Optional[dict]:
    """Resolve GPS coords for a POI id.

    Tries the real map graph first (converting the POI's indoor floor-plan
    x/y into lat/lng), then falls back to the OAK_GPS demo dict — exact
    match, then prefix/substring — for the old hardcoded test ids."""
    try:
        for lvl in load_map_levels(airport_id):
            for p in lvl["pois"]:
                if p["id"] == poi_id:
                    latlng = _plan_to_latlng(airport_id, p["x"], p["y"])
                    if latlng:
                        return {**latlng, "name": p.get("name", poi_id)}
    except Exception:
        pass  # Supabase not configured / unreachable — fall through to demo dict

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


class ShowOptionsInput(BaseModel):
    poi_ids: list[str] = Field(description="POI ids of the places to pin, in the order you'll talk about them")
    names: list[str] = Field(description="Display name for each place (same order as poi_ids)")
    notes: Optional[list[str]] = Field(
        default=None,
        description="Optional one-line caption per place (same order), e.g. '3 min · halal-certified'",
    )


@tool(args_schema=ShowOptionsInput)
def show_map_options(poi_ids: list[str], names: list[str], notes: Optional[list[str]] = None) -> str:
    """Pin several candidate places on the user's map at once — use this when
    you're presenting a choice ("here are three places to eat"), not a route.

    Unlike show_map_destination (one pin) or show_map_trajectory (a path), this
    draws no line: the places aren't stops on a walk, they're alternatives.
    Call it once with all the options, then talk through them in the same
    order. If you later narrow to one, call navigate() for that one."""
    print("[map] show_options")
    resolved: list[dict] = []
    for i, poi_id in enumerate(poi_ids):
        gps = _lookup_gps(poi_id)
        if not gps:
            continue
        resolved.append({
            "poiId": poi_id,
            "name": names[i] if i < len(names) else gps.get("name", poi_id),
            "lat": gps["lat"],
            "lng": gps["lng"],
            "note": (notes[i] if notes and i < len(notes) else None),
        })

    if not resolved:
        return "No GPS coordinates available for any of those places."

    _emit({"type": "show_options", "options": resolved})
    return f"Map updated — showing {len(resolved)} options: " + ", ".join(o["name"] for o in resolved) + "."


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


# ---------------------------------------------------------------------------
# Multi-level trajectory tools
# ---------------------------------------------------------------------------

def _resolve_trajectory_segments(raw: dict) -> list[dict]:
    """Resolve GPS for every stop in raw['segments'] (from get_route), tagging
    the one that's the elevator/escalator/stairs used to leave that floor.

    Each resolved stop keeps its original ("index") position within the
    segment's stop list, even though stops with no GPS get filtered out —
    that original index is what active_stop_index refers to, so it must stay
    stable regardless of which stops happen to drop out for missing GPS."""
    out_segments: list[dict] = []
    for seg in raw.get("segments", []):
        portal = seg.get("portal_out")
        portal_id = portal["id"] if portal else None
        stops: list[dict] = []
        for i, s in enumerate(seg.get("stops", [])):
            gps = _lookup_gps(s["id"])
            if not gps:
                continue
            stops.append({
                "id": s["id"],
                "name": s["name"],
                "lat": gps["lat"],
                "lng": gps["lng"],
                "kind": s.get("type") or "other",
                "isPortal": portal_id is not None and s["id"] == portal_id,
                "index": i,
            })
        out_segments.append({
            "levelName": seg.get("level_name", ""),
            "stops": stops,
            "portalOut": next((s for s in stops if s["isPortal"]), None),
        })
    return out_segments


def _emit_trajectory(raw: dict, route_id: str, active_seg_idx: int, active_stop_idx: int) -> bool:
    """Build and emit a show_trajectory map action. Returns False (and emits
    nothing) if GPS coordinates couldn't be resolved for the endpoints."""
    segments = _resolve_trajectory_segments(raw)
    first_stops = segments[0]["stops"] if segments else []
    last_stops = segments[-1]["stops"] if segments else []
    if not first_stops or not last_stops:
        return False

    _emit({
        "type": "show_trajectory",
        "routeId": route_id,
        "origin": {"name": first_stops[0]["name"], "lat": first_stops[0]["lat"], "lng": first_stops[0]["lng"]},
        "destination": {"name": last_stops[-1]["name"], "lat": last_stops[-1]["lat"], "lng": last_stops[-1]["lng"]},
        "segments": segments,
        "activeSegmentIndex": active_seg_idx,
        "activeStopIndex": active_stop_idx,
        "etaMinutes": raw.get("estimated_minutes"),
    })
    return True


def _emit_checkpoint_prompt(raw: dict, route_id: str, seg_idx: int, stop_idx: int) -> dict | None:
    """Emit a checkpoint_prompt for the given (segment, stop), if it exists.
    Returns the target stop dict (name/id) so callers can build a message,
    or None if there's no such checkpoint (e.g. already at the destination)."""
    stop = _current_checkpoint_stop(raw, seg_idx, stop_idx)
    if not stop:
        return None
    gps = _lookup_gps(stop["id"])
    _emit({
        "type": "checkpoint_prompt",
        "routeId": route_id,
        "segmentIndex": seg_idx,
        "stopIndex": stop_idx,
        "poiName": stop["name"],
        "promptText": "",
        "gpsTarget": {"lat": gps["lat"], "lng": gps["lng"]} if gps else None,
    })
    return stop


@tool
def show_map_trajectory(state: Annotated[dict, InjectedState]) -> str:
    """Show the full planned route on the user's map — call this after
    EVERY get_route, single-level or multi-level. It replaces
    show_map_directions/show_map_route; don't call those for the same route.

    Unlike a flat two-point line, this shows the user's current location,
    every named POI along the way as a dot, and the final
    destination as a pin — so the user can see the whole path they're
    walking, not just the start and end. This call also automatically puts
    a "Made it to __ / Need help" confirmation button on the user's screen
    for the first checkpoint — you don't need a separate tool call for that.

    The tool's return value tells you the next checkpoint's name. ALWAYS use
    that exact name when you tell the user where to go (e.g. "Head toward the
    coffee shop — it's highlighted on your map now.") — the highlighted dot
    is captioned with that same name, so the words and the screen match.

    If the route crosses more than one level, the map shows only the
    CURRENT floor. It will NOT show later floors yet — the user hasn't
    gotten there. Once the user confirms they reached the current
    checkpoint (by voice, or a system note that they tapped the button),
    call advance_checkpoint — the map reveals the next floor's leg
    automatically if that checkpoint was the floor-changing portal.
    """
    print("[map] show_trajectory")
    raw = state.get("last_route")
    if not raw or not raw.get("segments"):
        return "No route is available yet — call get_route first."

    route_id = state.get("route_id") or ""
    seg_idx = state.get("active_segment_index") or 0
    stop_idx = state.get("active_stop_index") or 1
    if not _emit_trajectory(raw, route_id, seg_idx, stop_idx):
        return "GPS coordinates not available for this route yet."

    n = len(raw["segments"])
    next_stop = _emit_checkpoint_prompt(raw, route_id, seg_idx, stop_idx)
    checkpoint_note = (
        f" Next checkpoint: {next_stop['name']} — highlighted on the map, captioned with that name."
        if next_stop else ""
    )
    return f"Trajectory shown on map ({n} floor{'s' if n != 1 else ''}).{checkpoint_note}"


def _current_checkpoint_stop(raw: dict, segment_idx: int, stop_idx: int) -> Optional[dict]:
    segments = raw.get("segments") or []
    if segment_idx >= len(segments):
        return None
    stops = segments[segment_idx].get("stops") or []
    if stop_idx >= len(stops):
        return None
    return stops[stop_idx]


class RequestCheckpointInput(BaseModel):
    prompt_text: str = Field(
        description="Exactly what to say to the user to ask them to confirm "
                     "they've reached the current checkpoint. Only needed if "
                     "you want to re-ask or re-emphasize — the button is "
                     "already on their screen automatically."
    )


@tool(args_schema=RequestCheckpointInput)
def request_checkpoint_confirmation(prompt_text: str, state: Annotated[dict, InjectedState]) -> str:
    """Re-ask the user to confirm they've reached the current checkpoint.

    OPTIONAL — show_map_trajectory and advance_checkpoint already put the
    confirmation button on screen automatically every time there's an
    active checkpoint. Only call this if you want to re-emphasize the ask
    out loud (e.g. the user seems confused, or asked you to repeat it).
    Say prompt_text yourself as your next reply.
    """
    print("[map] checkpoint_prompt (manual re-ask)")
    raw = state.get("last_route")
    seg_idx = state.get("active_segment_index") or 0
    stop_idx = state.get("active_stop_index") or 1
    route_id = state.get("route_id") or ""
    if not raw or not raw.get("segments"):
        return "No active route to request a checkpoint on."

    stop = _emit_checkpoint_prompt(raw, route_id, seg_idx, stop_idx)
    if not stop:
        return "There's no further checkpoint on this route — the user has reached the destination."
    return f"Checkpoint confirmation re-shown for {stop['name']} — now say prompt_text to the user."


@tool
def advance_checkpoint(
    reason: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
) -> Command:
    """Mark the current checkpoint reached and move on to the next one.

    reason: one short phrase for why you're advancing now, e.g. 'user
    confirmed by voice' or 'user tapped the checkpoint button'. Logged
    only, not shown to the user.

    Only call this after the user has actually confirmed they reached the
    checkpoint you last asked about — either they said so out loud, or a
    system note told you they tapped the on-screen checkpoint button.

    Before calling this, check CURRENT_USER_LOCATION (if available in
    context) against the checkpoint's coordinates from the route. If they're
    implausibly far apart for someone who just said they arrived, don't call
    this — ask the user to double check out loud instead (e.g. "Hmm, your
    location's showing you might still be near security — are you sure
    you're there?"). GPS drifts indoors, so use judgment: a mismatch of a
    few dozen meters is normal, hundreds of meters is not.

    If the checkpoint just confirmed was an elevator/escalator/stairs that
    changes floors, this also flips the map to the next floor's leg. This
    also automatically puts the confirmation button on screen for the NEW
    current checkpoint (no separate tool call needed) — the tool's return
    value tells you its name so you can mention it. It also
    updates the user's remembered current_location itself using the
    checkpoint's real POI id — you do NOT need to (and should not) call
    set_nav_state for this; you don't actually have that id, only the name.
    If it was the final stop (the destination itself), don't call this —
    the trip is over; continue narrating arrival and call clear_map when
    appropriate.
    """
    print(f"[map] advance_checkpoint ({reason})")
    raw = state.get("last_route")
    seg_idx = state.get("active_segment_index") or 0
    stop_idx = state.get("active_stop_index") or 1
    route_id = state.get("route_id") or ""

    if not raw or not raw.get("segments"):
        msg = ToolMessage(content="No active route to advance.", tool_call_id=tool_call_id)
        return Command(update={"messages": [msg]})

    segments = raw["segments"]
    seg = segments[seg_idx] if seg_idx < len(segments) else None
    if seg is None or stop_idx >= len(seg.get("stops", [])):
        msg = ToolMessage(
            content="Already at the final checkpoint — nothing further to advance to.",
            tool_call_id=tool_call_id,
        )
        return Command(update={"messages": [msg]})

    # The checkpoint just confirmed — its real POI id becomes the new
    # current_location, set here (not by the agent, which only ever sees
    # this stop's name, never its id) to avoid it having to guess.
    confirmed_stop = seg["stops"][stop_idx]
    confirmed_location_update = {"current_location": confirmed_stop["id"]}

    is_last_stop_of_segment = stop_idx == len(seg["stops"]) - 1
    is_last_segment = seg_idx == len(segments) - 1

    if is_last_stop_of_segment and not is_last_segment:
        # That checkpoint was the portal — flip the map to the next floor
        # and immediately surface the button for the new floor's first stop.
        next_seg_idx = seg_idx + 1
        _emit_trajectory(raw, route_id, next_seg_idx, 1)
        _emit({
            "type": "checkpoint_resolved",
            "routeId": route_id,
            "segmentIndex": seg_idx,
            "stopIndex": stop_idx,
            "nextSegmentIndex": next_seg_idx,
        })
        next_stop = _emit_checkpoint_prompt(raw, route_id, next_seg_idx, 1)
        note = f" Next checkpoint: {next_stop['name']} — highlighted on the new floor's map." if next_stop else ""
        msg = ToolMessage(
            content=f"Advanced to floor {next_seg_idx + 1} of {len(segments)}.{note}",
            tool_call_id=tool_call_id,
        )
        return Command(update={
            "messages": [msg],
            "active_segment_index": next_seg_idx,
            "active_stop_index": 1,
            **confirmed_location_update,
        })

    if is_last_stop_of_segment and is_last_segment:
        msg = ToolMessage(content="User has reached the final destination.", tool_call_id=tool_call_id)
        return Command(update={"messages": [msg], **confirmed_location_update})

    # More stops remain on this same floor — every stop is already visible
    # as a dot (show_map_trajectory drew the whole segment up front), so no
    # map redraw is needed beyond re-highlighting the new active stop, and
    # the confirmation button is re-shown for it automatically.
    next_stop_idx = stop_idx + 1
    _emit_trajectory(raw, route_id, seg_idx, next_stop_idx)
    _emit({
        "type": "checkpoint_resolved",
        "routeId": route_id,
        "segmentIndex": seg_idx,
        "stopIndex": stop_idx,
        "nextSegmentIndex": seg_idx,
    })
    next_stop = _emit_checkpoint_prompt(raw, route_id, seg_idx, next_stop_idx)
    note = f" Next checkpoint: {next_stop['name']} — highlighted on the map, captioned with that name." if next_stop else ""
    msg = ToolMessage(
        content=f"Advanced to checkpoint {next_stop_idx + 1} of {len(seg['stops'])} on this floor.{note}",
        tool_call_id=tool_call_id,
    )
    return Command(update={"messages": [msg], "active_stop_index": next_stop_idx, **confirmed_location_update})


@tool
def clear_map() -> str:
    """Clear the map display.

    Call this when navigation is complete, the user has arrived at their
    destination, or the user explicitly asks to close the map."""
    print("[map] clear")
    _emit({"type": "clear"})
    return "Map cleared."


MAP_TOOLS = [
    show_map_destination,
    show_map_options,
    show_map_directions,
    show_map_route,
    show_map_trajectory,
    request_checkpoint_confirmation,
    advance_checkpoint,
    clear_map,
]

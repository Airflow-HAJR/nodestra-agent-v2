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
from agent.map_tools import MAP_TOOLS, _emit_checkpoint_prompt, _emit_trajectory


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
# POI knowledge (retrieval) tools
#
# find_nearest tells you which place is closest; these tell you what the place
# actually is. Both read from agent/poi_rag.py, which embeds a corpus of venue
# write-ups and returns the passages that matched — the agent answers from
# those passages, and says so when nothing matched, rather than from its own
# impression of a restaurant name.
# ---------------------------------------------------------------------------

# Memory categories that can disqualify a venue rather than merely colour the
# recommendation. "travel" (favorite airline) and "personal" don't belong here —
# they don't make a restaurant wrong, so checking them against a menu would just
# burn a turn.
_CONSTRAINING_CATEGORIES = {"dietary", "accessibility", "preference", "payment"}


def _format_constraint(m: dict) -> str:
    meta = m.get("metadata") or {}
    content = (m.get("content") or "").strip()
    if isinstance(meta, dict) and meta.get("source"):
        where = " at " + str(meta["airport"]) if meta.get("airport") else ""
        when = " on " + str(meta["date"]) if meta.get("date") else ""
        return f"{content} — learned {meta['source']}{where}{when}"
    return content


class SuggestPlacesInput(BaseModel):
    need: str = Field(
        description="What the user is after in their own words, e.g. 'hungry, something quick', "
                    "'sit-down dinner', 'coffee', 'a quiet place to work'"
    )
    poi_type: str = Field(
        default="restaurant",
        description="Category to rank by walking distance: 'restaurant'|'cafe'|'lounge'|'shop'|'bookstore'",
    )
    source: Optional[str] = Field(
        default=None,
        description="POI id to measure walking distance from. Omit to use the user's current_location.",
    )
    limit: int = Field(default=3, description="How many options to return (2-4 reads best out loud)")
    airport_id: str = Field(default=DEFAULT_AIRPORT)


@tool(args_schema=SuggestPlacesInput)
def suggest_places(
    need: str,
    state: Annotated[dict, InjectedState],
    poi_type: str = "restaurant",
    source: Optional[str] = None,
    limit: int = 3,
    airport_id: str = DEFAULT_AIRPORT,
) -> dict:
    """Find several nearby places of a category and describe each one.

    Use this — not find_nearest — whenever the user wants OPTIONS rather than
    one destination ("I'm hungry, what's nearby?", "anywhere to get coffee?").
    It ranks candidates by real walking distance from the user, attaches a
    retrieved description of each, AND pins them all on the user's map, so you
    do not need to call show_map_options yourself.

    Returns one entry per option with its poi_id — pass those ids to
    lookup_poi_info to check them against a specific requirement, and to
    navigate() once the user picks one. Describe each option out loud using
    only the 'about' text returned here.
    """
    print("[suggest_places]")
    from agent.map_tools import show_map_options
    from agent import poi_rag

    levels = load_map_levels(airport_id)
    src = source or state.get("current_location")
    if not src:
        return {
            "found": False,
            "message": "I don't know where the user is yet — ask them which gate or area they're near, then call this again.",
        }
    if not any(p["id"] == src for lvl in levels for p in lvl["pois"]):
        return {"found": False, "message": f"Source POI '{src}' not found — call find_poi to get a valid id."}

    matched_types = matching_poi_types(poi_type, levels)
    if not matched_types:
        return {"found": False, "message": f"No POIs of type '{poi_type}' at this airport."}

    scored: list[tuple[float, dict, str]] = []
    for lvl in levels:
        for p in lvl["pois"]:
            if p["id"] == src or (p.get("type") or "").lower().strip() not in matched_types:
                continue
            route = dijkstra_multilevel(levels, src, p["id"])
            if route is None:
                continue
            scored.append((route["distance"], p, lvl["name"]))

    if not scored:
        return {"found": False, "message": f"Nothing of type '{poi_type}' is reachable from there."}

    scored.sort(key=lambda t: t[0])
    limit = max(1, min(limit, 5))

    # Which of the nearby candidates to surface is half "does this match what
    # they asked for" and half "is it actually close". Relevance alone sends a
    # hungry traveler to the far end of the other terminal; distance alone
    # ignores what they said they wanted. Both are min-max normalized within
    # the candidate pool before being averaged, because raw cosine scores sit
    # in a narrow band and would otherwise be swamped by the distance term.
    pool = scored[: limit * 3]
    pool_ids = [p["id"] for _, p, _ in pool]
    ranked = poi_rag.search(need, poi_ids=pool_ids, top_k=len(pool) * 4)
    relevance: dict[str, float] = {}
    for hit in ranked:
        relevance[hit["poi_id"]] = max(relevance.get(hit["poi_id"], 0.0), hit["score"])

    def _normalize(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        lo, hi = min(values.values()), max(values.values())
        span = hi - lo
        return {k: (1.0 if span == 0 else (v - lo) / span) for k, v in values.items()}

    rel_n = _normalize(relevance)
    prox_n = _normalize({p["id"]: -d for d, p, _ in pool})  # negated: closer scores higher
    combined = sorted(
        pool_ids,
        key=lambda pid: 0.5 * rel_n.get(pid, 0.0) + 0.5 * prox_n.get(pid, 0.0),
        reverse=True,
    )[:limit]
    by_id = {p["id"]: (d, p, lvl) for d, p, lvl in pool}
    # Presented nearest-first even though relevance chose the set: the walking
    # time is the thing the user compares options on out loud.
    picked = sorted((by_id[pid] for pid in combined), key=lambda t: t[0]) or scored[:limit]

    options = []
    for distance, poi, level_name in picked:
        about = poi_rag.summarize_poi(poi["id"])
        options.append({
            "poi_id": poi["id"],
            "name": poi.get("name") or poi_type.capitalize(),
            "type": poi.get("type", ""),
            "level": level_name,
            "walk_minutes": max(1, round(distance * 10)),
            "about": about or "No description on file — say you don't have details for this one.",
        })

    show_map_options.invoke({
        "poi_ids": [o["poi_id"] for o in options],
        "names": [o["name"] for o in options],
        "notes": [f"{o['walk_minutes']} min walk" for o in options],
    })

    result: dict = {
        "found": True,
        "options": options,
        "map": "All options are now pinned on the user's map — no further map call needed.",
    }

    # A remembered constraint is only worth remembering if it changes what the
    # user is offered. Rather than hoping the agent connects "eats halal" to
    # three restaurants it has just been handed, the shortlist itself carries
    # the constraint back out with the ids to check it against — the retrieval
    # step becomes the obvious next move instead of an easily-skipped one.
    constraining = [
        m for m in (state.get("user_memories") or [])
        if (m.get("category") or "") in _CONSTRAINING_CATEGORIES
    ]
    if constraining:
        result["must_check_first"] = {
            "instruction": (
                "You already know things about this user that could rule some of these out. "
                "BEFORE you recommend one: call lookup_poi_info with poi_ids set to ALL of the ids "
                "above and a question phrased from the preferences below (e.g. 'is the meat halal "
                "certified?'). Then, in this order and out loud: (1) name where the preference came "
                "from, using the 'learned' detail below — 'last time you came through OAK you were "
                "checking whether places were halal, so I checked all three'; (2) give the verdict "
                "for EVERY option, the failures included, in one short sentence each; (3) recommend "
                "the one that passed. Never silently drop an option — the user should hear why it's "
                "out. Plain speech only, no lists or markdown."
            ),
            "preferences": [_format_constraint(m) for m in constraining],
            "poi_ids": [o["poi_id"] for o in options],
        }

    return result


class LookupPOIInfoInput(BaseModel):
    question: str = Field(
        description="The specific thing to check, e.g. 'is the meat halal certified?', "
                    "'vegan options', 'what time does it close?', 'is there alcohol?'"
    )
    poi_ids: Optional[List[str]] = Field(
        default=None,
        description="POI ids to check. Pass ALL the candidates when comparing them; "
                    "omit to search every venue on file.",
    )


@tool(args_schema=LookupPOIInfoInput)
def lookup_poi_info(question: str, poi_ids: Optional[List[str]] = None) -> dict:
    """Look up what's actually known about specific venues — menu, dietary and
    halal information, hours, price, alcohol, accessibility.

    This is the ONLY source you have for these details. Never answer a "does
    this place have X" question from your own knowledge of a restaurant or
    chain — call this, then answer from the passages it returns, and say
    plainly that you don't know when it returns nothing.

    Pass every candidate's poi_id at once when the user's requirement has to be
    checked against several places (a diet, an allergy, a closing time): each
    venue is retrieved separately, so every one of them comes back with its own
    evidence instead of the strongest match drowning out the others.
    """
    print(f"[lookup_poi_info] {question!r} over {poi_ids or 'all POIs'}")
    from agent import poi_rag

    hits = poi_rag.search(
        question,
        poi_ids=poi_ids,
        top_k=2 if poi_ids else 5,
        per_poi=bool(poi_ids),
    )
    if not hits:
        return {
            "found": False,
            "message": "Nothing on file about that. Tell the user you don't have the details rather than guessing.",
        }

    by_poi: dict[str, dict] = {}
    for h in hits:
        entry = by_poi.setdefault(h["poi_id"], {"poi_id": h["poi_id"], "name": h["name"], "evidence": []})
        entry["evidence"].append({"section": h.get("section", ""), "text": h["text"], "score": h["score"]})

    return {
        "found": True,
        "question": question,
        "results": list(by_poi.values()),
        "instruction": "Answer only from these passages. If a venue's passages don't settle the question, say so for that venue.",
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
    # Nothing to set — bail before loading the map. Guards against the agent
    # calling this as a no-op (e.g. reaching for a "save" tool it doesn't need);
    # the map load and state churn for an all-null call are pure waste.
    if final_destination is None and current_location is None:
        return Command(update={
            "messages": [ToolMessage(content="No navigation change to apply.", tool_call_id=tool_call_id)],
        })
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


# ---------------------------------------------------------------------------
# Composite navigation tool — replaces the find_poi → set_nav_state →
# get_route → show_map_trajectory chain with a single LLM call.
# ---------------------------------------------------------------------------

def _display_name(stop: dict) -> str:
    """Human-friendly name for a stop. Gate POIs carry a bare number as their
    name ("10", "8A"), which reads badly spoken aloud — turn those into
    "Gate 10". Everything else already has a descriptive name."""
    name = (stop.get("name") or "").strip()
    typ = (stop.get("type") or "").lower().strip()
    if typ == "gate" and name and not name.lower().startswith("gate"):
        return f"Gate {name}"
    return name


def _generate_checkpoint_scripts(raw: dict, dest_name: str) -> list[str]:
    """Pre-write the "head toward next stop" reply for every checkpoint in the route.

    Scripts are consumed in order — scripts[0] is the reply after the user confirms
    the first checkpoint, scripts[1] after the second, etc. The router in graph.py
    uses these to skip the second main-LLM call on each checkpoint confirmation turn.
    """
    scripts: list[str] = []
    segments = raw.get("segments", [])
    n_segs = len(segments)

    for seg_i, seg in enumerate(segments):
        stops = seg.get("stops", [])
        n_stops = len(stops)
        is_last_seg = seg_i == n_segs - 1

        for stop_i in range(1, n_stops):  # skip index 0 (origin / portal-in)
            is_last_stop = stop_i == n_stops - 1
            is_final = is_last_stop and is_last_seg
            is_portal = is_last_stop and not is_last_seg

            if is_final:
                scripts.append(
                    f"You've made it to {dest_name}! Great work. "
                    "Is there anything else I can help you with?"
                )
            elif is_portal:
                next_seg = segments[seg_i + 1]
                next_stops = next_seg.get("stops", [])
                next_cp = next_stops[1] if len(next_stops) > 1 else None
                if next_cp:
                    scripts.append(
                        f"You're through! We're now on {next_seg['level_name']}. "
                        f"Look for {_display_name(next_cp)} — it's highlighted on your map. "
                        "Tap the button when you get there."
                    )
                else:
                    scripts.append(
                        f"You're on the next level — keep heading toward {dest_name}."
                    )
            else:
                next_stop = stops[stop_i + 1]
                scripts.append(
                    f"Keep going — now look for {_display_name(next_stop)}, highlighted on your map. "
                    "Tap the button when you get there."
                )

    return scripts


class NavigateInput(BaseModel):
    destination: str = Field(
        description="Destination name or description (e.g. 'Gate 5', 'Escape Lounge', 'Security')"
    )
    start: Optional[str] = Field(
        default=None,
        description="Start location name — omit if the user's current_location is already set in state",
    )
    airport_id: str = Field(default=DEFAULT_AIRPORT)


@tool(args_schema=NavigateInput)
def navigate(
    destination: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    start: Optional[str] = None,
    airport_id: str = DEFAULT_AIRPORT,
) -> Command:
    """All-in-one navigation: find POIs, compute route, save nav state, and show map trajectory.

    Use this as the PRIMARY tool whenever the user wants directions or to go somewhere.
    Returns estimated walking time and first checkpoint name so you can brief the user.
    Fall back to the individual tools (find_poi, get_route, etc.) only for edge cases:
    explicit multi-floor disambiguation, detour routing from a mid-route POI, or
    nearest-type searches (use find_nearest for those).
    """
    print("[navigate]")
    levels = load_map_levels(airport_id)

    # Resolve destination
    dest_results = search_pois(levels, destination)
    if not dest_results:
        msg = ToolMessage(
            content=f"No location matching '{destination}' found. Try a different spelling or name.",
            tool_call_id=tool_call_id,
        )
        return Command(update={"messages": [msg]})

    best_score = dest_results[0]["_score"]
    dest_top = [r for r in dest_results if r["_score"] == best_score]
    unique_names = {r["name"].lower().strip() for r in dest_top}
    if len(dest_top) > 1 and len(unique_names) == 1:
        matches_text = "; ".join(
            f"{r['name']} ({r.get('level', 'unknown floor')})" for r in dest_top
        )
        msg = ToolMessage(
            content=(
                f"Found {len(dest_top)} places named '{dest_top[0]['name']}' on different floors: {matches_text}. "
                "Ask the user which floor they need, then call navigate again."
            ),
            tool_call_id=tool_call_id,
        )
        return Command(update={"messages": [msg]})

    dest_poi = dest_top[0]
    dest_id = dest_poi["id"]
    dest_display = _display_name(dest_poi)

    # Resolve start — prefer state, then the start arg if provided
    start_id = state.get("current_location")
    if start and not start_id:
        start_results = search_pois(levels, start)
        if start_results:
            start_id = start_results[0]["id"]

    if not start_id:
        msg = ToolMessage(
            content=(
                f"Destination set to {dest_display}. "
                "Ask the user where they are now, then call navigate again once you know."
            ),
            tool_call_id=tool_call_id,
        )
        return Command(update={"messages": [msg], "final_destination": dest_id})

    # Compute route
    result = dijkstra_multilevel(levels, start_id, dest_id)
    if not result:
        msg = ToolMessage(
            content=f"No route found to {dest_display}.",
            tool_call_id=tool_call_id,
        )
        return Command(update={"messages": [msg], "final_destination": dest_id})

    route_id = str(uuid.uuid4())
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

    # Emit trajectory + first checkpoint confirm button
    _emit_trajectory(raw, route_id, 0, 1)
    _emit_checkpoint_prompt(raw, route_id, 0, 1)

    eta = raw["estimated_minutes"]
    n_segs = len(raw["segments"])
    first_seg_stops = (raw["segments"][0].get("stops") or []) if raw["segments"] else []
    first_checkpoint = first_seg_stops[1] if len(first_seg_stops) > 1 else None
    checkpoint_note = (
        f" First checkpoint: {_display_name(first_checkpoint)} — highlighted on the map."
        if first_checkpoint else ""
    )
    content = (
        f"Route to {dest_display}: ~{eta} min, "
        f"{n_segs} floor level{'s' if n_segs != 1 else ''}. "
        f"Map updated with full trajectory.{checkpoint_note}"
    )

    existing_intents = list(state.get("active_intents") or [])
    return Command(update={
        "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)],
        "final_destination": dest_id,
        "current_location": start_id,
        "last_route": raw,
        "active_segment_index": 0,
        "active_stop_index": 1,
        "route_id": route_id,
        "active_intents": list({*existing_intents, "navigate"}),
        "checkpoint_scripts": _generate_checkpoint_scripts(raw, dest_display),
    })


TOOLS = [
    navigate,
    suggest_places,
    lookup_poi_info,
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

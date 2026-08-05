"""
Pure navigation algorithms — no I/O, no side effects.
All functions take pre-loaded `levels` (from db.load_map_levels).
"""
from __future__ import annotations

import heapq
import re
from collections import defaultdict

# ---------------------------------------------------------------------------
# Type priority (for picking the best POI when multiple share a waypoint)
# ---------------------------------------------------------------------------

_TYPE_PRIORITY = {
    "gate": 100, "security": 90, "elevator": 80, "escalator": 75,
    "stairs": 70, "restroom": 60, "restaurant": 50, "cafe": 50,
    "coffee": 50, "food": 50, "bar": 45, "lounge": 40, "shop": 30,
    "bookstore": 30, "atm": 20, "info": 20, "charging": 20,
    "baggage": 15, "telephone": 10, "aed": 5,
}

PORTAL_COST = 0.5  # extra weight for a floor transition

_PORTAL_TYPES = {"elevator", "escalator", "stairs"}


def _poi_priority(poi: dict) -> int:
    return _TYPE_PRIORITY.get((poi.get("type") or "").lower(), 25)


# ---------------------------------------------------------------------------
# POI search
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return 0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def _search_score(query: str, poi_name: str) -> float:
    """Levenshtein with a hard penalty for number mismatches — Gate 5 never matches Gate 6."""
    q_nums = set(re.findall(r"\d+", query))
    p_nums = set(re.findall(r"\d+", poi_name))
    if q_nums and q_nums != p_nums:
        return float("inf")
    return _levenshtein(query, poi_name)


def search_pois(levels: list[dict], query: str) -> list[dict]:
    """
    Search all POIs across all levels.
    Returns list sorted by match score (lower = better).
    Each result has extra keys: _score, _level_idx, _level_name.
    """
    scored = []
    for i, lvl in enumerate(levels):
        for p in lvl["pois"]:
            if not p.get("name"):
                continue
            score = _search_score(query, p["name"])
            if score < float("inf"):
                scored.append({**p, "_score": score, "_level_idx": i, "_level_name": lvl["name"]})
    scored.sort(key=lambda x: x["_score"])
    return scored


def fmt_poi(p: dict) -> dict:
    return {"id": p["id"], "name": p["name"], "type": p.get("type", ""), "level": p["_level_name"]}


# Common words travelers use that don't literally appear as a POI type. Maps a
# user word to the canonical type(s) it should resolve to. Edit-distance alone
# can't bridge these ("coffee" → "cafe" is 4 edits), so they're spelled out.
_TYPE_SYNONYMS: dict[str, set[str]] = {
    "coffee": {"cafe"},
    "cafe": {"cafe"},
    "espresso": {"cafe"},
    "food": {"restaurant", "cafe"},
    "eat": {"restaurant", "cafe"},
    "restaurant": {"restaurant"},
    "dining": {"restaurant"},
    "drink": {"restaurant", "cafe", "bar"},
    "bar": {"bar", "restaurant"},
    "bathroom": {"restroom"},
    "toilet": {"restroom"},
    "washroom": {"restroom"},
    "water": {"water-fountain"},
    "book": {"bookstore"},
    "books": {"bookstore"},
    "shopping": {"shop", "duty-free", "bookstore"},
    "store": {"shop", "duty-free"},
    "nursing": {"nursing-room"},
    "baby": {"nursing-room", "baby-changing-station"},
    "lounge": {"lounge"},
}


def matching_poi_types(query: str, levels: list[dict]) -> set[str]:
    """
    Resolve a user-provided type string to actual types in the map.
    Synonym table first (coffee→cafe, bathroom→restroom), then substring match,
    then Levenshtein for the remaining typos.
    """
    types = {
        (p.get("type") or "").lower().strip()
        for lvl in levels for p in lvl["pois"]
        if (p.get("type") or "").strip()
    }
    q = query.lower().strip()
    if not q or not types:
        return set()

    # Synonyms take priority — but only keep the ones that actually exist in this map.
    synonyms = _TYPE_SYNONYMS.get(q)
    if synonyms:
        present = synonyms & types
        if present:
            return present

    substring = {t for t in types if q in t or t in q}
    if substring:
        return substring

    scored = sorted((_levenshtein(q, t), t) for t in types)
    best_score = scored[0][0]
    if best_score <= max(2, len(q) // 2):
        return {t for s, t in scored if s == best_score}
    return set()


# ---------------------------------------------------------------------------
# Dijkstra (multi-level)
# ---------------------------------------------------------------------------

def dijkstra_multilevel(levels: list[dict], start_poi_id: str, end_poi_id: str) -> dict | None:
    """
    Shortest path between any two POIs across one or more levels.
    Portal POIs (elevator/escalator/stairs) add cross-level edges via linkedPortalIds.
    Returns {wp_path, poi_stops, distance, level_changes} or None if unreachable.
    """
    poi_map: dict[str, tuple[int, dict]] = {}
    wp_level: dict[str, int] = {}

    for i, lvl in enumerate(levels):
        for p in lvl["pois"]:
            poi_map[p["id"]] = (i, p)
        for w in lvl["waypoints"]:
            wp_level[w["id"]] = i

    start_entry = poi_map.get(start_poi_id)
    end_entry = poi_map.get(end_poi_id)
    if not start_entry or not end_entry:
        return None

    start_wp = start_entry[1].get("waypointId", "")
    end_wp = end_entry[1].get("waypointId", "")
    if not start_wp or not end_wp:
        return None

    adj: dict[str, list] = defaultdict(list)
    for lvl in levels:
        for e in lvl["edges"]:
            w = e.get("weight", 1)
            adj[e["from"]].append((e["to"], w))
            adj[e["to"]].append((e["from"], w))

    for _, (_, poi) in poi_map.items():
        src_wp = poi.get("waypointId", "")
        if not src_wp:
            continue
        for linked_id in poi.get("linkedPortalIds", []):
            if not linked_id:
                continue
            linked_entry = poi_map.get(linked_id)
            if not linked_entry:
                continue
            dst_wp = linked_entry[1].get("waypointId", "")
            if dst_wp:
                adj[src_wp].append((dst_wp, PORTAL_COST))

    dist: dict[str, float] = {start_wp: 0.0}
    prev: dict[str, str] = {}
    pq = [(0.0, start_wp)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        if u == end_wp:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if end_wp not in prev and start_wp != end_wp:
        return None

    wp_path, cur = [], end_wp
    while cur in prev:
        wp_path.append(cur)
        cur = prev[cur]
    wp_path.append(start_wp)
    wp_path.reverse()

    wp_to_best: dict[str, dict] = {}
    for _, (_, p) in poi_map.items():
        if not p.get("waypointId") or not p.get("name"):
            continue
        wp = p["waypointId"]
        if wp not in wp_to_best or _poi_priority(p) > _poi_priority(wp_to_best[wp]):
            wp_to_best[wp] = p

    def make_stop(p: dict, wp_id: str) -> dict:
        s = dict(p)
        s["level_name"] = levels[wp_level[wp_id]]["name"] if wp_id in wp_level else ""
        return s

    poi_stops: list[dict] = []
    level_changes: list[dict] = []
    seen_wps: set[str] = set()
    prev_level: int | None = None

    for i, wp_id in enumerate(wp_path):
        if wp_id in seen_wps:
            continue
        seen_wps.add(wp_id)

        cur_level = wp_level.get(wp_id)
        if cur_level is not None and prev_level is not None and cur_level != prev_level:
            level_changes.append({
                "from_level": levels[prev_level]["name"],
                "to_level": levels[cur_level]["name"],
            })
        if cur_level is not None:
            prev_level = cur_level

        is_first = (i == 0)
        is_last = (i == len(wp_path) - 1)

        if is_first:
            poi_stops.append(make_stop(start_entry[1], wp_id))
        elif is_last:
            poi_stops.append(make_stop(end_entry[1], wp_id))
        elif wp_id in wp_to_best:
            poi_stops.append(make_stop(wp_to_best[wp_id], wp_id))

    return {
        "wp_path": wp_path,
        "poi_stops": poi_stops,
        "distance": dist.get(end_wp, 0),
        "level_changes": level_changes,
        "segments": _build_segments(poi_stops),
    }


def _build_segments(poi_stops: list[dict]) -> list[dict]:
    """Group a flat poi_stops list into per-floor segments, in travel order.

    Each segment's last stop is the portal (elevator/escalator/stairs) the
    user takes to leave that floor — None on the final segment, since there's
    nowhere further to go. Segments always cover every stop in poi_stops;
    a single-level route yields exactly one segment with portal_out=None."""
    segments: list[dict] = []
    current: list[dict] = []
    current_level = poi_stops[0].get("level_name", "") if poi_stops else ""

    for s in poi_stops:
        lvl = s.get("level_name", "")
        if lvl != current_level and current:
            segments.append({"level_name": current_level, "stops": current, "portal_out": None})
            current = []
            current_level = lvl
        current.append(s)
    if current:
        segments.append({"level_name": current_level, "stops": current, "portal_out": None})

    for i, seg in enumerate(segments):
        if i < len(segments) - 1 and seg["stops"]:
            seg["portal_out"] = seg["stops"][-1]

    return segments


# ---------------------------------------------------------------------------
# Route formatting (agent-readable)
# ---------------------------------------------------------------------------

def format_route_speech(raw: dict) -> str:
    """Compact, LLM-readable route summary."""
    if not raw.get("found"):
        return "No route found."

    stops = raw.get("stops", [])
    mins = raw.get("estimated_minutes", 1)
    changes = raw.get("level_changes", [])

    def fmt_stop(s: dict) -> str:
        level = s.get("level", "")
        node_type = s.get("type", "")
        details = ", ".join(part for part in [node_type, level] if part)
        return f"{s['name']} ({details})" if details else s["name"]

    path = " → ".join(f"Step {i+1}: {fmt_stop(s)}" for i, s in enumerate(stops))
    floor_note = " | floor change: " + ", ".join(
        f"{c['from_level']} → {c['to_level']}" for c in changes
    ) if changes else ""

    return f"~{mins} min | {path}{floor_note}"

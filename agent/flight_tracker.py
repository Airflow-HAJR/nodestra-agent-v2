"""
In-process flight subscription store and change detection.

Subscriptions are registered by the track_flight_changes agent tool and consumed
by the flight watch loop in server.py. No external HTTP dependency.
"""

import threading
from datetime import datetime

_lock = threading.Lock()
# airport_code → { normalized_flight_number → [phone_number, ...] }
_subscriptions: dict[str, dict[str, list[str]]] = {}

_MIN_TIME_DIFF_MINUTES = 5


def subscribe(airport: str, flight: str, phone: str) -> None:
    """Register a phone number to be called when a flight's gate/status changes."""
    airport, flight = airport.upper(), _norm(flight)
    with _lock:
        phones = _subscriptions.setdefault(airport, {}).setdefault(flight, [])
        if phone not in phones:
            phones.append(phone)


def unsubscribe_flight(airport: str, flight: str) -> None:
    """Remove all subscribers for a flight (e.g. after it departs)."""
    airport, flight = airport.upper(), _norm(flight)
    with _lock:
        _subscriptions.get(airport, {}).pop(flight, None)


def get_tracked() -> dict[str, list[str]]:
    """Return {airport: [flight, ...]} for all airports with active subscriptions."""
    with _lock:
        return {
            airport: list(flights.keys())
            for airport, flights in _subscriptions.items()
            if flights
        }


def get_subscribers(airport: str, flight: str) -> list[str]:
    airport, flight = airport.upper(), _norm(flight)
    with _lock:
        return list(_subscriptions.get(airport, {}).get(flight, []))


def find_significant_changes(prev: dict, curr: dict) -> list[dict]:
    """Return list of significant change dicts between two flight data snapshots."""
    changes = []
    for field in ("gate", "terminal", "status", "scheduled_time", "actual_time"):
        old_v = (prev.get(field) or "").strip()
        new_v = (curr.get(field) or "").strip()
        if old_v == new_v or not old_v or old_v == "-":
            continue

        if field in ("gate", "terminal"):
            changes.append({"field": field, "old_value": old_v, "new_value": new_v})
        elif field == "status":
            lowered = new_v.lower()
            if any(kw in lowered for kw in ("cancel", "delay", "late", "divert")):
                changes.append({"field": field, "old_value": old_v, "new_value": new_v})
        elif field in ("scheduled_time", "actual_time"):
            if _time_diff_minutes(old_v, new_v) >= _MIN_TIME_DIFF_MINUTES:
                changes.append({"field": "departure_time", "old_value": old_v, "new_value": new_v})
    return changes


def _norm(flight: str) -> str:
    return flight.strip().upper().replace(" ", "")


def _time_diff_minutes(t1: str, t2: str) -> int:
    for fmt in ("%I:%M %p", "%H:%M"):
        try:
            d1 = datetime.strptime(t1.strip(), fmt)
            d2 = datetime.strptime(t2.strip(), fmt)
            return int(abs((d2 - d1).total_seconds()) / 60)
        except ValueError:
            continue
    return 0

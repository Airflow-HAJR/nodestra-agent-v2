"""
Thread-safe shared state for GateGetter — per-airport.
"""

import threading

state_lock = threading.Lock()

# Keyed by uppercase IATA code → per-airport state dict
airports_state: dict[str, dict] = {}

shutdown_event = threading.Event()


def _empty_airport_state() -> dict:
    return {
        "tracked": [],
        "flights": [],
        "changes": [],
        "notifications": [],
        "last_poll": None,
        "next_poll": None,
        "poll_count": 0,
        "status": "starting",
        # flight_number → [phone_number, ...]
        "subscribers": {},
    }


def ensure_airport(code: str) -> None:
    """Create state entry for an airport if it doesn't exist."""
    code = code.upper()
    with state_lock:
        if code not in airports_state:
            airports_state[code] = _empty_airport_state()


def get_airport_state(code: str) -> dict:
    """Return state dict for an airport. Creates it if missing."""
    code = code.upper()
    ensure_airport(code)
    with state_lock:
        return dict(airports_state[code])


def remove_airport(code: str) -> None:
    """Remove an airport's state entirely."""
    code = code.upper()
    with state_lock:
        airports_state.pop(code, None)


def add_tracked(code: str, flight: str) -> None:
    code = code.upper()
    ensure_airport(code)
    with state_lock:
        if flight not in airports_state[code]["tracked"]:
            airports_state[code]["tracked"].append(flight)


def remove_tracked(code: str, flight: str) -> None:
    code = code.upper()
    ensure_airport(code)
    with state_lock:
        tracked = airports_state[code]["tracked"]
        if flight in tracked:
            tracked.remove(flight)


def add_subscriber(code: str, flight: str, phone: str) -> None:
    """Subscribe a phone number to changes for a specific tracked flight."""
    code = code.upper()
    ensure_airport(code)
    with state_lock:
        subs = airports_state[code].setdefault("subscribers", {})
        phones = subs.setdefault(flight, [])
        if phone not in phones:
            phones.append(phone)


def get_subscribers(code: str, flight: str) -> list:
    """Return all phone numbers subscribed to a flight's changes."""
    code = code.upper()
    with state_lock:
        return list(airports_state.get(code, {}).get("subscribers", {}).get(flight, []))

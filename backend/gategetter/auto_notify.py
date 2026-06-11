"""
Auto-notification: when significant flight changes are detected,
look up subscribed passengers from Supabase and trigger VAPI calls
via the Vercel backend.
"""

import os
import threading
from datetime import datetime
from state import airports_state, state_lock

# When co-deployed with the API, use localhost on the same Railway port
_default_backend = f"http://localhost:{os.environ.get('PORT', '8000')}" if os.environ.get("PORT") else "https://airflowbackendv2-production.up.railway.app"
BACKEND_URL = os.environ.get("AIRFLOW_BACKEND_URL", _default_backend)

# Fields that warrant a notification call
SIGNIFICANT_FIELDS = {"gate", "status", "terminal", "scheduled_time", "actual_time"}

# Minimum time difference (in minutes) to trigger a call for time changes
MIN_TIME_CHANGE_MINUTES = 5


def _is_significant_change(change: dict) -> bool:
    """Determine if a change is severe enough to warrant a call."""
    field = change.get("field", "")
    if field not in SIGNIFICANT_FIELDS:
        return False

    old_val = change.get("old_value") or ""
    new_val = change.get("new_value") or ""

    # Gate/terminal/status changes are always significant
    if field in ("gate", "terminal", "status"):
        # Skip if going from empty to a value (initial population)
        if not old_val or old_val == "-":
            return False
        # For status: only notify on cancellation, delay, or diversion
        if field == "status":
            lowered = new_val.lower()
            return any(kw in lowered for kw in ("cancel", "delay", "late", "divert"))
        return True

    # Time changes: only significant if >= MIN_TIME_CHANGE_MINUTES apart
    if field in ("scheduled_time", "actual_time"):
        if not old_val or not new_val or old_val == "-" or new_val == "-":
            return False
        minutes_diff = _estimate_time_diff_minutes(old_val, new_val)
        return minutes_diff is not None and minutes_diff >= MIN_TIME_CHANGE_MINUTES

    return False


def _estimate_time_diff_minutes(old_time: str, new_time: str) -> int | None:
    """Estimate the difference in minutes between two time strings like '3:45 PM'."""
    from datetime import datetime

    for fmt in ("%I:%M %p", "%H:%M"):
        try:
            t1 = datetime.strptime(old_time.strip(), fmt)
            t2 = datetime.strptime(new_time.strip(), fmt)
            diff = abs((t2 - t1).total_seconds()) / 60
            return int(diff)
        except ValueError:
            continue
    return None


def _get_subscribers(flight_number: str, airport_code: str) -> list[str]:
    """Look up phone numbers of passengers subscribed to a flight.

    Checks local in-memory state first (registered via the voice agent's
    track_flight_changes tool), then falls back to Supabase flight_passengers.
    """
    from state import get_subscribers
    local_phones = get_subscribers(airport_code, flight_number)

    return local_phones


def _trigger_call(phone_number: str, airport_code: str, flight_number: str, changes: list[dict]):
    """Trigger an outbound agent call via the main server's /flight-change-call endpoint."""
    import urllib.request
    import json

    payload = {
        "phone": phone_number,
        "flight": flight_number,
        "airport": airport_code,
        "changes": [
            {
                "field": _normalize_field(c["field"]),
                "old_value": c.get("old_value") or "",
                "new_value": c.get("new_value") or "",
            }
            for c in changes
        ],
    }

    try:
        req = urllib.request.Request(
            f"{BACKEND_URL}/flight-change-call",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
            call_id = result.get("call_sid", result.get("status", "ok"))
            print(f"  [{airport_code}] Outbound call triggered for {phone_number} on {flight_number}: {call_id}")

            # Add to changes log so it shows in the UI sidebar
            change_summary = ", ".join(c.get("detail", f"{c['field']}: {c.get('old_value')} → {c.get('new_value')}") for c in changes)
            with state_lock:
                if airport_code in airports_state:
                    airports_state[airport_code]["changes"].insert(0, {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "flight": flight_number,
                        "type": "call_initiated",
                        "field": "call",
                        "old_value": None,
                        "new_value": phone_number,
                        "detail": f"📞 Auto-call to {phone_number} for {flight_number}: {change_summary}",
                    })
    except Exception as e:
        print(f"  [{airport_code}] Outbound call failed for {phone_number} on {flight_number}: {e}")
        # Log the failure in the changes feed too
        with state_lock:
            if airport_code in airports_state:
                airports_state[airport_code]["changes"].insert(0, {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "flight": flight_number,
                    "type": "call_failed",
                    "field": "call",
                    "old_value": None,
                    "new_value": phone_number,
                    "detail": f"❌ Auto-call failed for {flight_number} to {phone_number}: {e}",
                })


def _normalize_field(field: str) -> str:
    """Map internal field names to the API's expected field names."""
    mapping = {
        "scheduled_time": "scheduled",
        "actual_time": "scheduled",
        "gate": "gate",
        "terminal": "terminal",
        "status": "status",
    }
    return mapping.get(field, field)


def notify_subscribers_of_changes(changes: list[dict], airport_code: str):
    """
    Given a list of changes from the diffing module, filter to significant ones,
    group by flight, look up subscribers, and trigger VAPI calls.

    Runs in a background thread to avoid blocking the scraper loop.
    """
    # Group significant changes by flight
    significant_by_flight: dict[str, list[dict]] = {}
    for change in changes:
        if _is_significant_change(change):
            fn = change["flight"]
            significant_by_flight.setdefault(fn, []).append(change)

    if not significant_by_flight:
        return

    def _do_notify():
        for flight_number, flight_changes in significant_by_flight.items():
            change_desc = ", ".join(c["detail"] for c in flight_changes)
            print(f"  [{airport_code}] Significant changes for {flight_number}: {change_desc}")

            subscribers = _get_subscribers(flight_number, airport_code)
            if not subscribers:
                print(f"  [{airport_code}] No subscribers for {flight_number}, skipping calls")
                continue

            print(f"  [{airport_code}] Notifying {len(subscribers)} subscriber(s) for {flight_number}")
            for phone in subscribers:
                _trigger_call(phone, airport_code, flight_number, flight_changes)

    # Run in background thread so we don't block scraping
    t = threading.Thread(target=_do_notify, daemon=True, name=f"notify-{airport_code}")
    t.start()

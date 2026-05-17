"""
Change detection: compares two snapshots of flights and returns differences.
"""

from datetime import datetime


def diff_flights(prev, curr):
    """Compare all flights, return list of important changes."""
    changes = []
    now = datetime.now().strftime("%H:%M:%S")

    prev_by_fn = {f["flight_number"]: f for f in prev if f.get("flight_number")}
    curr_by_fn = {f["flight_number"]: f for f in curr if f.get("flight_number")}

    for fn in sorted(set(prev_by_fn) & set(curr_by_fn)):
        p, c = prev_by_fn[fn], curr_by_fn[fn]
        for field, label in [
            ("gate", "Gate"),
            ("status", "Status"),
            ("actual_time", "Departure time"),
            ("scheduled_time", "Scheduled time"),
            ("terminal", "Terminal"),
        ]:
            old_val = p.get(field)
            new_val = c.get(field)
            if old_val != new_val:
                changes.append({
                    "time": now,
                    "flight": fn,
                    "type": "changed",
                    "field": field,
                    "old_value": old_val,
                    "new_value": new_val,
                    "detail": f"{fn} {label}: {old_val or '-'} → {new_val or '-'}",
                })

    return changes


def is_cancel_status(status):
    return "cancel" in (status or "").lower()


def is_delay_status(status):
    lowered = (status or "").lower()
    return "delay" in lowered or "late" in lowered

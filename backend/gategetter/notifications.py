"""
Notification generation: creates SMS-style alerts for actionable flight changes.
"""

import random
from diffing import is_cancel_status, is_delay_status

_phone_cache = {}


def _phone_for(flight):
    if flight not in _phone_cache:
        area = random.choice(["415", "650", "510", "408", "628"])
        _phone_cache[flight] = f"({area}) {random.randint(200,999)}-{random.randint(1000,9999)}"
    return _phone_cache[flight]


def build_notifications(new_changes):
    """Create phone alerts only for actionable flight updates."""
    notifications = []
    time_changes = {}

    for change in new_changes:
        fn = change["flight"]
        field = change.get("field", "")

        if field in ("scheduled_time", "actual_time"):
            existing = time_changes.get(fn)
            if existing is None or field == "scheduled_time":
                time_changes[fn] = change
            continue

        if field == "gate":
            old_g = change.get("old_value") or "-"
            new_g = change.get("new_value") or "-"
            msg = f"GateGetter Alert: Your flight {fn} gate changed from {old_g} to {new_g}."
        elif field == "status":
            new_st = change.get("new_value") or ""
            if is_cancel_status(new_st):
                msg = f"GateGetter Alert: Your flight {fn} has been CANCELLED."
            elif is_delay_status(new_st):
                msg = f"GateGetter Alert: Your flight {fn} is now showing as delayed."
            else:
                continue
        else:
            continue

        notifications.append({
            "time": change["time"],
            "phone": _phone_for(fn),
            "flight": fn,
            "msg": msg,
        })

    for fn, change in time_changes.items():
        old_t = change.get("old_value") or "-"
        new_t = change.get("new_value") or "-"
        notifications.append({
            "time": change["time"],
            "phone": _phone_for(fn),
            "flight": fn,
            "msg": f"GateGetter Alert: Your flight {fn} departure time changed from {old_t} to {new_t}.",
        })

    return notifications

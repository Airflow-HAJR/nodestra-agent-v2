import json
import logging
import time

_log = logging.getLogger("agent.events")


def log_event(event: str, thread_id: str = "", **fields) -> None:
    """Emit a single-line JSON structured log record to stdout."""
    record = {"ts": time.time(), "event": event, "thread_id": thread_id, **fields}
    _log.info(json.dumps(record, default=str))

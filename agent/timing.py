"""Turn-level latency accumulator. Reset once per invoke(), read after.
Call-level accumulator persists across turns; reset with call_reset()."""

# Per-turn stats (reset each turn)
_stats: dict = {
    "llm_ms": 0.0, "llm_calls": 0,
    "tool_ms": 0.0, "tool_calls": 0,
    "memory_ms": 0.0, "memory_calls": 0,
    "moss_ms": 0.0, "moss_calls": 0,
    "tts_ms": 0.0, "tts_calls": 0,
}

# Call-level stats (reset once per call/session)
_call: dict = {
    "total_llm_ms": 0.0, "llm_calls": 0,
    "total_tool_ms": 0.0, "tool_calls": 0,
    "total_tts_ms": 0.0,
    "turn_count": 0,
}


def reset():
    """Flush per-turn stats into call-level accumulator, then reset turn stats."""
    _call["total_llm_ms"] += _stats["llm_ms"]
    _call["llm_calls"] += _stats["llm_calls"]
    _call["total_tool_ms"] += _stats["tool_ms"]
    _call["tool_calls"] += _stats["tool_calls"]
    _call["total_tts_ms"] += _stats["tts_ms"]
    _call["turn_count"] += 1

    _stats.update({
        "llm_ms": 0.0, "llm_calls": 0,
        "tool_ms": 0.0, "tool_calls": 0,
        "memory_ms": 0.0, "memory_calls": 0,
        "moss_ms": 0.0, "moss_calls": 0,
        "tts_ms": 0.0, "tts_calls": 0,
    })
    import agent.graph as _graph
    _graph.reset_turn_state()


def call_reset():
    """Clear call-level accumulator at the start of a new call/session."""
    _call.update({
        "total_llm_ms": 0.0, "llm_calls": 0,
        "total_tool_ms": 0.0, "tool_calls": 0,
        "total_tts_ms": 0.0,
        "turn_count": 0,
    })


def turn_snapshot(total_seconds: float) -> dict:
    """Return current turn's stats (call after invoke, before next reset)."""
    total_ms = total_seconds * 1000
    return {
        "llm_ms": _stats["llm_ms"],
        "llm_calls": _stats["llm_calls"],
        "tool_ms": _stats["tool_ms"],
        "tool_calls": _stats["tool_calls"],
        "tts_ms": _stats["tts_ms"],
        "total_ms": total_ms,
        "other_ms": max(0.0, total_ms - _stats["llm_ms"] - _stats["tool_ms"] - _stats["tts_ms"]),
    }


def call_snapshot() -> dict:
    """Return call-level stats accumulated so far (includes current in-flight turn)."""
    return {
        "total_llm_ms": _call["total_llm_ms"] + _stats["llm_ms"],
        "llm_calls": _call["llm_calls"] + _stats["llm_calls"],
        "total_tool_ms": _call["total_tool_ms"] + _stats["tool_ms"],
        "tool_calls": _call["tool_calls"] + _stats["tool_calls"],
        "total_tts_ms": _call["total_tts_ms"] + _stats["tts_ms"],
        "turn_count": _call["turn_count"],
    }


def add_llm(seconds: float):
    _stats["llm_ms"] += seconds * 1000
    _stats["llm_calls"] += 1


def add_tool(seconds: float, count: int = 1):
    _stats["tool_ms"] += seconds * 1000
    _stats["tool_calls"] += count


def add_memory(seconds: float):
    _stats["memory_ms"] += seconds * 1000
    _stats["memory_calls"] += 1


def add_moss(seconds: float):
    _stats["moss_ms"] += seconds * 1000
    _stats["moss_calls"] += 1


def add_tts(seconds: float):
    _stats["tts_ms"] += seconds * 1000
    _stats["tts_calls"] += 1


def _fmt(ms: float, calls: int) -> str:
    return f"{ms:.0f}ms×{calls}" if calls else "—"


def summary(total_seconds: float) -> str:
    total_ms = total_seconds * 1000
    llm_ms   = _stats["llm_ms"]
    tool_ms  = _stats["tool_ms"]
    mem_ms   = _stats["memory_ms"]
    moss_ms  = _stats["moss_ms"]
    tts_ms   = _stats["tts_ms"]
    other_ms = total_ms - llm_ms - tool_ms - mem_ms - moss_ms - tts_ms
    return (
        f"LATENCY  total={total_ms:.0f}ms"
        f"  llm={_fmt(llm_ms, _stats['llm_calls'])}"
        f"  tools={_fmt(tool_ms, _stats['tool_calls'])}"
        f"  memory={_fmt(mem_ms, _stats['memory_calls'])}"
        f"  moss={_fmt(moss_ms, _stats['moss_calls'])}"
        f"  tts={_fmt(tts_ms, _stats['tts_calls'])}"
        f"  other={other_ms:.0f}ms"
    )
